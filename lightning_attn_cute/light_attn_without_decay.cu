#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cute/tensor.hpp>

using namespace cute;

#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

#define PRINT_TENSOR(name, content) \
    print(name);             \
    print(" : ");            \
    print_tensor(content);          \
    print("\n");


namespace config {
using namespace cute;

// BLOCK 用于外层for循环, q, k, v三个矩阵每次切出来 BLOCK x d大小的矩阵加载到smem
template <typename T_, int kHeadDim_ = 64, int BLOCK_ = 64>
struct FlashConfig {
  using T = T_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int BLOCK = BLOCK_;

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int kMmaEURepeatM = 1; // 4 -> 1
  static constexpr int kMmaEURepeatN = 1;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{}); // 2 -> 1
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  using TiledMMA =
      decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
  static constexpr int kThreadNum = size(TiledMMA{});

};

}  // namespace config



//for i in range(NUM_BLOCK):
//        q = tl.load(Q_start + q_off, mask=block_off[:, None] < n, other=0.0).to(tl.float32)
//        k_t = tl.load(K_start + k_off, mask=block_off[None, :] < n, other=0.0).to(tl.float32)
//        v = tl.load(V_start + vo_off, mask=block_off[:, None] < n, other=0.0).to(tl.float32)
//        o_intra = tl.dot(tl.dot(q, k_t) * diag_decay, v)
//
//        o_inter = tl.dot(q, kv) * q_decay
//        o = o_intra + o_inter
//        tl.store(O_start + vo_off, o.to(O.dtype.element_ty), mask=block_off[:, None] < n)
//        new_kv = tl.dot(k_t * k_decay, v)
//        kv = kv * block_decay + new_kv
//
//        block_off += BLOCK


// TODO:
// 1. smem要怎么处理才能避免相互覆盖的问题
// 2. smem如何处理多stage
// 3. gmem到smem的copy似乎没有流水线
// 4. 给smem增加static check. 参考 https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md
template <typename config>
__global__ void flash_forward(const half_t* q, const half_t* k, const half_t* v, half_t* o, const int B, const int H, const int N) {
  using namespace cute;
  using TiledMMA = typename config::TiledMMA;


  constexpr int BLOCK = config::BLOCK;
  constexpr int kHeadDim = config::kHeadDim;


  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int bs_head_offset = bx * N * kHeadDim;
  // const int num_block = N / BLOCK;
  const int num_block = 1;


  Tensor Q = make_tensor(make_gmem_ptr<half_t>(q + bs_head_offset), make_shape(N, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d
  Tensor K = make_tensor(make_gmem_ptr<half_t>(k + bs_head_offset), make_shape(N, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d
  Tensor Vt = make_tensor(make_gmem_ptr<half_t>(v + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N

  TiledMMA mma;
  ThrMMA thr_mma = mma.get_slice(tx);

  if (thread0()) {
    PRINT("mma size", size(mma));
  }

  Tensor kv = make_tensor<half_t>(make_shape(Int<kHeadDim>{}, Int<kHeadDim>{})); // d x d
  cute::clear(kv);
  for (int block_id = 0; block_id < num_block; block_id++) {
    Tensor gQ = local_tile(Q, make_tile(Int<BLOCK>{}, Int<kHeadDim>{}), make_coord(block_id, 0)); //BLOCK x d
    Tensor gK = local_tile(K, make_tile(Int<BLOCK>{}, Int<kHeadDim>{}), make_coord(block_id, 0)); //BLOCK x d
    Tensor gVt = local_tile(Vt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); //d x BLOCK

    // compute q @ k.T BLOCK x BLOCK
    Tensor tAgQ = thr_mma.partition_A(gQ);
    Tensor tArQ = thr_mma.partition_fragment_A(gQ);
    Tensor tBgK = thr_mma.partition_B(gK);
    Tensor tBrK = thr_mma.partition_fragment_B(gK);

    cute::copy(tAgQ, tArQ);
    cute::copy(tBgK, tBrK);

    Tensor tCrS = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<BLOCK>{})); //BLOCK x BLOCK
    clear(tCrS);
//    if (thread0()) {
//      PRINT("tCrS", tCrS);
//      PRINT_TENSOR("tCrS", tCrS)
//    }

	__syncthreads();

    cute::gemm(mma, tArQ, tBrK, tCrS);

//    if (thread0()) {
//      PRINT_TENSOR("tCrS", tCrS);
//    }

    // 读入v 并且计算 o_intra = s @ v [BLOCK, BLOCK] @ [BLOCK, d] -> [BLOCK, d]
    // Tensor tArS = thr_mma.partition_fragment_A(tCrS);
    // Tensor tArS = partition_fragment_A(mma, make_shape(Int<BLOCK>{}, Int<BLOCK>{}));
    // Tensor tArS = thr_mma.partition_A(tCrS);
    // cute::copy(tCrS, tArS);
    Tensor tArS = thr_mma.partition_A(make_tensor(tCrS.data(), make_shape(Int<BLOCK>{}, Int<BLOCK>{})));
    tArS(0) = 0;
    if (thread0()) {
      // PRINT_TENSOR("tCrS after partition fragment", tCrS);
      // PRINT_TENSOR("tArS after partition fragment", tArS);
      PRINT_TENSOR("tCrS", tCrS);
      PRINT_TENSOR("tArS", tArS);
    }

//	Tensor tBgVt = thr_mma.partition_B(gVt);
//    Tensor tBrVt = thr_mma.partition_fragment_B(gVt);
//    Tensor tCrO_intra = thr_mma.partition_fragment_C(make_shape(Int<BLOCK>{}, Int<kHeadDim>{})); //BLOCK x d
//    cute::clear(tCrO_intra);
//    cute::gemm(tArS, tBrVt, tCrO_intra);
//
//
//    // 计算 o_inter = tl.dot(q, kv)

  }

}




// q [B, H, N, d] k  [B, H, N, d] v [B, H, N, d]
torch::Tensor forward_without_decay(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  int B = q.size(0);
  int H = q.size(1);
  int N = q.size(2);
  int d = q.size(3);

  auto out = torch::empty_like(q);

  // only for head_dim=64
  config::FlashConfig<cute::half_t> config;
  dim3 block = config.kThreadNum;
  dim3 grid(B * H);
  auto partition_kernel = flash_forward<decltype(config)>;
  PRINT("grid", grid);
  PRINT("block", block);

  partition_kernel<<<grid, block>>>((cute::half_t*)q.data_ptr(), (cute::half_t*)k.data_ptr(),
                                              (cute::half_t*)v.data_ptr(), (cute::half_t*)out.data_ptr(), B, H, N);
  return out;
}