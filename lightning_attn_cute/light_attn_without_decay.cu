#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

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


template<typename Tensor>
__forceinline__ __device__ auto fp32_to_fp16(Tensor& src_fp32) {
  using namespace cute;
  auto dest_fp16 = make_tensor_like<half_t>(src_fp32);
  auto src_fp32x2 = recast<float2>(src_fp32);
  auto dest_fp16x2 = recast<half2>(dest_fp16);
#pragma unroll
  for (int si = 0; si < size(dest_fp16x2); si++) {
    dest_fp16x2(si) = __float22half2_rn(src_fp32x2(si));
  }
  return dest_fp16;
}

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
  Tensor Kt = make_tensor(make_gmem_ptr<half_t>(k + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N
  Tensor Vt = make_tensor(make_gmem_ptr<half_t>(v + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N
  Tensor O = make_tensor(make_gmem_ptr<half_t>(o + bs_head_offset), make_shape(N, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d


  TiledMMA mma;
  ThrMMA thr_mma = mma.get_slice(tx);

  if (thread0()) {
    PRINT("mma size", size(mma));
  }

  Tensor tCrKV = partition_fragment_C(mma, make_shape(Int<kHeadDim>{}, Int<kHeadDim>{})); //d x d

  for (int block_id = 0; block_id < num_block; block_id++) {
    Tensor gQ = local_tile(Q, make_tile(Int<BLOCK>{}, Int<kHeadDim>{}), make_coord(block_id, 0)); //BLOCK x d
    Tensor gK = local_tile(K, make_tile(Int<BLOCK>{}, Int<kHeadDim>{}), make_coord(block_id, 0)); //BLOCK x d
    Tensor gKt = local_tile(Kt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); //BLOCK x d
    Tensor gVt = local_tile(Vt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); //d x BLOCK
    Tensor gO = local_tile(O, make_tile(Int<BLOCK>{}, Int<kHeadDim>{}), make_coord(block_id, 0));


    // compute q @ k.T BLOCK x BLOCK
    Tensor tAgQ = thr_mma.partition_A(gQ);
    Tensor tArQ = thr_mma.partition_fragment_A(gQ);
    Tensor tBgK = thr_mma.partition_B(gK);
    Tensor tBrK = thr_mma.partition_fragment_B(gK);
    cute::copy(tAgQ, tArQ);
    cute::copy(tBgK, tBrK);
    Tensor tCrS = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<BLOCK>{}));
    clear(tCrS);


	__syncthreads();

    if (thread0()) {
      PRINT("tArQ size", size(tArQ));
      PRINT("tBrK size", size(tBrK));
      PRINT("tCrS size", size(tCrS));
    }

    cute::gemm(mma, tArQ, tBrK, tCrS);


    if (thread0()) {
      PRINT("tAgQ", tAgQ);
      PRINT("tArQ", tArQ);
      PRINT("tBgK", tBgK);
      PRINT("tBrK", tBrK);
      // PRINT_TENSOR("tCrS tensor", tCrS);
    }
//
//    if (thread0()) {
//      // PRINT_TENSOR("tArQ tensor", tArQ);
//      PRINT_TENSOR("tCrS tensor", tCrS);
//    }
    // ((_2,_2),_4,_8):((_1,_2),_4,_16)

//    auto tCrS_fp16 = make_tensor_like<half_t>(tCrS);
//    auto tCrS_fp32x2 = recast<float2>(tCrS);
//    auto tCrS_fp16x2 = recast<half2>(tCrS_fp16);
//#pragma unroll
//    for (int si = 0; si < size(tCrS_fp16x2); si++) {
//      tCrS_fp16x2(si) = __float22half2_rn(tCrS_fp32x2(si));
//    }
    auto tCrS_fp16 = fp32_to_fp16(tCrS);

//    if (thread0()) {
//      PRINT_TENSOR("tCrS_fp16", tCrS_fp16);
//    }

    // 将tCrS_f16转换为A layout，并且进行第二个gemm的计算
    // ((_2,_2),_4,_8) -> ((_2,_2),_4, (2, 4)) ->  -> ((2, 2, 2), 4, 4)
    auto l = logical_divide(tCrS_fp16.layout(), Shape<X, X, Int<2>>{});
    auto tOrS_laytout = make_layout(make_layout(get<0, 0>(l), get<0, 1>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    if (thread0()) {
      PRINT("l", l);
      PRINT("tOrS_laytout", tOrS_laytout);
    }
    Tensor tOrS = make_tensor(tCrS_fp16.data(), tOrS_laytout);
    if (thread0()) {
      PRINT("tOrS", tOrS);
      // PRINT_TENSOR("tOrS tensor", tOrS);
    }


	  Tensor tOgVt = thr_mma.partition_B(gVt);
    Tensor tOrVt = thr_mma.partition_fragment_B(gVt);
    cute::copy(tOgVt, tOrVt);

    if (thread0()){
      PRINT("tOrVt", tOrVt);
    }

    Tensor tOrO_intra = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<kHeadDim>{})); //BLOCK x d
    cute::clear(tOrO_intra);

    if (thread0()) {
      PRINT_TENSOR("tOrO_intra", tOrO_intra);
    }


    cute::gemm(mma, tOrS, tOrVt, tOrO_intra);

//    if (thread0()) {
//      PRINT_TENSOR("tOrO_intra", tOrO_intra);
//    }


    // 计算 o_inter = q @ kv -> BLOCK x d @ d x d = BLOCK x d

    Tensor tCrO_inter = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<kHeadDim>{}));
    cute::clear(tCrO_inter);

    auto tCrKV_fp16 = fp32_to_fp16(tCrKV);

    if (thread0()) {
      PRINT("tCrKV_fp16", tCrKV_fp16);
    }

    auto l2 = tCrKV_fp16.layout();
    auto tBrKV_fp16 = make_tensor(tCrKV_fp16.data(), make_layout(get<0>(l2), get<2>(l2), get<1>(l2)));

    cute::gemm(mma, tArQ, tBrKV_fp16, tCrO_inter);



    // O = O_intra + O_inter
    cute::axpby(1.0, tOrO_intra, 1.0, tCrO_inter);
    if (thread0()) {
      PRINT_TENSOR("tCrO_inter", tCrO_inter);
    }

//
//    // write O to global memory
//    Tensor tCgO = thr_mma.partition_C(gO);
//    cute::copy(tCrO_inter, tCgO);
//    __syncthreads();
//
//    // Update KV
//    // new_kv = tl.dot(k_t, v) d x BLOCK @ d x BLOCK = d x  d
//    // kv = kv * block_decay + new_kv, block_decay = 1.0
//    Tensor tAgKt = thr_mma.partition_A(gKt);
//    Tensor tArKt = thr_mma.partition_fragment_A(gKt);
//    cute::copy(tAgKt, tArKt);
//
//    Tensor tCrNewKV = thr_mma.partition_fragment_C(make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}));
//    clear(tCrNewKV);
//    cute::gemm(tArKt, tBrVt, tCrNewKV);
//    Tensor tCsKV = thr_mma.partition_C(sKV);
//
//    cute::axpby(1.0, tCrNewKV, 1.0, tCsKV);
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