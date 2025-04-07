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

template <typename Thr_MMA, typename config>
__forceinline__ __device__ auto load_decay_tensor_q(const half_t* data_ptr, Thr_MMA thr_mma, const int head_id) {
    using namespace cute;
    constexpr int BLOCK = config::BLOCK;
    constexpr int kHeadDim = config::kHeadDim;
    Tensor g_decay = make_tensor(make_gmem_ptr<half_t>(data_ptr + head_id * BLOCK * kHeadDim), make_shape(Int<BLOCK>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor r_decay = thr_mma.partition_fragment_A(g_decay);
    copy(g_decay, r_decay);
    return r_decay;
}

template <typename Thr_MMA, typename config>
__forceinline__ __device__ auto load_decay_tensor_k(const half_t* data_ptr, Thr_MMA thr_mma, const int head_id) {
    using namespace cute;
    constexpr int BLOCK = config::BLOCK;
    constexpr int kHeadDim = config::kHeadDim;
    Tensor g_decay = make_tensor(make_gmem_ptr<half_t>(data_ptr + head_id * BLOCK * kHeadDim), make_shape(Int<BLOCK>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor r_decay = thr_mma.partition_fragment_B(g_decay);
    copy(g_decay, r_decay);
    return r_decay;
}

template <typename Thr_MMA, typename config>
__forceinline__ __device__ auto load_decay_tensor_kt(const half_t* data_ptr, Thr_MMA thr_mma, const int head_id) {
    using namespace cute;
    constexpr int BLOCK = config::BLOCK;
    constexpr int kHeadDim = config::kHeadDim;
    Tensor g_decay = make_tensor(make_gmem_ptr<half_t>(data_ptr + head_id * BLOCK * kHeadDim), make_shape(Int<BLOCK>{}, Int<kHeadDim>{}), make_stride(Int<1>{}, Int<kHeadDim>{}));
    Tensor r_decay = thr_mma.partition_fragment_B(g_decay);
    copy(g_decay, r_decay);
    return r_decay;
}


template <typename Thr_MMA, typename config>
__forceinline__ __device__ auto load_decay_tensor_diag_block(const half_t* data_ptr, Thr_MMA thr_mma, const int head_id) {
    using namespace cute;
    constexpr int BLOCK = config::BLOCK;
    constexpr int kHeadDim = config::kHeadDim;
    Tensor g_decay = make_tensor(make_gmem_ptr<half_t>(data_ptr + head_id * BLOCK * kHeadDim), make_shape(Int<BLOCK>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor r_decay = thr_mma.partition_fragment_C(g_decay);
    copy(g_decay, r_decay);
    return r_decay;
}

// TODO:
// 1. smem要怎么处理才能避免相互覆盖的问题
// 2. smem如何处理多stage
// 3. gmem到smem的copy似乎没有流水线
// 4. 给smem增加static check. 参考 https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md
template <typename config>
__global__ void flash_forward(const half_t* q, const half_t* k, const half_t* v, half_t* o,
                              const half_t* q_decay, const half_t* k_decay, const half_t* diag_decay, const half_t* block_decay,
                              const int B, const int H, const int N) {
  using namespace cute;
  using TiledMMA = typename config::TiledMMA;


  constexpr int BLOCK = config::BLOCK;
  constexpr int kHeadDim = config::kHeadDim;


  const int bx = blockIdx.x;
  const int head_id = bx % H;
  const int tx = threadIdx.x;
  const int bs_head_offset = bx * N * kHeadDim;
  const int num_block = (N + BLOCK - 1) / BLOCK;

  __shared__ float smem_kv[kHeadDim * kHeadDim];
//  for (int i = tx; i < kHeadDim * kHeadDim; i += blockDim.x) {
//    smem_kv[i] = __float2half(0.0f);
//  }
//  __syncthreads();

  Tensor Q = make_tensor(make_gmem_ptr<half_t>(q + bs_head_offset), make_shape(N, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d
  Tensor K = make_tensor(make_gmem_ptr<half_t>(k + bs_head_offset), make_shape(N, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d
  Tensor Kt = make_tensor(make_gmem_ptr<half_t>(k + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N
  Tensor Vt = make_tensor(make_gmem_ptr<half_t>(v + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N
  Tensor O = make_tensor(make_gmem_ptr<half_t>(o + bs_head_offset), make_shape(N, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d

  Tensor sKV = make_tensor(make_smem_ptr<float>(&smem_kv), make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{},Int<1>{}));
  Tensor sKVt = make_tensor(make_smem_ptr<float>(&smem_kv), make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}), make_stride(Int<1>{}, Int<kHeadDim>{}));


  TiledMMA mma;
  ThrMMA thr_mma = mma.get_slice(tx);

  Tensor tBsKVt = thr_mma.partition_B(sKVt);
  clear(tBsKVt);

  if (thread0()) {
    PRINT("mma size", size(mma));
    PRINT("num_block", num_block);
  }


  // load decay tensors
  Tensor q_decay_r = load_decay_tensor_q<decltype(thr_mma), config>(q_decay, thr_mma, head_id);
  Tensor k_decay_r = load_decay_tensor_k<decltype(thr_mma), config>(k_decay, thr_mma, head_id);
  Tensor diag_decay_r = load_decay_tensor_diag_block<decltype(thr_mma), config>(diag_decay, thr_mma, head_id);
  Tensor block_decay_r = load_decay_tensor_diag_block<decltype(thr_mma), config>(block_decay, thr_mma, head_id);

  if (thread0()) {
    PRINT("q_decay_r", q_decay_r);
    PRINT("k_decay_r", k_decay_r);
    PRINT("diag_decay_r", diag_decay_r);
    PRINT("block_decay_r", block_decay_r);
  }


  auto multiply_op = [] (auto a, auto b) {
      return a * b;
  };

  auto elementwise_add_op = [] (auto a, auto b) {
    return a + b;
  }

  for (int block_id = 0; block_id < num_block; block_id++) {
    Tensor gQ = local_tile(Q, make_tile(Int<BLOCK>{}, Int<kHeadDim>{}), make_coord(block_id, 0)); //BLOCK x d
    Tensor gK = local_tile(K, make_tile(Int<BLOCK>{}, Int<kHeadDim>{}), make_coord(block_id, 0)); //BLOCK x d
    Tensor gKt = local_tile(Kt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); // d x BLOCK
    Tensor gVt = local_tile(Vt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); //d x BLOCK
    Tensor gO = local_tile(O, make_tile(Int<BLOCK>{}, Int<kHeadDim>{}), make_coord(block_id, 0));


    // compute q @ k.T BLOCK x BLOCK
    Tensor tAgQ = thr_mma.partition_A(gQ);
    Tensor tArQ = thr_mma.partition_fragment_A(gQ);
    Tensor tBgK = thr_mma.partition_B(gK);
    Tensor tBrK = thr_mma.partition_fragment_B(gK);
    cute::copy(tAgQ, tArQ);
    cute::copy(tBgK, tBrK);

    // multiply q, k decay
    if (thread0()) {
      PRINT_TENSOR("tArQ before", tArQ);
    }
    Tensor tArQ_decay = make_tensor_like(tArQ);
    Tensor tBrK_decay = make_tensor_like(tBrK);
    clear(tArQ_decay);
    clear(tBrK_decay);
    assert(q_decay_r.layout() == tArQ.layout());
    assert(k_decay_r.layout() == tBrK.layout());
    cute::transform(q_decay_r, tArQ, tArQ_decay, multiply_op);
    cute::transform(k_decay_r, tBrK, tBrK_decay, multiply_op);

    if (thread0()) {
      PRINT_TENSOR("tArQ after", tArQ);
    }

    Tensor tCrS = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<BLOCK>{}));
    clear(tCrS);



    cute::gemm(mma, tArQ, tBrK, tCrS);

    auto tCrS_fp16 = fp32_to_fp16(tCrS);

    // compute

    // 将tCrS_f16转换为A layout，并且进行第二个gemm的计算
    // ((_2,_2),_4,_8) -> ((_2,_2),_4, (2, 4)) ->  -> ((2, 2, 2), 4, 4)
    auto l = logical_divide(tCrS_fp16.layout(), Shape<X, X, Int<2>>{});
    auto tOrS_laytout = make_layout(make_layout(get<0, 0>(l), get<0, 1>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    Tensor tOrS = make_tensor(tCrS_fp16.data(), tOrS_laytout);

	Tensor tOgVt = thr_mma.partition_B(gVt);
    Tensor tOrVt = thr_mma.partition_fragment_B(gVt);
    cute::copy(tOgVt, tOrVt);

    Tensor tOrO_intra = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<kHeadDim>{})); //BLOCK x d
    cute::clear(tOrO_intra);

    cute::gemm(mma, tOrS, tOrVt, tOrO_intra);



    // 计算 o_inter = q @ kv -> BLOCK x d @ d x d = BLOCK x d

    Tensor tCrO_inter = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<kHeadDim>{}));
    cute::clear(tCrO_inter);

    Tensor tBrKVt = thr_mma.partition_fragment_B(sKVt);

    cute::copy(tBsKVt, tBrKVt);

//    if (thread0()) {
//      PRINT("tBrKVt", tBrKVt);
//    }

    cute::gemm(mma, tArQ, tBrKVt, tCrO_inter);

//    if (thread0()) {
//      PRINT_TENSOR("tCrO_inter", tCrO_inter(_, 0, 0));
//    }

    // TODO: 需要确认torch的矩阵加法的acc是fp16还是fp32
    // O = O_intra + O_inter
    Tensor tCrO_inter_f16 = fp32_to_fp16(tCrO_inter);
    Tensor tOrO_intra_f16 = fp32_to_fp16(tOrO_intra);
    Tensor tOrO_f16 = make_tensor_like(tOrO_intra_f16);

//    halt_t one = half_t(1.0f);
//    cute::axpby(one, tOrO_intra_f16, one, tCrO_inter_f16);
    cute::transform(tOrO_intra_f16, tCrO_inter_f16, tOrO_f16, elementwise_add_op)
    // write O to global memory
    Tensor tCgO = thr_mma.partition_C(gO);
    cute::copy(tOrO_f16, tCgO);

    // Update KV
    // new_kv = tl.dot(k_t, v) d x BLOCK @ d x BLOCK = d x  d
    // kv = kv * block_decay + new_kv, block_decay = 1.0
    Tensor tAgKt = thr_mma.partition_A(gKt);
    Tensor tArKt = thr_mma.partition_fragment_A(gKt);
    Tensor tBgVt = thr_mma.partition_B(gVt);
    Tensor tBrVt = thr_mma.partition_fragment_B(gVt);

    cute::copy(tAgKt, tArKt);
    cute::copy(tBgVt, tBrVt);

    Tensor tCrNewKV = thr_mma.partition_fragment_C(sKV);
    Tensor tCsKV = thr_mma.partition_C(sKV);
    clear(tCrNewKV);
    cute::gemm(mma, tArKt, tBrVt, tCrNewKV);
    cute::axpby(1.0, tCrNewKV, 1.0, tCsKV);
  }

}




// q [B, H, N, d] k  [B, H, N, d] v [B, H, N, d]
// q_decay,k_decay,diag_decay, block_decay [H, BLOCK, d]


torch::Tensor forward_with_decay(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                torch::Tensor q_decay, torch::Tensor k_decay, torch::Tensor diag_decay, torch::Tensor block_decay) {
  int B = q.size(0);
  int H = q.size(1);
  int N = q.size(2);
  int d = q.size(3);

  int BLOCK = 64;
  int num_block = (N + BLOCK - 1) / BLOCK;

  // auto kv_out = torch.zeros((num_block, d, d), device=q.device, dtype=q.dtype);

  auto out = torch::empty_like(q);

  // only for head_dim=64
  config::FlashConfig<cute::half_t> config;
  dim3 block = config.kThreadNum;
  dim3 grid(B * H);
  auto partition_kernel = flash_forward<decltype(config)>;
  PRINT("grid", grid);
  PRINT("block", block);

  partition_kernel<<<grid, block>>>((cute::half_t*) q.data_ptr(), (cute::half_t*) k.data_ptr(),
                                              (cute::half_t*) v.data_ptr(), (cute::half_t*) out.data_ptr(),
                                            (cute::half_t*)q_decay.data_ptr(),
                                            (cute::half_t*)k_decay.data_ptr(),
                                            (cute::half_t*)diag_decay.data_ptr(),
                                            (cute::half_t*)block_decay.data_ptr(),
                                            B, H, N);
  return out;
}