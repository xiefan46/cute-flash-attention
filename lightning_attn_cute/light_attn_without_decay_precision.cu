#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <tuple>

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
__global__ void flash_forward(const half_t* q, const half_t* k, const half_t* v, half_t* o, const int B, const int H, const int N, float* kv_out,
                              float* o_inter_out, float* o_intra_out) {
  using namespace cute;
  using TiledMMA = typename config::TiledMMA;


  constexpr int BLOCK = config::BLOCK;
  constexpr int kHeadDim = config::kHeadDim;


  const int bx = blockIdx.x;
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
    Tensor tCrS = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<BLOCK>{}));
    clear(tCrS);


	__syncthreads();

    cute::gemm(mma, tArQ, tBrK, tCrS);

    auto tCrS_fp16 = fp32_to_fp16(tCrS);

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

    // output debug info
    Tensor O_intra = make_tensor(make_gmem_ptr<float>(o_intra_out + block_id * BLOCK * kHeadDim),
                             make_shape(Int<BLOCK>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // d x d
    Tensor gO_intra = thr_mma.partition_C(O_intra);
    cute::copy(tOrO_intra, gO_intra);

    // 计算 o_inter = q @ kv -> BLOCK x d @ d x d = BLOCK x d
    Tensor tCrO_inter = partition_fragment_C(mma, make_shape(Int<BLOCK>{}, Int<kHeadDim>{}));
    cute::clear(tCrO_inter);

    Tensor tBrKVt = thr_mma.partition_fragment_B(sKVt);

    cute::copy(tBsKVt, tBrKVt);

//    if (thread0()) {
//      PRINT("tBrKVt", tBrKVt);
//    }

    cute::gemm(mma, tArQ, tBrKVt, tCrO_inter);

    // output debug info
    Tensor O_inter = make_tensor(make_gmem_ptr<float>(o_inter_out + block_id * BLOCK * kHeadDim),
                             make_shape(Int<BLOCK>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // d x d
    Tensor gO_inter = thr_mma.partition_C(O_inter);
    cute::copy(tCrO_inter, gO_inter);


//    if (thread0()) {
//      PRINT_TENSOR("tCrO_inter", tCrO_inter(_, 0, 0));
//    }

    // O = O_intra + O_inter
    cute::axpby(1.0, tOrO_intra, 1.0, tCrO_inter);
    // write O to global memory
    Tensor tCgO = thr_mma.partition_C(gO);
    cute::copy(tCrO_inter, tCgO);
    __syncthreads();

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


    Tensor gKV = make_tensor(make_gmem_ptr<float>(kv_out + block_id * kHeadDim * kHeadDim),
                             make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // d x d

    Tensor tCgKV = thr_mma.partition_C(gKV);
    // copy kv result to global
    cute::copy(tCsKV, tCgKV);

  }

}




// q [B, H, N, d] k  [B, H, N, d] v [B, H, N, d]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_without_decay_precision(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  int B = q.size(0);
  int H = q.size(1);
  int N = q.size(2);
  int d = q.size(3);

  int BLOCK = 64;
  int num_block = (N + BLOCK - 1) / BLOCK;

  PRINT("num_block", num_block);

  auto kv_out = torch::zeros({num_block, d, d}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, 0)));

  auto o_inter_out = torch::zeros({num_block, BLOCK, d}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, 0)));
  auto o_intra_out = torch::zeros({num_block, BLOCK, d}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, 0)));

  auto out = torch::empty_like(q);

  // only for head_dim=64
  config::FlashConfig<cute::half_t> config;
  dim3 block = config.kThreadNum;
  dim3 grid(B * H);
  auto partition_kernel = flash_forward<decltype(config)>;
  PRINT("grid", grid);
  PRINT("block", block);

  partition_kernel<<<grid, block>>>((cute::half_t*)q.data_ptr(), (cute::half_t*)k.data_ptr(),
                                              (cute::half_t*)v.data_ptr(), (cute::half_t*)out.data_ptr(), B, H, N, (float*)kv_out.data_ptr(),
                                    (float*)o_inter_out.data_ptr(), (float*)o_intra_out.data_ptr());

  cudaDeviceSynchronize();
  return std::make_tuple(out, kv_out, o_inter_out, o_intra_out);
}



template <typename config>
__global__ void compute_kv_kernel(const half_t* k, const half_t* v, float* kv_out, const int B, const int H, const int N)
{
  using namespace cute;
  using TiledMMA = typename config::TiledMMA;


  constexpr int BLOCK = config::BLOCK;
  constexpr int kHeadDim = config::kHeadDim;


  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int bs_head_offset = bx * N * kHeadDim;
  const int num_block = (N + BLOCK - 1) / BLOCK;

  __shared__ float smem_kv[kHeadDim * kHeadDim];

  Tensor K = make_tensor(make_gmem_ptr<half_t>(k + bs_head_offset), make_shape(N, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // N x d
  Tensor Kt = make_tensor(make_gmem_ptr<half_t>(k + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N
  Tensor Vt = make_tensor(make_gmem_ptr<half_t>(v + bs_head_offset), make_shape(Int<kHeadDim>{}, N), make_stride(Int<1>{}, Int<kHeadDim>{})); // d x N

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

  for (int block_id = 0; block_id < num_block; block_id++) {

    Tensor gKt = local_tile(Kt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); // d x BLOCK
    Tensor gVt = local_tile(Vt, make_tile(Int<kHeadDim>{}, Int<BLOCK>{}), make_coord(0, block_id)); //d x BLOCK
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

    Tensor gKV = make_tensor(make_gmem_ptr<float>(kv_out + block_id * kHeadDim * kHeadDim),
                             make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})); // d x d

    Tensor tCgKV = thr_mma.partition_C(gKV);
    // copy kv result to global
    cute::copy(tCsKV, tCgKV);

  }

}


torch::Tensor cute_compute_kv(torch::Tensor k, torch::Tensor v) {
    int B = k.size(0);
    int H = k.size(1);
    int N = k.size(2);
    int d = k.size(3);

    int BLOCK = 64;
    int num_block = (N + BLOCK - 1) / BLOCK;

    PRINT("num_block", num_block);

    auto kv_out = torch::zeros({num_block, d, d}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, 0)));


    // only for head_dim=64
    config::FlashConfig<cute::half_t> config;
    dim3 block = config.kThreadNum;
    dim3 grid(B * H);
    auto partition_kernel = compute_kv_kernel<decltype(config)>;
    PRINT("grid", grid);
    PRINT("block", block);

    partition_kernel<<<grid, block>>>((cute::half_t*)k.data_ptr(), (cute::half_t*)v.data_ptr(), (float*)kv_out.data_ptr(), B, H, N);
    cudaDeviceSynchronize();

    return kv_out;
}