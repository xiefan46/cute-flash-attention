#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <iostream>

#include <cute/tensor.hpp>

using namespace cute;

#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

template <typename config>
__global__ void flash_forward(void* output, const void* q, const void* k,
                              const void* v, int head_stride, int q_len,
                              int k_len, float sm_scale) {
  using namespace cute;
  using X = Underscore;
  const int m_block = blockIdx.x;
  const int base_id = blockIdx.y;
  const int tidx = threadIdx.x;

  using T = typename config::T;
  using SmemLayoutQ = typename config::SmemLayoutQ;
  using SmemLayoutK = typename config::SmemLayoutKV;
  using SmemLayoutV = typename config::SmemLayoutKV;
  using SmemLayoutO = typename config::SmemLayoutO;
  using SmemCopyAtom = typename config::SmemCopyAtom;
  using SmemCopyAtomO = typename config::SmemCopyAtomO;
  using GmemTiledCopyQKV = typename config::GmemTiledCopyQKV;
  using GmemTiledCopyO = typename config::GmemTiledCopyO;
  using SmemCopyAtomTransposed = typename config::SmemCopyAtomTransposed;
  using TiledMMA = typename config::TiledMMA;
  using SmemLayoutVt = typename config::SmemLayoutVtransposed;
  using SmemLayoutVtNoSwizzle = typename config::SmemLayoutVtransposedNoSwizzle;

  constexpr int kBlockM = config::kBlockM;
  constexpr int kBlockN = config::kBlockN;
  constexpr int kHeadDim = config::kHeadDim;

  extern __shared__ T shm_data[];
  auto q_shm = shm_data;
  auto k_shm = q_shm + cosize(SmemLayoutQ{});
  auto v_shm = k_shm + cosize(SmemLayoutK{});

  const int bs_head_offset = base_id * head_stride;


  if (thread0()) {
    PRINT("kBlockM", kBlockM);
    PRINT("kBlockN", kBlockN);
    PRINT("kHeadDim", kHeadDim);
    PRINT("head_stride", head_stride);
    PRINT("bs_head_offset", bs_head_offset);
    PRINT("SmemLayoutQ", SmemLayoutQ{});
    PRINT("SmemLayoutK", SmemLayoutK{});
    PRINT("SmemLayoutV", SmemLayoutV{});
    PRINT("SmemLayoutO", SmemLayoutO{});
    PRINT("SmemLayoutVt", SmemLayoutVt{});
    PRINT("SmemLayoutVtNoSwizzle", SmemLayoutVtNoSwizzle{});
    PRINT("size(SmemLayoutQ{})", size(SmemLayoutQ{}));
    PRINT("size(SmemLayoutK{})", size(SmemLayoutK{}));
    PRINT("cosize(SmemLayoutQ{})", cosize(SmemLayoutQ{}));
    PRINT("cosize(SmemLayoutK{})", cosize(SmemLayoutK{}));
}

  auto Q = make_tensor(make_gmem_ptr<half_t>((T*)q + bs_head_offset),
                       make_shape(q_len, Int<kHeadDim>{}),
                       make_stride(Int<kHeadDim>{}, Int<1>{}));
  auto K = make_tensor(make_gmem_ptr<half_t>((T*)k + bs_head_offset),
                       make_shape(k_len, Int<kHeadDim>{}),
                       make_stride(Int<kHeadDim>{}, Int<1>{}));
  auto V = make_tensor(make_gmem_ptr<half_t>((T*)v + bs_head_offset),
                       make_shape(k_len, Int<kHeadDim>{}),
                       make_stride(Int<kHeadDim>{}, Int<1>{}));
  auto O = make_tensor(make_gmem_ptr<half_t>((T*)output + bs_head_offset),
                       make_shape(q_len, Int<kHeadDim>{}),
                       make_stride(Int<kHeadDim>{}, Int<1>{}));
   if (thread0()) {
  PRINT("Q", Q);
  PRINT("K", K);
  PRINT("V", V);
  PRINT("O", O);
}

  auto gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}),
                       make_coord(m_block, _));
  auto gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}),
                       make_coord(0, _));
  auto gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}),
                       make_coord(0, _));

  auto sQ = make_tensor(make_smem_ptr<half_t>(q_shm), SmemLayoutQ{});
  auto sK = make_tensor(make_smem_ptr<half_t>(k_shm), SmemLayoutK{});
  auto sV = make_tensor(make_smem_ptr<half_t>(v_shm), SmemLayoutV{});


 if (thread0()) {
  PRINT("gQ", gQ);
  PRINT("gK", gK);
  PRINT("gV", gV);
}


  // Tensor for V Transpose; used in GEMM-II.
  auto sVt = make_tensor(make_smem_ptr<half_t>(v_shm), SmemLayoutVt{});
  auto sVtNoSwizzle =
      make_tensor(make_smem_ptr<half_t>(v_shm), SmemLayoutVtNoSwizzle{});


  if (thread0()) {
  PRINT("sQ", sQ);
  PRINT("sK", sK);
  PRINT("sV", sV);
  PRINT("sVt", sVt);
  PRINT("sVtNoSwizzle", sVtNoSwizzle);
}

  GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  auto tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  auto tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  auto tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  auto tKsK = gmem_thr_copy_QKV.partition_D(sK);
  auto tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  auto tVsV = gmem_thr_copy_QKV.partition_D(sV);


  if (thread0()) {
  PRINT("tQgQ", tQgQ);
  PRINT("size tQgQ", size(tQgQ));
  PRINT("tQsQ", tQsQ);
  PRINT("size tQsQ", size(tQsQ));
  PRINT("tKgK", tKgK);
  PRINT("size tKgK", size(tKgK));
  PRINT("tKsK", tKsK);
  PRINT("size tKsK", size(tKsK));
  PRINT("tVgV", tVgV);
  PRINT("size tVgV", size(tVgV));
  PRINT("tVsV", tVsV);
  PRINT("size tVsV", size(tVsV));
}

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);
  auto tSrQ = thr_mma.partition_fragment_A(sQ);             // (MMA,MMA_M,MMA_K)
  auto tSrK = thr_mma.partition_fragment_B(sK);             // (MMA,MMA_N,MMA_K)
  auto tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA,MMA_K,MMA_N)


  if (thread0()) {
  PRINT("tSrQ", tSrQ);
  PRINT("size tSrQ", size(tSrQ));
  PRINT("tSrK", tSrK);
  PRINT("size tSrK", size(tSrK));
  PRINT("tOrVt", tOrVt);
  PRINT("size tOrVt", size(tOrVt));
}


  auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
  auto tSrQ_view = smem_thr_copy_Q.retile_D(tSrQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  auto tSsK = smem_thr_copy_K.partition_S(sK);
  auto tSrK_view = smem_thr_copy_K.retile_D(tSrK);

  auto smem_tiled_copy_V =
      make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  auto tOsVt = smem_thr_copy_V.partition_S(sVt);
  auto tOrVt_view = smem_thr_copy_V.retile_D(tOrVt);

  // copy q
  cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // multiply sm scale
  half2 sm_half2 = {__float2half_rn(sm_scale), __float2half_rn(sm_scale)};
  auto tQsQ_int4 = recast<int4>(tQsQ);
#pragma unroll
  for (int ii = 0; ii < size(tQsQ_int4); ii++) {
    auto tmp = tQsQ_int4(ii);
    auto tmp_half2 = (half2*)&tmp;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
      tmp_half2[jj] = __hmul2_rn(sm_half2, tmp_half2[jj]);
    }
    tQsQ_int4(ii) = tmp;
  }
  // multiply sm scale

  // copy kv
  cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
  cp_async_fence();
  cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
  cp_async_fence();
  // copy kv

  // ((2,2),MMA_M,MMA_K)
  // rAccOut存的是最终想要的 Br x d 的矩阵
  auto rAccOut =
      partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  // rAccScore存的是 中间attention score的 S矩阵
  auto rAccScore = partition_fragment_C(
      tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

  if (thread0()) {
    PRINT("rAccOut", rAccOut);
    PRINT("rAccScore", rAccScore);
  }

  clear(rAccOut);

  // ((2,2),MMA_M,MMA_N) to ((2,MMA_M),(2,MMA_N))
  auto ol = logical_divide(rAccOut.layout(), Shape<Int<2>>{});
  auto rAccOut_new_layout =
      make_layout(make_layout(get<1>(get<0>(ol)), get<1>(ol)),
                  make_layout(get<0>(get<0>(ol)), get<2>(ol)));
  auto rAccOut_new = make_tensor(rAccOut.data(), rAccOut_new_layout);


  if (thread0()) {
    PRINT("ol", ol);
    PRINT("rAccOut_new_layout", rAccOut_new_layout);
    PRINT("rAccOut_new", rAccOut_new);
  }

	auto test_sl = logical_divide(rAccScore.layout(), Shape<Int<2>>{});
	auto test_rAccScore_new_layout =
    make_layout(make_layout(get<1>(get<0>(test_sl)), get<1>(test_sl)),
                make_layout(get<0>(get<0>(test_sl)), get<2>(test_sl)));
	auto test_scores = make_tensor(rAccScore.data(), test_rAccScore_new_layout);
	if (thread0()) {
    	PRINT("test_sl", test_sl);
    	PRINT("test_rAccScore_new_layout",test_rAccScore_new_layout);
    	PRINT("test_scores",test_scores );
	}


  const int n_block_min = 0;
  int n_block_max = cute::ceil_div(k_len, kBlockN);
  #pragma unroll 1
  for (int ii = n_block_min; ii < n_block_max; ii++) {
    clear(rAccScore);
    // wait k
    cp_async_wait<1>();
    __syncthreads();

    // S = Q@K.T
    cute::copy(smem_tiled_copy_Q, tSsQ(_, _, Int<0>{}),
               tSrQ_view(_, _, Int<0>{}));
    cute::copy(smem_tiled_copy_K, tSsK(_, _, Int<0>{}),
               tSrK_view(_, _, Int<0>{}));
#pragma unroll
    for (int si = 0; si < size<2>(tSrQ); si++) {
      if (si < size<2>(tSrQ) - 1) {
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, si + 1),
                   tSrQ_view(_, _, si + 1));
        cute::copy(smem_tiled_copy_K, tSsK(_, _, si + 1),
                   tSrK_view(_, _, si + 1));
      }
      cute::gemm(tiled_mma, tSrQ(_, _, si), tSrK(_, _, si), rAccScore);
    }

    // ((2, 2),(MMA_M, MMA_N)) -> ((2,MMA_M),(2,MMA_N))
    auto sl = logical_divide(rAccScore.layout(), Shape<Int<2>>{});
    auto rAccScore_new_layout =
        make_layout(make_layout(get<1>(get<0>(sl)), get<1>(sl)),
                    make_layout(get<0>(get<0>(sl)), get<2>(sl)));




    // 所有线程的scores加起来应该就是小的S矩阵，size: [Br, Bc]. 注意这里是一个thread block共同持有这个S矩阵
    auto scores = make_tensor(rAccScore.data(), rAccScore_new_layout);

    __syncthreads();
    // advance k
    if (ii != n_block_max - 1) {
      gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}),
                      make_coord(ii + 1, _));
      tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    }
    cp_async_fence();
    // wait v
    cp_async_wait<1>();
    __syncthreads();

    // O = softmax(S)*V
    auto scores_fp16 = make_tensor_like<half_t>(scores);
    auto scores_fp32x2 = recast<float2>(scores);
    auto scores_fp16x2 = recast<half2>(scores_fp16);
#pragma unroll
    for (int si = 0; si < size(scores_fp16x2); si++) {
      scores_fp16x2(si) = __float22half2_rn(scores_fp32x2(si));
    }
    // ((2,MMA_M),(2,MMA_N)) to ((2,2,2),MMA_M,MMA_N/2)
    // ((2,MMA_M),(2,(2,MMA_N/2)))
    auto l = logical_divide(scores.layout(), Shape<X, Shape<X, Int<2>>>{});
    auto scores_new_layout =
        make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)),
                                get<0>(get<1>(get<1>(l)))),
                    get<1>(get<0>(l)), get<1>(get<1>(get<1>(l))));
    auto tOrS = make_tensor(scores_fp16.data(), scores_new_layout);


    cute::copy(smem_tiled_copy_V, tOsVt(_, _, Int<0>{}),
               tOrVt_view(_, _, Int<0>{}));
#pragma unroll
    for (int oi = 0; oi < size<2>(tOrS); oi++) {
      if (oi < size<2>(tOrS) - 1) {
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, oi + 1),
                   tOrVt_view(_, _, oi + 1));
      }
      cute::gemm(tiled_mma, tOrS(_, _, oi), tOrVt(_, _, oi), rAccOut);
    }

    __syncthreads();
    if (ii != n_block_max - 1) {
      gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}),
                      make_coord(ii + 1, _));
      tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
      cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    }
    cp_async_fence();
  }
  // write back
  auto rAccOut_fp16 = make_tensor_like<half_t>(rAccOut);
  auto rAccOut_fp32x2 = recast<float2>(rAccOut);
  auto rAccOut_fp16x2 = recast<half2>(rAccOut_fp16);
#pragma unroll
  for (int si = 0; si < size(rAccOut_fp16x2); si++) {
    rAccOut_fp16x2(si) = __float22half2_rn(rAccOut_fp32x2(si));
  }

  auto sO = make_tensor(sQ.data(), SmemLayoutO{});
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  // ((Atom,AtomNum),MMA_M,MMA_N)
  auto taccOrO = smem_thr_copy_O.retile_S(rAccOut_fp16);
  // ((Atom,AtomNum),PIPE_M,PIPE_N)
  auto taccOsO = smem_thr_copy_O.partition_D(sO);
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  auto gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}),
                       make_coord(m_block, _));
  GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  auto tOsO = gmem_thr_copy_O.partition_S(sO);
  auto tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

  __syncthreads();
  cute::copy(gmem_tiled_copy_O, tOsO, tOgO);
}

namespace config {
using namespace cute;

template <typename T_, int kHeadDim_ = 64, int kBlockM_ = 64, int kBlockN_ = 64>
struct FlashConfig {
  using T = T_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;

  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKGmem =
      kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  using SmemLayoutAtom = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{}, Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                                        Stride<Int<kBlockKSmem>, Int<1>>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

  using SmemLayoutAtomVtransposedNoSwizzle =
      Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>,
             Stride<Int<1>, Int<kBlockKSmem>>>;
  using SmemLayoutAtomVtransposed = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
  using SmemLayoutVtransposed = decltype(tile_to_shape(
      SmemLayoutAtomVtransposed{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(tile_to_shape(SmemLayoutAtomVtransposedNoSwizzle{},
                             Shape<Int<kHeadDim>, Int<kBlockN>>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;
  using SmemLayoutAtomO = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{}, Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                                        Stride<Int<kBlockKSmem>, Int<1>>>{}));
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<DefaultCopy, T>;

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int kMmaEURepeatM = 4;
  static constexpr int kMmaEURepeatN = 1;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  using TiledMMA =
      decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
  static constexpr int kThreadNum = size(TiledMMA{});
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / 8;
  using gmem_copy_atom = Copy_Atom<g2s_copy_traits, cute::half_t>;
  using gmem_thr_layout = Layout<
      Shape<Int<kThreadNum / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, Int<1>>>;
  using gmem_val_layout = Layout<Shape<Int<1>, Int<8>>>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      gmem_copy_atom{}, gmem_thr_layout{}, gmem_val_layout{}));
  using s2g_copy_atom = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using GmemTiledCopyO = decltype(make_tiled_copy(
      s2g_copy_atom{}, gmem_thr_layout{}, gmem_val_layout{}));

  static constexpr int shm_size_q = cute::cosize(SmemLayoutQ{});
  static constexpr int shm_size_kv = cute::cosize(SmemLayoutKV{}) * 2;
  static constexpr int kShmSize = (shm_size_kv + shm_size_q) * sizeof(half);
};

}  // namespace config


torch::Tensor forward_no_softmax(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  int bs = q.size(0);
  int head_num = q.size(1);
  int q_len = q.size(2);
  int head_dim = q.size(3);
  int k_len = k.size(2);

  int head_stride = q.stride(1);

  auto out = torch::empty_like(q);

  float sm_scale = 1.0 / sqrt(head_dim);

  int bx = (q_len + config.kBlockM - 1) / config.kBlockM;
  std::cout<<"q len: "<<q_len<" bx:"<<bx<std::endl;


  // only for head_dim=64
  config::FlashConfig<cute::half_t> config;
  dim3 block = config.kThreadNum;
  dim3 grid((q_len + config.kBlockM - 1) / config.kBlockM, bs * head_num);
  int shm_size = config.kShmSize;
  auto partition_kernel = flash_forward<decltype(config)>;
  cudaFuncSetAttribute(partition_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
//  partition_kernel<<<grid, block, shm_size>>>(
//      (void*)out.data_ptr(), (const void*)q.data_ptr(),
//      (const void*)k.data_ptr(), (const void*)v.data_ptr(), head_stride, q_len,
//      k_len, sm_scale);

  PRINT("grid", grid);
  PRINT("block", block);

  partition_kernel<<<grid, block, shm_size>>>(
      (void*)out.data_ptr(), (const void*)q.data_ptr(),
      (const void*)k.data_ptr(), (const void*)v.data_ptr(), head_stride, q_len,
      k_len, sm_scale);
  return out;
}