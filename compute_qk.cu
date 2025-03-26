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


    Tensor Q = make_tensor(make_gmem_ptr(q), make_shape());


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


torch::Tensor compute_qk_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  int bs = q.size(0);
  int head_num = q.size(1);
  int q_len = q.size(2);
  int head_dim = q.size(3);
  int k_len = k.size(2);

  int head_stride = q.stride(1);

  auto out = torch::empty_like(q);

  float sm_scale = 1.0 / sqrt(head_dim) * M_LOG2E;

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