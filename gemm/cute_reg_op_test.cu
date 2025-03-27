#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>


#define PRINT(name, content) \
print(name);             \
print(" : ");            \
print(content);          \
print("\n");

template <typename T>
void gen_rand_data(T *data, int n);

namespace config {
template <typename T_, int kHeadDim_ = 64, int kBLOCK_ = 64>
struct FlashConfig {
  using T = T_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBLOCK = kBLOCK_;
}

} //end of namespace config

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  constexpr int

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));
  constexpr int kTileM = 128; 
  constexpr int kTileN = 128; 
  constexpr int kTileK = 32; 

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  for (int i = 0; i < 100; ++i) {
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));


}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}
