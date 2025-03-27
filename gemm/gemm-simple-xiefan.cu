#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T* data, int n);


template <typename T, int bM, int bN, int bK, class TiledMma>
__global__ static void gemm_simple(T* Aptr, T* Bptr, T* Cptr, int m, int n, int k, TiledMma mma) {
  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, 1));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, 1));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, 1));

  const int bx = blockIdx.x, by = blockIdx.y;

  Tensor gA = local_tile(make_gmem_ptr(A), make_tile(Int<bM>{}, Int<bK>{}), make_coord(by, _));
  Tensor gB = local_tile(make_gmem_ptr(B), make_tile(Int<bN>{}, Int<bK>{}), make_coord(bx, _));
  Tensor gC = local_tile(make_gmem_ptr(C), make_tile(Int<bM>{}, Int<bN>{}), make_coord(by, bx));


  auto thr_mma = mma.get_slice(threadIdx.x);
  Tensor tAgA = thr_mma.partition_A(gA);
  Tensor tBgB = thr_mma.partition_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
  Tensor tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  Tensor tCrC = thr_mma.partition_fragment_C(gC(_, _));

  clear(tCrC);

  const int num_tiled_k = size<2>(gA);
  for (int i = 0; i < num_tiled_k; i++) {
    copy(tAgA(_, _, _, i), tArA);
    copy(tBgB(_, _, _, i), tBrB);
    gemm(mma, tCrC, tArA, tBrB, tCrC);
  }

  copy(tCrC, tCgC);

}

int main() {
  using T = cute::half_t;
  using namespace cute;

  srand(10086);

  const int m = 81920;
  const int n = 256;
  const int k = 256;

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<32>{};



  T* Aptr_d;
  T* Bptr_d;
  T* Cptr_d;

  cudaMalloc(&Aptr_d, sizeof(T) * m * k);
  cudaMalloc(&Bptr_d, sizeof(T) * n * k);
  cudaMalloc(&Cptr_d, sizeof(T) * m * n);

  T* Aptr_h = (T*)malloc(sizeof(T) * m * k);
  T* Bptr_h = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_h, m * k);
  gen_rand_data(Bptr_h, n * k);

  cudaMemcpy(Aptr_d, Aptr_h, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr_d, Bptr_h, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;


  TiledMMA mma = make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{}), make_layout(Shape<_1, _2, _1>{}));

  std::cout<<" size mma: "<<size(mma)<<std::endl;
  print(mma);

  // print_latex(mma);

  dim3 block(size(mma));
  dim3 grid(n / bN, m / bM);

  for (int i = 0; i < 1; i++) {
    gemm_simple<T, bM, bN, bK><<<grid, block>>>(Aptr_d, Bptr_d, Cptr_d, m, n, k, mma);
  }

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 1; ++i) {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
              n, m, k,
              &alpha,
              (half *)Bptr_d, k,
              (half *)Aptr_d, k,
              &beta,
              (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  T *Cptr_host;
  T *Cptr_cublas_host;

  Cptr_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // compare
  cudaMemcpy(Cptr_host, Cptr_d, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.1;
  for (int i = 0; i < m * n; ++i) {
    float v1 = Cptr_host[i];
    float v2 = Cptr_cublas_host[i];
    if (fabs(v2 - v1) > threshold) {
      printf("error! v1 = %f, v2 = %f\n", v1, v2);
    }
  }

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_cublas_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(2, 2);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc2 = local_tile(tensor_C_cublas, tile, coor);

  print_tensor(tc1);
  print_tensor(tc2);


}

template <typename T>
void gen_rand_data(T* data, int n) {
  for (int i = 0; i < n; i++) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = static_cast<T>(v);
  }
}