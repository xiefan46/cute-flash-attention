#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


template <const int kElementPerThread=8>
__global__ void vector_add(float* A, float* B, float* C, const int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= N / kElementPerThread) {
    return;
  }
  cute::Tensor ta = cute::make_tensor(cute::make_gmem_ptr(A), cute::make_shape(N));
  cute::Tensor tb = cute::make_tensor(cute::make_gmem_ptr(B), cute::make_shape(N));
  cute::Tensor tc = cute::make_tensor(cute::make_gmem_ptr(C), cute::make_shape(N));

  cute::Tensor ta_tile = cute::local_tile(ta, cute::make_shape(cute::Int<kElementPerThread>{}), cute::make_coord(idx));
  cute::Tensor tb_tile = cute::local_tile(tb, cute::make_shape(cute::Int<kElementPerThread>{}), cute::make_coord(idx));
  cute::Tensor tc_tile = cute::local_tile(tc, cute::make_shape(cute::Int<kElementPerThread>{}), cute::make_coord(idx));

  cute::Tensor ta_reg = cute::make_tensor_like(ta_tile);
  cute::Tensor tb_reg = cute::make_tensor_like(tb_tile);
  cute::Tensor tc_reg = cute::make_tensor_like(tc_tile);

  cute::copy(ta_tile, ta_reg);
  cute::copy(tb_tile, tb_reg);

  #pragma unroll
  for (int i = 0; i < kElementPerThread; i++) {
    tc_reg(i) = ta_reg(i) + tb_reg(i);
  }

  cute::copy(tc_reg, tc_tile);
}


int main(int argc, char** argv)
{
  const int N = 2048;
  const int BLOCK_SIZE = 256;
  const int NUM_ELEMENT_PER_THREAD = 8;

  std::cout << "N = " << N << std::endl;

  cute::device_init(0);

  thrust::host_vector<float> h_A(N);
  thrust::host_vector<float> h_B(N);
  thrust::host_vector<float> h_C(N);
  thrust::host_vector<float> h_correct_C(N);

  for (int j = 0; j < N; ++j) {
    h_A[j] = static_cast<float>(2*(rand() / float(RAND_MAX)) - 1);
  }
  for (int j = 0; j < N; ++j) {
    h_B[j] = static_cast<float>(2*(rand() / float(RAND_MAX)) - 1);
  }
  for (int j = 0; j < N; j++) {
    h_C[j] = static_cast<float>(0);
    h_correct_C[j] = h_A[j] + h_B[j];
  }


  thrust::device_vector<float> d_A = h_A;
  thrust::device_vector<float> d_B = h_B;
  thrust::device_vector<float> d_C;

  d_C = h_C;

  // kernel launch
  dim3 Grid(N / BLOCK_SIZE);
  dim3 Block(BLOCK_SIZE / NUM_ELEMENT_PER_THREAD);
  vector_add<NUM_ELEMENT_PER_THREAD><<<Grid, Block>>>(d_A.data().get(), d_B.data().get(), d_C.data().get(), N);
  CUTE_CHECK_LAST();
  thrust::host_vector<float> cute_result = d_C;

  std::cout<<"Start to check if result equals"<<std::endl;
  for (int i = 0; i < N; i++) {
    if (std::abs(h_correct_C[i] - cute_result[i]) > 0.001f) {
      printf("Result not correct! position %d, correct_res: %f, cute result: %f", i, h_correct_C[i], cute_result[i]);
      exit(1);
    }
  }

  for (int i = 0; i < 10; i++) {
    std::cout<<"value: "<<cute_result[i]<<std::endl;
  }
  std::cout<<"Vector add finished"<<std::endl;
  return 0;
}
