#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cute/tensor.hpp>
#include <torch/torch.h>
#include <iostream>

using namespace cute;

#define PRINT(name, content) \
print(name);             \
print(" : ");            \
print(content);          \
print("\n");


// compute S = Qk_T
// Q [B, H, N, d] K [B, H, N, d] S = [B, H, N, N]
torch::Tensor gemm_simple(torch::Tensor Q, torch::Tensor K) {
  int B = q.size(0);
  int H = q.size(1);
  int N = q.size(2);
  int d = q.dize(3);

  torch::Tensor S = torch::zeros({B, H, N, N}, torch::device(torch::kCUDA));
}