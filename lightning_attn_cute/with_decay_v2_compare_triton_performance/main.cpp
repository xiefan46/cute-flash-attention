#include <torch/extension.h>
#include <tuple>

//torch::Tensor forward_with_decay(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                                torch::Tensor q_decay, torch::Tensor k_decay, torch::Tensor diag_decay, torch::Tensor block_decay);

torch::Tensor forward_with_decay(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                                       torch::Tensor q_decay, torch::Tensor k_decay, torch::Tensor diag_decay, torch::Tensor block_decay);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_with_decay", torch::wrap_pybind_function(forward_with_decay), "forward_with_decay");
}