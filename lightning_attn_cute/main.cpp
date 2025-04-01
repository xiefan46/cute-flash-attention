#include <torch/extension.h>
#include <tuple>

torch::Tensor forward_without_decay(torch::Tensor q, torch::Tensor k, torch::Tensor v);
std::tuple<torch::Tensor, torch::Tensor> forward_without_decay_precision(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_without_decay", torch::wrap_pybind_function(forward_without_decay), "forward_without_decay");
  m.def("forward_without_decay_precision", torch::wrap_pybind_function(forward_without_decay_precision), "forward_without_decay_precision");
}