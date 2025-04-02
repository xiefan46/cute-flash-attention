#include <torch/extension.h>
#include <tuple>

torch::Tensor forward_without_decay(torch::Tensor q, torch::Tensor k, torch::Tensor v);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_without_decay_precision(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor cute_compute_kv(torch::Tensor k, torch::Tensor v);
torch::Tensor cute_compute_kv_all_f16(torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_without_decay", torch::wrap_pybind_function(forward_without_decay), "forward_without_decay");
  m.def("forward_without_decay_precision", torch::wrap_pybind_function(forward_without_decay_precision), "forward_without_decay_precision");
  m.def("cute_compute_kv", torch::wrap_pybind_function(cute_compute_kv), "cute_compute_kv");
  m.def("cute_compute_kv_all_f16", torch::wrap_pybind_function(cute_compute_kv_all_f16), "cute_compute_kv_all_f16");
}