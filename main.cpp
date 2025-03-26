#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor forward_no_softmax(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor forward_no_softmax_no_normal(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor compute_qk_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("forward", torch::wrap_pybind_function(forward), "forward");
  m.def("forward_no_softmax", torch::wrap_pybind_function(forward_no_softmax), "forward_no_softmax");
  // m.def("forward_no_softmax_no_normal", torch::wrap_pybind_function(forward_no_softmax_no_normal), "forward_no_softmax_no_normal");
  // m.def("compute_qk_forward", torch::wrap_pybind_function(compute_qk_forward), "compute_qk_forward");
}