#include <torch/extension.h>

#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

  return at::cudnn_convolution(
      input,
      weight,
      bias,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
}

at::Tensor convolution_backward_weight(
    const at::Tensor& input,
    c10::ArrayRef<int64_t> weight_size,
    const at::Tensor& grad_output,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

  return at::cudnn_convolution_backward_weight(
      weight_size,
      grad_output,
      input,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
}

at::Tensor convolution_backward_input(
    c10::ArrayRef<int64_t> input_size,
    const at::Tensor& weight,
    const at::Tensor& grad_output,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

  return at::cudnn_convolution_backward_input(
      input_size,
      grad_output,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convolution", &convolution, "convolution");
  m.def("convolution_backward_weight", &convolution_backward_weight, "convolution backward weight");
  m.def("convolution_backward_input", &convolution_backward_input, "convolution backward input");
}