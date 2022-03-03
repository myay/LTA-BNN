#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor customconv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor threshold,
    int nr_xnor_gates,
    int nr_additional_samples,
    int majv_shift,
    int threshold_scaling
  );

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor customconv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor threshold,
    int nr_xnor_gates,
    int nr_additional_samples,
    int majv_shift,
    int threshold_scaling
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(output);
  CHECK_INPUT(threshold);
  return customconv2d_cuda(input, weight, output, threshold, nr_xnor_gates, nr_additional_samples, majv_shift, threshold_scaling);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("customconv2d", &customconv2d, "CUSTOMCONV2D");
}
