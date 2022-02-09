#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor customconv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor threshold,
    int nr_xnor_gates,
    int nr_additional_samples
  );

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor customconv1d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor threshold,
    int nr_xnor_gates,
    int nr_additional_samples
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(output);
  CHECK_INPUT(threshold);
  return customconv1d_cuda(input, weight, output, threshold, nr_xnor_gates, nr_additional_samples);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("customconv1d", &customconv1d, "CUSTOMCONV1D");
}
