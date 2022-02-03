#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor customconv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output
  );

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor customconv1d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output
  ) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(output);
  return customconv1d_cuda(input, weight, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("customconv1d", &customconv1d, "CUSTOMCONV1D");
}
