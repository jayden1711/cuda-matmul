#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../include/matmul.h"

// Validates inputs are 2D float32 CUDA tensors and dimensions are compatible
static void check_inputs(
    const torch::Tensor& A,
    const torch::Tensor& B)
{
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2, "A must be 2-dimensional");
    TORCH_CHECK(B.dim() == 2, "B must be 2-dimensional");
    TORCH_CHECK(A.size(1) == B.size(0),
        "Inner dimensions must match: A is (", A.size(0), "x", A.size(1),
        "), B is (", B.size(0), "x", B.size(1), ")");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
}

torch::Tensor matmul_naive_forward(
    const torch::Tensor& A,
    const torch::Tensor& B)
{
    check_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    launch_matmul_naive(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        at::cuda::getCurrentCUDAStream());
    return C;
}

torch::Tensor matmul_tiled_forward(
    const torch::Tensor& A,
    const torch::Tensor& B)
{
    check_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    launch_matmul_tiled(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        at::cuda::getCurrentCUDAStream());
    return C;
}

torch::Tensor matmul_vectorized_forward(
    const torch::Tensor& A,
    const torch::Tensor& B)
{
    check_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    launch_matmul_vectorized(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        at::cuda::getCurrentCUDAStream());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom CUDA matrix multiplication kernels";
    m.def("naive",      &matmul_naive_forward,
          "Naive CUDA matmul (global memory only)",
          py::arg("A"), py::arg("B"));
    m.def("tiled",      &matmul_tiled_forward,
          "Shared memory tiled CUDA matmul",
          py::arg("A"), py::arg("B"));
    m.def("vectorized", &matmul_vectorized_forward,
          "Double-buffered vectorized CUDA matmul",
          py::arg("A"), py::arg("B"));
}
