#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Kernel launchers
void launch_matmul_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream = nullptr);

void launch_matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream = nullptr);

void launch_matmul_vectorized(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream = nullptr);

#ifdef __cplusplus
}
#endif

// Error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Timing utility
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start(cudaStream_t s = nullptr) { cudaEventRecord(start, s); }
    void Stop(cudaStream_t s = nullptr)  { cudaEventRecord(stop,  s); }

    float ElapsedMs() {
        float ms;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};
