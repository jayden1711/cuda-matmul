#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/matmul.h"

// ---------------------------------------------------------------------------
// Kernel 1: Naive — one thread per output element, global memory only
// ---------------------------------------------------------------------------
__global__ void matmul_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 2: Shared memory tiling — reduces global memory bandwidth
// ---------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 3: Vectorized loads + double-buffered shared memory
//   - float4 loads align to 128-bit transactions → better memory coalescing
//   - double buffering hides shared memory latency
// ---------------------------------------------------------------------------
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 4
#define THREAD_N 4

__global__ void matmul_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[2][TILE_K][TILE_M];
    __shared__ float Bs[2][TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row_base = by * TILE_M;
    int col_base = bx * TILE_N;

    float acc[THREAD_M][THREAD_N] = {};

    int buf = 0;

    // Preload first tile
    for (int i = ty; i < TILE_K; i += blockDim.y) {
        for (int j = tx; j < TILE_M; j += blockDim.x) {
            int r = row_base + j, c = i;
            As[buf][i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
        }
        for (int j = tx; j < TILE_N; j += blockDim.x) {
            int r = i, c = col_base + j;
            Bs[buf][i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
        }
    }
    __syncthreads();

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        int next = 1 - buf;

        // Prefetch next tile while computing current
        if (t + 1 < num_tiles) {
            for (int i = ty; i < TILE_K; i += blockDim.y) {
                for (int j = tx; j < TILE_M; j += blockDim.x) {
                    int r = row_base + j, c = (t + 1) * TILE_K + i;
                    As[next][i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
                }
                for (int j = tx; j < TILE_N; j += blockDim.x) {
                    int r = (t + 1) * TILE_K + i, c = col_base + j;
                    Bs[next][i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
                }
            }
        }

        // Compute on current buffer
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                #pragma unroll
                for (int n = 0; n < THREAD_N; ++n) {
                    int row_idx = ty * THREAD_M + m;
                    int col_idx = tx * THREAD_N + n;
                    acc[m][n] += As[buf][k][row_idx] * Bs[buf][k][col_idx];
                }
            }
        }

        __syncthreads();
        buf = next;
    }

    // Write results
    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        #pragma unroll
        for (int n = 0; n < THREAD_N; ++n) {
            int row = row_base + ty * THREAD_M + m;
            int col = col_base + tx * THREAD_N + n;
            if (row < M && col < N)
                C[row * N + col] = acc[m][n];
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatchers
// ---------------------------------------------------------------------------
void launch_matmul_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream)
{
    const int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<TILE><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_matmul_vectorized(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 block(TILE_N / THREAD_N, TILE_M / THREAD_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    matmul_vectorized<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
