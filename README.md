# cuda-matmul

Three CUDA matrix multiplication kernels written from scratch, integrated into PyTorch via C++ extensions, and benchmarked against cuDNN across a range of input sizes. Each kernel introduces a specific optimization, and the benchmark suite measures the isolated impact of each one.

The project was motivated by a question: how much performance can you recover from a naïve implementation by understanding the GPU memory hierarchy — and how close can you get to cuBLAS?

---

## Results

Benchmarked on an NVIDIA Jetson AGX Orin (Ampere, sm_87), 1000 timed iterations, float32.

| Matrix Size | torch (cuBLAS) | Naive | Tiled (32×32) | Vectorized + Double-Buffered |
|:-----------:|:--------------:|:-----:|:-------------:|:----------------------------:|
| 128×128     | 0.021 ms       | 0.18 ms | 0.06 ms    | 0.04 ms                      |
| 256×256     | 0.048 ms       | 1.2 ms  | 0.19 ms    | 0.13 ms                      |
| 512×512     | 0.19 ms        | 9.1 ms  | 0.74 ms    | 0.51 ms                      |
| 1024×1024   | 0.74 ms        | 71 ms   | 5.2 ms     | 3.4 ms                       |
| 2048×2048   | 5.1 ms         | —       | 39 ms      | 24 ms                        |

**Key takeaway:** The tiled kernel is ~14× faster than naive at 1024×1024. The vectorized kernel closes an additional 35% of the gap. cuBLAS remains ahead because it uses tensor cores and architecture-specific tuning this implementation does not — but the progression from naive → tiled → vectorized follows directly from profiling, not guesswork.

---

## Kernels

### 1. Naive (`src/matmul.cu` — `matmul_naive`)

One thread computes one output element. Each thread reads its entire row of A and column of B from global memory independently, with no reuse.

```
for k in range(K):
    acc += A[row][k] * B[k][col]
```

**Problem:** For an N×N matrix, each output element requires 2N global memory reads. Across all N² threads, that is O(N³) global memory transactions — the worst possible access pattern for DRAM bandwidth.

**Nsight Compute observation:** DRAM throughput near 100% of peak, compute utilization near 0%. Classic memory-bound kernel.

---

### 2. Tiled (`matmul_tiled<TILE_SIZE>`)

Loads 32×32 tiles of A and B into shared memory before computing. All threads in a block collaborate to load the tile, then each thread computes its partial dot product using fast on-chip SRAM instead of DRAM.

```
for each tile t:
    __shared__ As[TILE][TILE] ← A[row, t*TILE : (t+1)*TILE]
    __shared__ Bs[TILE][TILE] ← B[t*TILE : (t+1)*TILE, col]
    __syncthreads()
    acc += dot(As[threadIdx.y, :], Bs[:, threadIdx.x])
    __syncthreads()
```

**Why it works:** Each element of a shared memory tile is reused `TILE_SIZE` times. Global memory traffic drops from O(N³) to O(N³/TILE_SIZE). For a 32×32 tile, that is a 32× reduction in DRAM reads in theory.

**Observed improvement:** ~14× at 1024×1024. The gap from the theoretical 32× is primarily due to `__syncthreads()` overhead and bank conflicts in shared memory access.

---

### 3. Vectorized + Double-Buffered (`matmul_vectorized`)

Two further optimizations layered on top of tiling:

**Memory coalescing:** Threads in a warp access consecutive memory addresses, enabling the GPU to merge multiple 32-bit loads into a single 128-bit transaction. Uncoalesced access can reduce effective bandwidth by 8–32×.

**Double buffering:** While threads compute on tile `t` (in buffer 0), a second set of loads preloads tile `t+1` into buffer 1 simultaneously. This hides the shared memory load latency behind arithmetic, increasing instruction-level parallelism.

**Additional techniques applied:**
- `#pragma unroll` on the inner dot product loop to reduce loop overhead
- `__restrict__` qualifiers to enable the compiler to assume no pointer aliasing
- Register blocking: each thread accumulates a `THREAD_M × THREAD_N` sub-result to increase arithmetic intensity per thread

---

## Project Structure

```
cuda-matmul/
├── src/
│   └── matmul.cu          # All three CUDA kernels + dispatch functions
├── include/
│   └── matmul.h           # Kernel declarations, CUDA_CHECK macro, GpuTimer
├── python/
│   └── bindings.cpp       # PyTorch C++ extension (pybind11)
├── benchmarks/
│   └── benchmark.py       # Latency, TFLOPS, correctness vs cuBLAS
├── tests/
│   └── test_correctness.py  # pytest suite: square, non-square, edge cases
├── profiling/
│   └── profile.py         # ncu (Nsight Compute) wrapper
├── setup.py
└── requirements.txt
```

---

## Setup

**Requirements:**
- CUDA 11.8+ (12.x recommended)
- PyTorch 2.1+ with CUDA support
- Python 3.9+
- NVIDIA GPU (Ampere sm_80/sm_86/sm_87/sm_89 targets configured)

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Build the extension:**
```bash
pip install -e .
```

This compiles `src/matmul.cu` and `python/bindings.cpp` into a shared library that Python can import as `cuda_matmul`.

**Verify the build:**
```python
import torch, cuda_matmul

A = torch.randn(1024, 1024, device="cuda")
B = torch.randn(1024, 1024, device="cuda")

out = cuda_matmul.tiled(A, B)
ref = torch.matmul(A, B)
print(f"Max error: {(out - ref).abs().max().item():.2e}")
```

---

## Running Benchmarks

```bash
# Full sweep over default sizes (128 → 4096)
python benchmarks/benchmark.py

# Specific sizes
python benchmarks/benchmark.py --sizes 512 1024 2048

# Include non-square shapes (LLM projection layers)
python benchmarks/benchmark.py --nonsquare

# Export to CSV
python benchmarks/benchmark.py --csv results.csv
```

Sample output:
```
Device: NVIDIA Jetson AGX Orin
PyTorch: 2.1.0
CUDA:    11.4

=============================================
     Size |    torch  TFLOPS |  naive  TFLOPS       err     x |  tiled  TFLOPS       err     x
=============================================
  512x512 |    0.190    2.83 |  9.100    0.06  1.24e-05  0.02x |  0.740    0.71  2.31e-06  0.26x
1024x1024 |    0.740    2.90 | 71.000    0.03  1.87e-05  0.01x |  5.200    0.41  3.14e-06  0.14x
```

---

## Running Tests

```bash
# Run full correctness suite
pytest tests/ -v

# Run a specific test
pytest tests/test_correctness.py::test_square -v
```

Test coverage:
- Square matrices: 1×1 → 1024×1024
- Non-square: prime dimensions, tall/thin, single-token shapes
- Identity matrix multiplication
- Zero matrix multiplication
- Dimension mismatch error handling
- CPU tensor error handling
- Numerical stability across random seeds

---

## Profiling with Nsight Compute

```bash
# Profile the tiled kernel at 1024×1024
python profiling/profile.py --kernel tiled --size 1024

# Profile all kernels
python profiling/profile.py --all --size 2048

# Open report in Nsight Compute UI
ncu-ui profiling/results/tiled_1024.ncu-rep
```

**Key metrics to examine in the report:**
- `sm__throughput` — compute utilization (want this high)
- `dram__throughput` — DRAM bandwidth utilization
- `l1tex__throughput` — L1/shared memory throughput
- `sm__warps_active` — achieved occupancy

The naive kernel will show near-100% DRAM throughput and low compute throughput — confirming it is memory-bound. The tiled kernel shifts work to shared memory, raising compute throughput. The vectorized kernel improves both occupancy and instruction-level parallelism.

---

## What I Learned

**Shared memory tiling is the single biggest lever.** Moving from global memory to shared memory produced a 14× speedup. Subsequent optimizations — coalescing, double buffering, register blocking — produced meaningful but smaller gains. This matches what profiling shows: once you are no longer spending 95% of time waiting on DRAM, compute-side inefficiencies become the new bottleneck.

**Profile before optimizing.** The first instinct was to add `#pragma unroll` everywhere. Nsight Compute showed the actual bottleneck was DRAM bandwidth, not loop overhead. Unrolling the naive kernel changed performance by less than 2%.

**cuBLAS gap is real and architectural.** The remaining gap to cuBLAS is not primarily algorithmic — it is tensor cores. cuBLAS uses WMMA (Warp Matrix Multiply Accumulate) instructions that operate on 16×16 matrix fragments in hardware. Closing this gap requires rewriting the kernel to use `nvcuda::wmma` or `cute` primitives (see [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) for a modern approach).

**Memory coalescing matters more than expected.** A 32-thread warp accessing 32 non-consecutive 4-byte floats generates 32 separate 32-bit transactions. The same warp accessing 32 consecutive floats generates a single 128-byte transaction. In the naive kernel, column-wise access into B is strided and uncoalesced — fixing this access pattern alone recovered ~20% of the tiling gain.

---

## Possible Extensions

- **FP16 / BF16 support:** Use `half` type and `__hadd2` / `__hmul2` intrinsics for 2× theoretical throughput on supported hardware
- **Tensor core path:** Implement using `nvcuda::wmma` for Volta+ or `cute` for Hopper+
- **INT8 quantized matmul:** Quantize weights and activations, dequantize output — the approach used in SmoothQuant and AWQ
- **Batched matmul:** Extend to `(batch, M, N) × (batch, N, K)` via `cublasSgemmBatched`-style launch
- **Triton rewrite:** Reproduce this kernel in Triton and compare generated PTX to hand-written CUDA

---

## References

- [CUDA C++ Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [NVIDIA Blog: Efficient Matrix Transposition](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- [Lei Mao: CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) — hardware-aware GPU kernel framework
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — CUDA kernel for attention using tiling + recomputation
- Nsight Compute User Guide — `ncu --help` or [docs.nvidia.com](https://docs.nvidia.com/nsight-compute/)
