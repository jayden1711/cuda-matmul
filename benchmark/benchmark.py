"""
Benchmark: Custom CUDA matmul kernels vs PyTorch / cuDNN baseline

Measures:
  - Wall-clock latency (ms) averaged over many warm runs
  - Achieved TFLOPS
  - Memory bandwidth utilization
  - Numerical correctness (max absolute error vs torch.matmul)

Usage:
  python benchmarks/benchmark.py                   # default sweep
  python benchmarks/benchmark.py --sizes 512 1024  # specific sizes
  python benchmarks/benchmark.py --csv results.csv # export to CSV
"""

import argparse
import csv
import time
from typing import List, Tuple

import torch
import torch.utils.benchmark as bench

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuda_matmul
    CUSTOM_AVAILABLE = True
except ImportError:
    CUSTOM_AVAILABLE = False
    print("[warn] cuda_matmul extension not built. Run: pip install -e .")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_tflops(M: int, N: int, K: int, elapsed_ms: float) -> float:
    flops = 2.0 * M * N * K
    return flops / (elapsed_ms * 1e-3) / 1e12


def max_abs_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (out - ref).abs().max().item()


def warmup_and_time(fn, warmup: int = 10, repeats: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeats


# ---------------------------------------------------------------------------
# Benchmark one (M, N, K) configuration
# ---------------------------------------------------------------------------

def benchmark_size(
    M: int, N: int, K: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> dict:
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    ref = torch.matmul(A, B)

    results = {"M": M, "N": N, "K": K}

    # ----- PyTorch / cuBLAS baseline -----
    t_torch = warmup_and_time(lambda: torch.matmul(A, B))
    results["torch_ms"]    = round(t_torch, 4)
    results["torch_tflops"] = round(compute_tflops(M, N, K, t_torch), 3)

    if not CUSTOM_AVAILABLE:
        return results

    # ----- Naive kernel -----
    out_naive = cuda_matmul.naive(A, B)
    t_naive = warmup_and_time(lambda: cuda_matmul.naive(A, B))
    results["naive_ms"]       = round(t_naive, 4)
    results["naive_tflops"]   = round(compute_tflops(M, N, K, t_naive), 3)
    results["naive_err"]      = round(max_abs_error(out_naive, ref), 6)
    results["naive_speedup"]  = round(t_torch / t_naive, 3)

    # ----- Tiled kernel -----
    out_tiled = cuda_matmul.tiled(A, B)
    t_tiled = warmup_and_time(lambda: cuda_matmul.tiled(A, B))
    results["tiled_ms"]       = round(t_tiled, 4)
    results["tiled_tflops"]   = round(compute_tflops(M, N, K, t_tiled), 3)
    results["tiled_err"]      = round(max_abs_error(out_tiled, ref), 6)
    results["tiled_speedup"]  = round(t_torch / t_tiled, 3)

    # ----- Vectorized kernel -----
    out_vec = cuda_matmul.vectorized(A, B)
    t_vec = warmup_and_time(lambda: cuda_matmul.vectorized(A, B))
    results["vec_ms"]         = round(t_vec, 4)
    results["vec_tflops"]     = round(compute_tflops(M, N, K, t_vec), 3)
    results["vec_err"]        = round(max_abs_error(out_vec, ref), 6)
    results["vec_speedup"]    = round(t_torch / t_vec, 3)

    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(rows: List[dict]) -> None:
    header = (
        f"{'Size':>20} | {'torch':>8} {'TFLOPS':>7} | "
        f"{'naive':>8} {'TFLOPS':>7} {'err':>9} {'x':>5} | "
        f"{'tiled':>8} {'TFLOPS':>7} {'err':>9} {'x':>5} | "
        f"{'vec':>8} {'TFLOPS':>7} {'err':>9} {'x':>5}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in rows:
        size_str = f"{r['M']}x{r['N']}x{r['K']}"
        torch_str = f"{r['torch_ms']:>8.3f} {r['torch_tflops']:>7.2f}"

        if CUSTOM_AVAILABLE:
            naive_str = (f"{r['naive_ms']:>8.3f} {r['naive_tflops']:>7.2f} "
                         f"{r['naive_err']:>9.2e} {r['naive_speedup']:>5.2f}x")
            tiled_str = (f"{r['tiled_ms']:>8.3f} {r['tiled_tflops']:>7.2f} "
                         f"{r['tiled_err']:>9.2e} {r['tiled_speedup']:>5.2f}x")
            vec_str   = (f"{r['vec_ms']:>8.3f} {r['vec_tflops']:>7.2f} "
                         f"{r['vec_err']:>9.2e} {r['vec_speedup']:>5.2f}x")
        else:
            naive_str = tiled_str = vec_str = " (extension not built)"

        print(f"{size_str:>20} | {torch_str} | {naive_str} | {tiled_str} | {vec_str}")

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Square matrix sizes to benchmark",
    )
    parser.add_argument(
        "--nonsquare", action="store_true",
        help="Also benchmark non-square shapes (e.g. LLM projection layers)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to write CSV results",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
    )
    parser.add_argument(
        "--repeats", type=int, default=100,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device found.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"\nDevice: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")

    shapes: List[Tuple[int, int, int]] = [(s, s, s) for s in args.sizes]

    if args.nonsquare:
        shapes += [
            (4096, 1,    4096),    # batch=1 token, large projection
            (4096, 4096, 1024),    # typical FFN
            (2048, 512,  8192),    # down-projection
            (1,    4096, 4096),    # single-token decode
        ]

    rows = []
    for M, N, K in shapes:
        print(f"  Benchmarking {M}x{N}x{K}...", end="", flush=True)
        r = benchmark_size(M, N, K)
        rows.append(r)
        print(" done")

    print_results(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results written to {args.csv}")


if __name__ == "__main__":
    main()
