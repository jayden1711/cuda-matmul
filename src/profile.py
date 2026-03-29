#!/usr/bin/env python3
"""
Profiling helper — wraps ncu (Nsight Compute CLI) to profile each kernel
and extract key metrics: compute throughput, memory throughput, occupancy,
achieved TFLOPS, L1/L2 hit rates.

Prerequisites:
  - NVIDIA Nsight Compute CLI (ncu) installed and on PATH
  - Elevated permissions may be required: sudo or set perf_event_paranoid

Usage:
  python profiling/profile.py --kernel tiled --size 1024
  python profiling/profile.py --all --size 2048 --output profiling/results/
"""

import argparse
import os
import subprocess
import sys


NCU_METRICS = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",     # occupancy
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "gpu__time_duration.sum",
])

KERNEL_NAMES = {
    "naive":      "matmul_naive",
    "tiled":      "matmul_tiled",
    "vectorized": "matmul_vectorized",
}


def build_ncu_command(
    kernel: str,
    size: int,
    output_dir: str,
) -> list:
    kernel_regex = KERNEL_NAMES.get(kernel, kernel)
    report_path  = os.path.join(output_dir, f"{kernel}_{size}.ncu-rep")

    cmd = [
        "ncu",
        "--kernel-name", kernel_regex,
        "--launch-skip", "0",
        "--launch-count", "1",
        "--metrics", NCU_METRICS,
        "--export", report_path,
        "--force-overwrite",
        sys.executable,
        "-c",
        f"""
import sys, os
sys.path.insert(0, '.')
import torch, cuda_matmul
A = torch.randn({size}, {size}, device='cuda')
B = torch.randn({size}, {size}, device='cuda')
_ = cuda_matmul.{kernel}(A, B)
torch.cuda.synchronize()
""",
    ]
    return cmd


def run_profile(kernel: str, size: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd = build_ncu_command(kernel, size, output_dir)
    print(f"\n  Profiling: {kernel} @ {size}x{size}")
    print(f"  Command:   {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [warn] ncu exited with code {result.returncode}")
    else:
        print(f"  Report saved → {output_dir}/{kernel}_{size}.ncu-rep")
        print(f"  Open with:  ncu-ui {output_dir}/{kernel}_{size}.ncu-rep")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["naive", "tiled", "vectorized"],
                        default=None)
    parser.add_argument("--all", dest="all_kernels", action="store_true",
                        help="Profile all three kernels")
    parser.add_argument("--size", type=int, default=1024,
                        help="Square matrix dimension")
    parser.add_argument("--output", type=str, default="profiling/results",
                        help="Directory for .ncu-rep report files")
    args = parser.parse_args()

    if not args.kernel and not args.all_kernels:
        parser.error("Specify --kernel <name> or --all")

    kernels = list(KERNEL_NAMES.keys()) if args.all_kernels else [args.kernel]

    for k in kernels:
        run_profile(k, args.size, args.output)

    print("\nDone. Open reports in Nsight Compute UI:")
    print(f"  ncu-ui {args.output}/<report>.ncu-rep\n")


if __name__ == "__main__":
    main()
