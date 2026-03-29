"""
Correctness tests for custom CUDA matmul kernels.

Run with: python tests/test_correctness.py
Or via pytest: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

try:
    import cuda_matmul
    SKIP = False
except ImportError:
    SKIP = True

SKIP_MSG = "cuda_matmul extension not built — run: pip install -e ."
ATOL = 1e-4
RTOL = 1e-4

KERNELS = {
    "naive":      lambda a, b: cuda_matmul.naive(a, b),
    "tiled":      lambda a, b: cuda_matmul.tiled(a, b),
    "vectorized": lambda a, b: cuda_matmul.vectorized(a, b),
}


def ref(A, B):
    return torch.matmul(A.float(), B.float())


# ---------------------------------------------------------------------------
# Parametrized shape tests
# ---------------------------------------------------------------------------

SQUARE_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024]

NONSQUARE_SHAPES = [
    (1, 1, 1),
    (3, 5, 7),
    (17, 31, 13),       # prime dims — exercises boundary handling
    (128, 64, 256),
    (512, 1024, 256),
    (1, 4096, 4096),    # single-token decode
    (4096, 1, 4096),
]


@pytest.mark.skipif(SKIP, reason=SKIP_MSG)
@pytest.mark.parametrize("kernel_name,kernel_fn", KERNELS.items())
@pytest.mark.parametrize("size", SQUARE_SIZES)
def test_square(kernel_name, kernel_fn, size):
    A = torch.randn(size, size, device="cuda")
    B = torch.randn(size, size, device="cuda")
    expected = ref(A, B)
    out = kernel_fn(A, B)
    assert torch.allclose(out, expected, atol=ATOL, rtol=RTOL), (
        f"{kernel_name} failed on {size}x{size}: "
        f"max_err={( out - expected).abs().max().item():.2e}"
    )


@pytest.mark.skipif(SKIP, reason=SKIP_MSG)
@pytest.mark.parametrize("kernel_name,kernel_fn", KERNELS.items())
@pytest.mark.parametrize("M,N,K", NONSQUARE_SHAPES)
def test_nonsquare(kernel_name, kernel_fn, M, N, K):
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    expected = ref(A, B)
    out = kernel_fn(A, B)
    assert torch.allclose(out, expected, atol=ATOL, rtol=RTOL), (
        f"{kernel_name} failed on ({M},{N},{K}): "
        f"max_err={(out - expected).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Edge case: identity matrix — output should equal the other operand
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP, reason=SKIP_MSG)
@pytest.mark.parametrize("kernel_name,kernel_fn", KERNELS.items())
@pytest.mark.parametrize("size", [64, 128, 256])
def test_identity(kernel_name, kernel_fn, size):
    I = torch.eye(size, device="cuda")
    B = torch.randn(size, size, device="cuda")
    out = kernel_fn(I, B)
    assert torch.allclose(out, B, atol=ATOL, rtol=RTOL), (
        f"{kernel_name} identity test failed at size={size}"
    )


# ---------------------------------------------------------------------------
# Edge case: zero matrix — output should be all zeros
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP, reason=SKIP_MSG)
@pytest.mark.parametrize("kernel_name,kernel_fn", KERNELS.items())
def test_zero_matrix(kernel_name, kernel_fn):
    A = torch.zeros(256, 256, device="cuda")
    B = torch.randn(256, 256, device="cuda")
    out = kernel_fn(A, B)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), (
        f"{kernel_name} zero-matrix test failed"
    )


# ---------------------------------------------------------------------------
# Input validation — should raise on bad inputs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP, reason=SKIP_MSG)
@pytest.mark.parametrize("kernel_name,kernel_fn", KERNELS.items())
def test_dimension_mismatch_raises(kernel_name, kernel_fn):
    A = torch.randn(64, 32, device="cuda")
    B = torch.randn(64, 64, device="cuda")  # inner dim mismatch
    with pytest.raises(Exception):
        kernel_fn(A, B)


@pytest.mark.skipif(SKIP, reason=SKIP_MSG)
@pytest.mark.parametrize("kernel_name,kernel_fn", KERNELS.items())
def test_cpu_tensor_raises(kernel_name, kernel_fn):
    A = torch.randn(64, 64)       # CPU
    B = torch.randn(64, 64, device="cuda")
    with pytest.raises(Exception):
        kernel_fn(A, B)


# ---------------------------------------------------------------------------
# Numerical precision across multiple random seeds
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP, reason=SKIP_MSG)
@pytest.mark.parametrize("kernel_name,kernel_fn", KERNELS.items())
@pytest.mark.parametrize("seed", [0, 42, 1337, 9999])
def test_numerical_stability(kernel_name, kernel_fn, seed):
    torch.manual_seed(seed)
    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    expected = ref(A, B)
    out = kernel_fn(A, B)
    max_err = (out - expected).abs().max().item()
    assert max_err < 1e-3, (
        f"{kernel_name} numerical stability failed (seed={seed}): "
        f"max_err={max_err:.2e}"
    )


if __name__ == "__main__":
    if SKIP:
        print(SKIP_MSG)
        sys.exit(1)

    print("Running correctness tests...")
    passed = failed = 0

    for name, fn in KERNELS.items():
        for size in [64, 128, 256, 512]:
            A = torch.randn(size, size, device="cuda")
            B = torch.randn(size, size, device="cuda")
            out = fn(A, B)
            expected = ref(A, B)
            err = (out - expected).abs().max().item()
            status = "PASS" if err < ATOL else "FAIL"
            print(f"  [{status}] {name:>12} {size}x{size}  max_err={err:.2e}")
            if status == "PASS":
                passed += 1
            else:
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
