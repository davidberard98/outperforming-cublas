import torch
import triton

torch.ops.load_library("gemm.so")

m, n, k = 6 * 11 * 128, 6 * 12 * 128, 64*8

x, y = torch.randn(m, k, device="cuda", dtype=torch.bfloat16), torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

def fn_handwritten():
    return torch.ops.gemm.mm_v1(x, y)

def fn_torch():
    return torch.matmul(x, y)

def tflops(ms, m, n, k):
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)

torch_ms = triton.testing.do_bench(fn_torch)
handwritten_ms = triton.testing.do_bench(fn_handwritten)

print(f"torch.mm    Latency: {torch_ms:.3f} ms; TFLOPS: {tflops(torch_ms, m, n, k):.3f}")
print(f"Handwritten Latency: {handwritten_ms:.3f} ms; TFLOPS: {tflops(handwritten_ms, m, n, k):.3f}")
