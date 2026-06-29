"""
STRETCH — Write your own CUDA kernel (GELU).
============================================
PyTorch's `a @ b` on a GPU launches a hand-optimized *kernel*. Here you write one yourself: a
tiny CUDA kernel for GELU, compiled at runtime with `load_inline`, where each GPU thread computes
one element. Then we check it matches PyTorch.

⚠️  CUDA-ONLY. This needs an NVIDIA GPU + CUDA toolkit, so it runs on **Google Colab** (Runtime →
    GPU), NOT on a CPU or an Apple (MPS) GPU. It was written against the standard `load_inline`
    pattern but could NOT be tested on the author's machine (no CUDA here) — if a line needs a
    nudge on your Colab run, tell us. On non-CUDA it just prints a note and exits.

    python cuda_gelu_stretch.py        (Google Colab, Runtime → GPU)
"""
import torch

if not torch.cuda.is_available():
    print("This stretch needs an NVIDIA GPU (CUDA). Open the chapter notebook in Colab, set")
    print("Runtime → Change runtime type → GPU, and run this there.")
    raise SystemExit(0)

from torch.utils.cpp_extension import load_inline

# One CUDA kernel: each thread handles one element of the input.
cuda_source = r"""
__global__ void gelu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;        // which element this thread owns
    if (i < n) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

torch::Tensor gelu(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;             // enough blocks to cover all n elements
    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""
cpp_source = "torch::Tensor gelu(torch::Tensor x);"

print("compiling your CUDA kernel (first run takes ~30-60s)...")
ext = load_inline(name="gelu_ext", cpp_sources=cpp_source, cuda_sources=cuda_source,
                  functions=["gelu"], verbose=False)

x = torch.randn(100_000, device="cuda")
ours = ext.gelu(x)
ref = torch.nn.functional.gelu(x, approximate="tanh")     # match the tanh approximation we coded
maxdiff = (ours - ref).abs().max().item()
print(f"max difference vs torch GELU: {maxdiff:.2e}")
print("✅ your CUDA kernel matches PyTorch!" if maxdiff < 1e-4 else "❌ mismatch — check the formula")
print("\nEach of the 100,000 elements was computed by its own GPU thread, in parallel. That's a")
print("kernel — and PyTorch's built-ins are kernels like this, hand-tuned. Welcome to llm.c territory.")
