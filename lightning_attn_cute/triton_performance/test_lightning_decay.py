import os 
import math
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import math
import itertools
import triton

from lightning_attention_triton import lightning_attn_triton, fwd_kernel_v4


# from flashinfer import single_prefill_with_kv_cache
# from flash_attn import flash_attn_func

# Add a new environment variable  


def _build_slope_tensor(n_attention_heads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):  # output 2^(- 8h / H) for h in 1, 2, ... H
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    # h, 1, 1
    slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
        n_attention_heads, 1, 1
    )

    return slopes

def set_seed(seed=42):
    # Python 随机模块
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU（如果有）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有种子

        # cuDNN 确定性模式（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 关闭自动寻找最优卷积算法

    # 设置环境变量（针对某些CUDA版本）
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 针对某些CUDA操作

def torch_lightning_attn(q, k, v, q_decay, k_decay, diag_decay, block_decay, BLOCK):

    B, H, N, d = q.shape

    NUM_BLOCK = (N + BLOCK - 1) // BLOCK


    kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)

    output = torch.empty((B, H, N, d), dtype=q.dtype, device=q.device)
    for i in range(NUM_BLOCK):
        si = i * BLOCK
        ei = si + BLOCK
        m = ei - si
        assert m == BLOCK
        qi = q[:, :, si:ei].contiguous()
        ki = k[:, :, si:ei].contiguous()
        vi = v[:, :, si:ei].contiguous()
        q_times_decay = qi * q_decay[:, :m]

        kv_f16 = kv.to(torch.float16)

        qkv_none_diag = torch.matmul(q_times_decay, kv_f16)

        # diag
        qk = (
                torch.matmul(qi, ki.transpose(-1, -2))
                * diag_decay[:, :, :m, :m]
        )
        qkv_diag = torch.matmul(qk, vi)
        output[:, :, si:ei] = qkv_none_diag + qkv_diag
        kv = block_decay * kv + torch.matmul(
            (ki * k_decay[:, -m:]).transpose(-1, -2), vi)

    return output


def assert_close(actual, expected, atol=1e-5, rtol=1e-3, max_mismatch_ratio=0.001):

    close_mask = torch.isclose(actual, expected, atol=atol, rtol=rtol)

    mismatch_count = (~close_mask).sum().item()
    total_elements = close_mask.numel()
    mismatch_ratio = mismatch_count / total_elements

    if mismatch_ratio > max_mismatch_ratio:
        abs_diff = torch.abs(actual - expected)
        rel_diff = torch.abs((actual - expected) / torch.where(expected != 0, expected, torch.ones_like(expected)))
        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()

        raise AssertionError(
            f"Mismatch ratio {mismatch_ratio:.6f} exceeds threshold {max_mismatch_ratio}\n"
            f"Mismatched elements: {mismatch_count}/{total_elements}\n"
            f"Max absolute difference: {max_abs_diff}\n"
            f"Max relative difference: {max_rel_diff}"
        )


def compute_decay(q, BLOCK):
    B, H, N, d = q.shape
    array = torch.arange(BLOCK).to(q) + 1
    slope_rate = _build_slope_tensor(H).to(q.device)
    q_decay = torch.exp(-slope_rate * array.reshape (-1, 1)).to(torch.float16)
    k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1))).to(torch.float16)
    index = array[:, None] - array[None, :]
    s_index = (
            slope_rate
            * index[
                None,
                None,
            ]
    )
    s_index = torch.where(index >= 0, -s_index, float("-inf"))
    diag_decay = torch.exp(s_index).to(torch.float16)
    block_decay = torch.exp(-slope_rate * BLOCK).to(torch.float32)


    q_decay_cute = q_decay.expand(-1, -1, d).to(torch.float16).contiguous()
    k_decay_cute = k_decay.expand(-1, -1, d).to(torch.float16).contiguous()
    diag_decay_cute = diag_decay.squeeze(dim=0).to(torch.float16).contiguous()



    block_decay_cute = block_decay.squeeze().to(torch.float32)
    if block_decay_cute.dim() == 0:
        block_decay_cute = block_decay_cute.unsqueeze(0)
    # print(f"block_decay_cute shape: {block_decay_cute.shape}")
    # print(f"H: {H}")

    assert q_decay_cute.shape == (H, BLOCK, d)
    assert k_decay_cute.shape == (H, BLOCK, d)
    assert diag_decay_cute.shape == (H, BLOCK, BLOCK)
    assert block_decay_cute.shape == (H,)

    # for t in (q_decay_cute, k_decay_cute, diag_decay_cute, block_decay_cute):
    #     print(f"min : {torch.min(t)}， max: {torch.max(t)}")

    return slope_rate, q_decay, k_decay, diag_decay, block_decay, q_decay_cute, k_decay_cute, diag_decay_cute, block_decay_cute

def veryfy_correct_result(q, k, v, myflash, BLOCK):

    B, H, N, d = q.shape
    slope_rate, q_decay, k_decay, diag_decay, block_decay, q_decay_cute, k_decay_cute, diag_decay_cute, block_decay_cute = compute_decay(q, BLOCK)
    torch_output = torch_lightning_attn(q, k, v, q_decay, k_decay, diag_decay, block_decay, BLOCK)
    cute_output = myflash.forward_with_decay(q, k, v, q_decay_cute, k_decay_cute, diag_decay_cute, block_decay_cute)
    triton_output= lightning_attn_triton(q, k, v, slope_rate, BLOCK)
    assert_close(torch_output, cute_output)
    print("✅ Torch and cute match")
    assert_close(torch_output, triton_output)
    print("✅ Torch and triton match")

# get q, k, v with shape (B, H, N, d)
def get_random_qkv(B, H, N, d):
    q = torch.randn(B, N, H, d).cuda().half()
    k = torch.randn(B, N, H, d).cuda().half()
    v = torch.randn(B, N, H, d).cuda().half()
    q1 = q.transpose(1, 2).contiguous()
    k1 = k.transpose(1, 2).contiguous()
    v1 = v.transpose(1, 2).contiguous()
    return q1, k1, v1

def run_benchmark(BLOCK):
    # batch_size_range = [2 ** i for i in range(0, 6)]
    batch_size_range = [1, 4, 32]
    seq_length_range = [256, 512, 1024]
    configs = list(itertools.product(batch_size_range, seq_length_range))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["torch_native", "cute", "triton"],
            line_names=[
                "torch_native",
                "triton",
                "cute",
            ],
            styles=[("blue", "-"), ("green", "-"), ("red", "--")],
            ylabel="us",
            plot_name="lightning-attention-prefill-performance",
            args={},
        )
    )
    def benchmark(batch_size, seq_len, provider):
        dtype = torch.bfloat16
        device = torch.device("cuda")

        q, k, v = get_random_qkv(B = batch_size, H = 64, N = seq_len, d = 64)

        quantiles = [0.5, 0.2, 0.8]
        slope_rate, q_decay, k_decay, diag_decay, block_decay, q_decay_cute, k_decay_cute, diag_decay_cute, block_decay_cute = compute_decay(q, BLOCK)
        def run_lib():
            if provider == "torch_native":
                torch_lightning_attn(q, k, v, q_decay, k_decay, diag_decay, block_decay, BLOCK)
            elif provider == "triton":
                lightning_attn_triton(q, k, v, slope_rate, BLOCK)
            elif provider == "cute":
                myflash.forward_with_decay(q, k, v, q_decay_cute, k_decay_cute, diag_decay_cute, block_decay_cute)
            else:
                raise ValueError("Unknown provider")

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_lib(),
            quantiles=quantiles,
        )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    benchmark.run(print_data=True)


if __name__ == "__main__":

    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


    torch.manual_seed(0)
    # Load the CUDA kernel as a python module
    myflash = load(name='myflash',
                   sources=[
                       'main.cpp',
                       'light_attention.cu',
                   ],
                   extra_cuda_cflags=[
                       '-O2',
                       '-lcublas',
                       '-lcublasLt',
                       '-std=c++17',
                       '-I/root/cutlass/include',
                       '-I/root/cutlass/tools/util/include',
                   ],
                   )


    set_seed(10086)
    BLOCK = 64
    # make sure cute and triton implmentations are the same as torch
    for i in range(3):
        q, k, v = get_random_qkv(B = 16, H = 64, N = 2048, d = 64)
        veryfy_correct_result(q, k, v, myflash, BLOCK)

    # compare performance
    run_benchmark(BLOCK)


