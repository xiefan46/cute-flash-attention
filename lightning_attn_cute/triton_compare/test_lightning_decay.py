import os 
import math
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import math

from lightning_attention_triton import lightning_attn_func, fwd_kernel_v4


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

    assert q.dtype == torch.float16
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16
    assert q_decay.dtype == torch.float16
    assert k_decay.dtype == torch.float16
    assert diag_decay.dtype == torch.float16
    assert block_decay.dtype == torch.float32

    B, H, N, d = q.shape

    assert N % BLOCK == 0
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK


    kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)
    kv_output = torch.zeros(NUM_BLOCK, B, H, d, d).to(kv.dtype).to(q.device)
    o_inter_output = torch.zeros(NUM_BLOCK, B, H, BLOCK, d).to(torch.float16).to(q.device)
    o_intra_output = torch.zeros(NUM_BLOCK, B, H, BLOCK, d).to(torch.float16).to(q.device)
    q_decay_out = torch.zeros(NUM_BLOCK, B, H, BLOCK, d).to(torch.float16).to(q.device)
    kv_t_out = torch.zeros(NUM_BLOCK, B, H, d, d).to(torch.float16).to(q.device)

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
        q_decay_out[i] = q_times_decay.detach().clone()
        kv_f16 = kv.to(torch.float16)
        # kv_t_out[i] = torch.transpose(kv_f16, -1, -2).detach().clone()
        kv_t_out[i] = kv_f16.detach().clone()
        qkv_none_diag = torch.matmul(q_times_decay, kv_f16)
        o_inter_output[i] = qkv_none_diag.detach().clone()
        # diag
        qk = (
                torch.matmul(qi, ki.transpose(-1, -2))
                * diag_decay[:, :, :m, :m]
        )
        qkv_diag = torch.matmul(qk, vi)
        o_intra_output[i] = qkv_diag.detach().clone()

        output[:, :, si:ei] = qkv_none_diag + qkv_diag
        kv = block_decay * kv + torch.matmul(
            (ki * k_decay[:, -m:]).transpose(-1, -2), vi)
        kv_output[i] = kv.detach().clone()
    return output, kv_output, o_inter_output, o_intra_output, q_decay_out, kv_t_out


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

def test_forward_with_decay(q, k, v, myflash):

    # Step1: compare accuracy between cute and torch
    B, H, N, d = q.shape
    BLOCK = 64
    num_block = (N + BLOCK - 1) // BLOCK
    array = torch.arange(BLOCK).to(q) + 1
    slope_rate = _build_slope_tensor(H).to(q.device)
    q_decay = torch.exp(-slope_rate * array.reshape(-1, 1)).to(torch.float16)
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

    torch_output, torch_kv_output, torch_o_inter_out, torch_o_intra_out, torch_q_decay_out, torch_kv_t_out = torch_lightning_attn(q, k, v, q_decay, k_decay, diag_decay, block_decay, BLOCK)

    q_decay_cute = q_decay.expand(-1, -1, d).to(torch.float16).contiguous()
    k_decay_cute = k_decay.expand(-1, -1, d).to(torch.float16).contiguous()
    diag_decay_cute = diag_decay.squeeze(dim=0).to(torch.float16).contiguous()

    block_decay_cute = block_decay.squeeze().to(torch.float32)
    if block_decay_cute.dim() == 0:
        block_decay_cute = block_decay_cute.unsqueeze(0)
    print(f"block_decay_cute shape: {block_decay_cute.shape}")
    print(f"H: {H}")

    assert q_decay_cute.shape == (H, BLOCK, d)
    assert k_decay_cute.shape == (H, BLOCK, d)
    assert diag_decay_cute.shape == (H, BLOCK, BLOCK)
    assert block_decay_cute.shape == (H,)


    # print(f"cute decay. q_decay_cute shape: {q_decay_cute.shape}, k_decay_cute shape: {k_decay_cute.shape}, diag_decay_cute shape: {diag_decay_cute.shape}, block_decay_cute shape: {block_decay_cute.shape}")
    # print(f"cute decay. q_decay_cute: {q_decay_cute}, k_decay_cute: {k_decay_cute}, diag_decay_cute: {diag_decay_cute}, block_decay_cute: {block_decay_cute}")


    for t in (q_decay_cute, k_decay_cute, diag_decay_cute, block_decay_cute):
        print(f"min : {torch.min(t)}， max: {torch.max(t)}")

    # cute_output, cute_kv_output, cute_o_inter_out, cute_o_intra_out, cute_q_decay_out, cute_kv_t_out = myflash.forward_with_decay(q, k, v, q_decay_cute, k_decay_cute, diag_decay_cute, block_decay_cute)
    #
    # for i in range(num_block):
    #     # print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
    #     # print(f"cute_kv_output shape: {cute_kv_output[i].shape}")
    #     # torch.testing.assert_close(
    #     #     torch_kv_output[i],
    #     #     cute_kv_output[i],
    #     # )
    #     assert_close(torch_kv_output[i], cute_kv_output[i])
    # print("✅ kv results match")
    #
    # for i in range(num_block):
    #     # print(f"torch_o_intra_out shape: {torch_o_intra_out[i].shape}")
    #     # print(f"cute_o_intra_out shape: {cute_o_intra_out[i].shape}")
    #     # print(f"torch_o_intra_out dtype/device: {torch_o_intra_out[i].dtype} device: {torch_o_intra_out[i].device}")
    #     # print(f"cute_o_intra_out dtype/device: {cute_o_intra_out[i].dtype}, device: {cute_o_intra_out[i].device}")
    #     # torch.testing.assert_close(
    #     #     torch_o_intra_out[i],
    #     #     cute_o_intra_out[i],
    #     # )
    #     assert_close(torch_o_intra_out[i], cute_o_intra_out[i])
    # print("✅ o intra result maches")
    #
    # for i in range(num_block):
    #     # print(f"torch_q_decay_out shape: {torch_q_decay_out[i].shape}")
    #     # print(f"cute_q_decay_out shape: {cute_q_decay_out[i].shape}")
    #     # print(f"torch_q_decay_out dtype: {torch_q_decay_out[i].dtype}")
    #     # print(f"cute_q_decay_out dtype: {cute_q_decay_out[i].dtype}")
    #     # torch.testing.assert_close(
    #     #     torch_q_decay_out[i],
    #     #     cute_q_decay_out[i],
    #     # )
    #     assert_close(torch_q_decay_out[i], cute_q_decay_out[i])
    # print("✅ q_decay_out  result matches")
    #
    # for i in range(num_block):
    #     # print(f"torch_kv_t_out  shape: {torch_kv_t_out[i].shape}")
    #     # print(f"cute_kv_t_out shape: {cute_kv_t_out[i].shape}")
    #     # print(f"torch_kv_t_out dtype: {torch_kv_t_out[i].dtype}")
    #     # print(f"cute_kv_t_out dtype: {cute_kv_t_out[i].dtype}")
    #     # torch.testing.assert_close(
    #     #     torch_kv_t_out[i],
    #     #     cute_kv_t_out[i],
    #     # )
    #     assert_close(torch_kv_t_out[i], cute_kv_t_out[i])
    # print("✅ kv_t_out  result matches")
    #
    # for i in range(num_block):
    #     # print(f"torch_o_inter_out shape: {torch_o_inter_out[i].shape}")
    #     # print(f"cute_o_inter_out shape: {cute_o_inter_out[i].shape}")
    #     # torch.testing.assert_close(
    #     #     torch_o_inter_out[i],
    #     #     cute_o_inter_out[i],
    #     # )
    #     assert_close(torch_o_inter_out[i], cute_o_inter_out[i])
    # print("✅ o inter result matches")
    #
    #
    # for i in range(num_block):
    #     b_torch_output = torch_output[:, :, i * BLOCK : (i + 1) * BLOCK]
    #     b_cute_output =  cute_output[:, :, i * BLOCK : (i + 1) * BLOCK]
    #
    #     # print(f"block: {i}, b_torch_output shape: {b_torch_output.shape}. value: {b_torch_output}")
    #     # print(f"block: {i}, b_cute_output shape: {b_cute_output.shape}. value: {b_cute_output}")
    #
    #     # torch.testing.assert_close(
    #     #     b_torch_output,
    #     #     b_cute_output,
    #     # )
    #     assert_close(b_torch_output, b_cute_output)
    # # assert_close(torch_output, cute_output)
    #
    # print("✅ Torch and cute two implementations match all tensor")


    # Step2: compare accuracy between triton and torch
    triton_output, triton_q_decay_out, triton_k_decay_out, triton_diag_decay_out, triton_block_decay_out, triton_kv_output, triton_o_inter_output, triton_o_intra_output = lightning_attn_func(q, k, v, slope_rate, BLOCK)

    # q_decay = torch.exp(-slope_rate * array.reshape(-1, 1)).to(torch.float16)
    # k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1))).to(torch.float16)
    # index = array[:, None] - array[None, :]
    # s_index = (
    #         slope_rate
    #         * index[
    #             None,
    #             None,
    #         ]
    # )
    # s_index = torch.where(index >= 0, -s_index, float("-inf"))
    # diag_decay = torch.exp(s_index).to(torch.float16)
    # block_decay = torch.exp(-slope_rate * BLOCK).to(torch.float32)




    print(f"torch q_decay shape: {q_decay.squeeze().shape}")
    print(f"torch k_decay shape: {k_decay.shape}")
    print(f"torch diag_decay shape: {diag_decay.shape}")
    print(f"torch block_decay shape: {block_decay.shape}")

    print(f"triton_q_decay_out shape: {triton_q_decay_out.shape}, triton_k_decay_out: {triton_k_decay_out.shape}, triton_diag_decay_out shape: {triton_diag_decay_out.shape}, triton_block_decay_out shape: {triton_block_decay_out.shape}")

    # torch.testing.assert_allclose(q_decay.reshape(BLOCK, ), triton_q_decay_out[0, 0])
    # torch.testing.assert_allclose(k_decay.reshape(BLOCK, ), triton_k_decay_out[0, 0])
    # torch.testing.assert_allclose(diag_decay.reshape(BLOCK, BLOCK), triton_diag_decay_out[0, 0])
    # torch.testing.assert_allclose(block_decay.reshape(1, ), triton_block_decay_out[0, 0] )


    for i in range(num_block):
        # print(f"triton_kv_output[i]: {triton_kv_output[i]}")
        # print(f"torch_kv_output[i]: {torch_kv_output[i]}")
        assert_close(torch_kv_output[i], triton_kv_output[i])
    print("✅ kv results match")

    for i in range(num_block):
        assert_close(torch_o_intra_out[i], triton_o_intra_output[i])
    print("✅ o intra result maches")

    for i in range(num_block):
        assert_close(torch_o_inter_out[i], triton_o_inter_output[i])
    print("✅ o inter result matches")

    for i in range(num_block):
        b_torch_output = torch_output[:, :, i * BLOCK : (i + 1) * BLOCK]
        b_triton_output =  triton_output[:, :, i * BLOCK : (i + 1) * BLOCK]

        assert_close(b_torch_output, b_triton_output)


    print("✅ Torch and cute two implementations match all tensor")



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
    B = 4
    H = 16
    N = 2048
    # NOTE: we only support d = 64!
    d = 64


    q = torch.randn(B, N, H, d).cuda().half()
    k = torch.randn(B, N, H, d).cuda().half()
    v = torch.randn(B, N, H, d).cuda().half()
    q1 = q.transpose(1, 2).contiguous()
    k1 = k.transpose(1, 2).contiguous()
    v1 = v.transpose(1, 2).contiguous()

    # print_decay_tensors(q1)

    test_forward_with_decay(q1, k1, v1, myflash)
