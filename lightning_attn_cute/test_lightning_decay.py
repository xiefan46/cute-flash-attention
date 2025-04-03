import os 
import math
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import math


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

def compute_qk_manual(q, k):
    return q @ k.transpose(-2, -1)

def manual_attn(q, k, v, attn_mask=None, use_softmax = True):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if attn_mask != None:
        att.masked_fill_(attn_mask, float('-inf'))  # Apply mask
    if use_softmax:
        att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def manual_attn_no_normal(q, k, v, attn_mask=None, use_softmax = True):
    att = q @ k.transpose(-2, -1)
    if attn_mask != None:
        att.masked_fill_(attn_mask, float('-inf'))  # Apply mask
    if use_softmax:
        att = F.softmax(att, dim=-1)
    y = att @ v
    return y


# def lightning_attn_no_decay(
#         q, k, v, BLOCK = 64
# ) -> torch.Tensor:
#     B, H, N, d = q.shape
#     NUM_BLOCK = (N + BLOCK - 1) // BLOCK
#     # kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)
#     kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)
#     kv_output = torch.zeros(NUM_BLOCK, d, d).to(kv.dtype).to(q.device)
#
#     o_inter_output = torch.zeros(NUM_BLOCK, BLOCK, d).to(torch.float32).to(q.device)
#     o_intra_output = torch.zeros(NUM_BLOCK, BLOCK, d).to(torch.float32).to(q.device)
#
#
#     output = torch.empty((B, H, N, d), dtype=torch.float16, device=q.device)
#     for i in range(NUM_BLOCK):
#         si = i * BLOCK
#         ei = min(si + BLOCK, N)
#         qi = q[:, :, si:ei, :].contiguous().to(torch.float32)
#         ki = k[:, :, si:ei, :].contiguous().to(torch.float32)
#         vi = v[:, :, si:ei, :].contiguous().to(torch.float32)
#
#         qkv_none_diag = torch.matmul(qi, kv.to(qi.dtype)).to(torch.float32)
#         o_inter_output[i] = qkv_none_diag.detach().clone()
#
#         # diag
#         qk = torch.matmul(qi, ki.transpose(-1, -2))
#
#         qkv_diag = torch.matmul(qk, vi).to(torch.float32)
#         o_intra_output[i] = qkv_diag.detach().clone()
#
#         output[:, :, si:ei] = (qkv_none_diag + qkv_diag).to(torch.float16)
#         # new_kv = torch.matmul(ki.transpose(-1, -2).to(vi.dtype), vi).to(torch.float32)
#         new_kv = torch.matmul(ki.transpose(-1, -2), vi)
#         kv = kv + new_kv
#         kv_output[i] = kv.detach().clone()
#
#         print(f"data types. qi : {qi.dtype}, ki : {ki.dtype}, vi : {vi.dtype}, qkv_none_diag : {qkv_none_diag.dtype}, qk : {qk.dtype}, qkv_diag: {qkv_diag.dtype}, "
#               f"output: {output.dtype}, new_kv: {new_kv.dtype}")
#
#     return output, kv_output, o_inter_output, o_intra_output


def lightning_attn_no_decay(
        q, k, v, BLOCK = 64
) -> torch.Tensor:
    B, H, N, d = q.shape
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK
    # kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)
    kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)
    kv_output = torch.zeros(NUM_BLOCK, d, d).to(kv.dtype).to(q.device)

    o_inter_output = torch.zeros(NUM_BLOCK, BLOCK, d).to(torch.float32).to(q.device)
    o_intra_output = torch.zeros(NUM_BLOCK, BLOCK, d).to(torch.float32).to(q.device)
    output = torch.empty((B, H, N, d), dtype=torch.float16, device=q.device)

    for i in range(NUM_BLOCK):
        si = i * BLOCK
        ei = min(si + BLOCK, N)
        qi = q[:, :, si:ei].contiguous().to(torch.float32)
        ki = k[:, :, si:ei].contiguous().to(torch.float32)
        vi = v[:, :, si:ei].contiguous().to(torch.float32)
        qkv_none_diag = torch.matmul(qi, kv)
        o_inter_output[i] = qkv_none_diag.detach().clone()


        # diag
        qk = torch.matmul(qi, ki.transpose(-1, -2))

        qkv_diag = torch.matmul(qk, vi)
        o_intra_output[i] = qkv_diag.detach().clone()


        output[:, :, si:ei] = qkv_none_diag + qkv_diag

        new_kv = torch.matmul(ki.transpose(-1, -2), vi)
        print(f"new_kv type: {new_kv.dtype}")
        kv = kv + new_kv
        kv_output[i] = kv.detach().clone()
        print(f"data types. qi : {qi.dtype}, ki : {ki.dtype}, vi : {vi.dtype}, qkv_none_diag : {qkv_none_diag.dtype}, qk : {qk.dtype}, qkv_diag: {qkv_diag.dtype}, "
              f"output: {output.dtype}, new_kv: {new_kv.dtype}")


    return output, kv_output, o_inter_output, o_intra_output


def torch_compute_kv(k, v, BLOCK = 64):
    B, H, N, d = k.shape
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK

    kv = torch.zeros(d, d).to(torch.float32).to(q.device)
    kv_output = torch.zeros(NUM_BLOCK, d, d).to(torch.float32).to(q.device)

    for i in range(NUM_BLOCK):
        si = i * BLOCK
        ei = min(si + BLOCK, N)
        ki = k[:, :, si:ei].contiguous().to(torch.float32)
        vi = v[:, :, si:ei].contiguous().to(torch.float32)

        new_kv = torch.matmul(ki.transpose(-1, -2), vi).to(torch.float32)
        kv = kv + new_kv
        kv_output[i] = kv.detach().clone()
        print(f"data types.ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
    return kv_output

def torch_compute_amp(k, v, BLOCK = 64):

    assert k.dtype == torch.float16
    assert v.dtype == torch.float16

    B, H, N, d = k.shape
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK

    kv = torch.zeros(d, d).to(torch.float32).to(q.device)
    kv_output = torch.zeros(NUM_BLOCK, d, d).to(torch.float32).to(q.device)

    for i in range(NUM_BLOCK):
        si = i * BLOCK
        ei = min(si + BLOCK, N)
        ki = k[:, :, si:ei].contiguous()
        vi = v[:, :, si:ei].contiguous()
        with autocast():
            new_kv = torch.matmul(ki.transpose(-1, -2), vi)
        new_kv = new_kv.to(torch.float32)
        kv = kv + new_kv
        kv_output[i] = kv.detach().clone()
        print(f"data types.ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
    return kv_output


def torch_compute_kv_f16(k, v, BLOCK = 64):
    B, H, N, d = k.shape
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK

    kv = torch.zeros(d, d).to(torch.float16).to(q.device)
    kv_output = torch.zeros(NUM_BLOCK, d, d).to(torch.float16).to(q.device)

    for i in range(NUM_BLOCK):
        si = i * BLOCK
        ei = min(si + BLOCK, N)
        ki = k[:, :, si:ei].contiguous()
        vi = v[:, :, si:ei].contiguous()

        new_kv = torch.matmul(ki.transpose(-1, -2), vi)
        kv = kv + new_kv
        kv_output[i] = kv.detach().clone()
        print(f"data types. ki : {ki.dtype}, vi : {vi.dtype},  new_kv: {new_kv.dtype}, kv: {kv.dtype}")
    return kv_output

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



def test_forward_without_decay(q, k, v):
    torch_output = lightning_attn_no_decay(q, k, v)
    cute_output = myflash.forward_without_decay(q, k, v)

    print(f"torch output: {torch_output}")
    print(f"cute_output: {cute_output}")

    torch.testing.assert_close(
        torch_output,
        cute_output,
        # rtol=1e-3,
        # atol=1e-2,
        msg="Lightning attention implementations produce different results",
    )

    print("✅ Two implementations match")


def test_forward_without_decay_precision(q, k, v):
    torch_output, torch_kv_output, torch_o_inter_out, torch_o_intra_out = lightning_attn_no_decay(q, k, v)
    cute_output, cute_kv_output, cute_o_inter_out, cute_o_intra_out = myflash.forward_without_decay_precision(q, k, v)

    BLOCK = 64
    B, H, N, d = q.shape
    num_block = (N + BLOCK - 1) // BLOCK

    print(f"num_block : {num_block}")

    for i in range(num_block):
        print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
        print(f"cute_kv_output shape: {cute_kv_output[i].shape}")
        torch.testing.assert_close(
            torch_kv_output[i],
            cute_kv_output[i],
            rtol=1e-3,
            atol=1e-5,
            msg=f"block : {i}, KV results are different.torch_kv_output: {torch_kv_output[i]}. cute_kv_output: {cute_kv_output[i]}",
        )
    print("✅ kv results match")

    for i in range(num_block):
        print(f"torch_o_intra_out shape: {torch_o_intra_out[i].shape}")
        print(f"cute_o_intra_out shape: {cute_o_intra_out[i].shape}")
        torch.testing.assert_close(
            torch_o_intra_out[i],
            cute_o_intra_out[i],
            # rtol=1e-3,
            # atol=1e-2,
            msg=f"block : {i}, o_intra results are different.torch_o_intra_out: {torch_o_intra_out[i]}, cute_o_intra_out: {cute_o_intra_out[i]}",
        )
    print("✅ o intra result maches")

    for i in range(num_block):
        print(f"torch_o_inter_out shape: {torch_o_inter_out[i].shape}")
        print(f"cute_o_inter_out shape: {cute_o_inter_out[i].shape}")
        torch.testing.assert_close(
            torch_o_inter_out[i],
            cute_o_inter_out[i],
            # rtol=1e-3,
            # atol=1e-2,
            msg=f"block : {i},  o inter results are different. torch_o_inter_out: {torch_o_inter_out[i]}, cute_o_inter_out: {cute_o_inter_out[i]}",
        )
    print("✅ o inter result matches")




    for i in range(num_block):
        b_torch_output = torch_output[:, :, i * BLOCK : (i + 1) * BLOCK]
        b_cute_output =  cute_output[:, :, i * BLOCK : (i + 1) * BLOCK]

        print(f"block: {i}, b_torch_output shape: {b_torch_output.shape}. value: {b_torch_output}")
        print(f"block: {i}, b_cute_output shape: {b_cute_output.shape}. value: {b_cute_output}")

        torch.testing.assert_close(
            b_torch_output,
            b_cute_output,
            # rtol=1e-3,
            # atol=1e-2,
            msg=f"block: {i}, Lightning attention implementations produce different results",
        )

    print("✅ Two implementations match")


def test_kv_match(k, v, myflash):
    torch_kv_output = torch_compute_kv(k, v)
    cute_kv_output = myflash.cute_compute_kv(k, v)

    BLOCK = 64
    B, H, N, d = k.shape
    num_block = (N + BLOCK - 1) // BLOCK

    print(f"test_kv_match. num_block : {num_block}")

    for i in range(num_block):
        print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
        print(f"cute_kv_output shape: {cute_kv_output[i].shape}")

        print(f"block: {i}, torch_kv_output: {torch_kv_output[i]}, cute_kv_output: {cute_kv_output[i]}")

        # torch.testing.assert_close(
        #     torch_kv_output[i],
        #     cute_kv_output[i],
        #     rtol=1e-3,
        #     atol=1e-5,
        #     msg=f"block : {i}, KV results are different.torch_kv_output: {torch_kv_output[i]}. cute_kv_output: {cute_kv_output[i]}",
        # )

        torch.testing.assert_close(
            torch_kv_output[i],
            cute_kv_output[i],
        )

        print(f"✅ block : {i}, kv results match")

    print("✅ kv results match")


def test_kv_match_f16(k, v, myflash):
    # torch_kv_output = torch_compute_kv_f16(k, v)
    torch_kv_output = torch_compute_kv(k, v)
    cute_kv_output = myflash.cute_compute_kv_all_f16(k, v).to(torch.float32)
    # cute_kv_output = myflash.cute_compute_kv(k, v).half()

    print(f"torch_kv_output dtype: {torch_kv_output.dtype}, cute_kv_output dtype: {cute_kv_output.dtype}")

    BLOCK = 64
    B, H, N, d = k.shape
    num_block = (N + BLOCK - 1) // BLOCK

    print(f"test_kv_match. num_block : {num_block}")

    for i in range(num_block):
        print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
        print(f"cute_kv_output shape: {cute_kv_output[i].shape}")

        print(f"block: {i}, torch_kv_output: {torch_kv_output[i]}, cute_kv_output: {cute_kv_output[i]}")

        # torch.testing.assert_close(
        #     torch_kv_output[i],
        #     cute_kv_output[i],
        #     rtol=1e-3,
        #     atol=1e-5,
        #     msg=f"block : {i}, KV results are different.torch_kv_output: {torch_kv_output[i]}. cute_kv_output: {cute_kv_output[i]}",
        # )

        torch.testing.assert_close(
            torch_kv_output[i],
            cute_kv_output[i],
        )

        print(f"✅ block : {i}, kv results match")

    print("✅ kv results match")


def test_kv_match_amp(k, v, myflash):
    # torch_kv_output = torch_compute_kv_f16(k, v)
    torch_kv_output_f32 = torch_compute_amp(k, v)
    cute_kv_output_f32 = myflash.cute_compute_kv(k, v)


    assert torch_kv_output_f32.dtype == torch.float32
    assert cute_kv_output_f32.dtype == torch.float32

    torch_kv_output = torch_kv_output_f32.to(torch.float16)
    cute_kv_output = cute_kv_output_f32.to(torch.float16)

    # cute_kv_output = myflash.cute_compute_kv(k, v).half()

    print(f"torch_kv_output dtype: {torch_kv_output.dtype}, cute_kv_output dtype: {cute_kv_output.dtype}")

    BLOCK = 64
    B, H, N, d = k.shape
    num_block = (N + BLOCK - 1) // BLOCK

    print(f"test_kv_match. num_block : {num_block}")

    for i in range(num_block):
        print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
        print(f"cute_kv_output shape: {cute_kv_output[i].shape}")

        print(f"block: {i}, torch_kv_output: {torch_kv_output[i]}, cute_kv_output: {cute_kv_output[i]}")

        # torch.testing.assert_close(
        #     torch_kv_output[i],
        #     cute_kv_output[i],
        #     rtol=1e-3,
        #     atol=1e-5,
        #     msg=f"block : {i}, KV results are different.torch_kv_output: {torch_kv_output[i]}. cute_kv_output: {cute_kv_output[i]}",
        # )

        print(f"(24, 21): torch: {torch_kv_output[i, 24, 21]}, cute: {cute_kv_output[i, 24, 21]}")

        torch.testing.assert_close(
            torch_kv_output[i],
            cute_kv_output[i],
            rtol=1e-1,
            atol=1e-2,
        )

        print(f"✅ block : {i}, kv results match")

    print("✅ kv results match")

# index = block_off[:, None] - block_off[None, :]  # 相对位置 BLOCK x BLOCK
# s_index = -slope * index  # BLOCK * BLOCK
# s_index = tl.where(index >= 0, s_index, float("-inf"))
# diag_decay = tl.exp(s_index)
# o_intra = tl.dot(tl.dot(q, k_t) * diag_decay, v)
def test_intra_block_compute(q, k, v):
    pass


def print_decay_tensors(q, BLOCK = 64):
    num_attention_heads = q.size(1)
    slope_rate = _build_slope_tensor(num_attention_heads).to(q.device)
    array = torch.arange(BLOCK).to(q) + 1
    q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
    k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
    index = array[:, None] - array[None, :]
    s_index = (
            slope_rate
            * index[
                None,
                None,
            ]
    )
    s_index = torch.where(index >= 0, -s_index, float("-inf"))
    diag_decay = torch.exp(s_index)
    block_decay = torch.exp(-slope_rate * BLOCK)

    print(f"slope_rate: {slope_rate.shape}, q_decay: {q_decay.shape}, k_decay: {k_decay.shape}, diag_decay: {diag_decay.shape}, block_decay: {block_decay.shape}")

    print(f"q_decay: {q_decay}")

    print(f"k_decay: {k_decay}")


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
                       'light_attn_decay.cu',
                       'light_attn_without_decay_precision.cu',
                       'light_attn_without_decay_precision_all_f16.cu',
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
    B = 1
    H = 1
    N = 512
    # NOTE: we only support d = 64!
    d = 64


    q = torch.randn(B, N, H, d).cuda().half()
    k = torch.randn(B, N, H, d).cuda().half()
    v = torch.randn(B, N, H, d).cuda().half()
    q1 = q.transpose(1, 2).contiguous()
    k1 = k.transpose(1, 2).contiguous()
    v1 = v.transpose(1, 2).contiguous()


    print_decay_tensors(q1)
