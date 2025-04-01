import os 
import math
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
import random
import numpy as np
# from flashinfer import single_prefill_with_kv_cache
# from flash_attn import flash_attn_func

# Add a new environment variable  
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

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
                        'light_attn_without_decay.cu',
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


def lightning_attn_no_decay(
        q, k, v, BLOCK = 64
) -> torch.Tensor:
    B, H, N, d = q.shape
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK
    kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)
    output = torch.empty((B, H, N, d), dtype=q.dtype, device=q.device)
    for i in range(NUM_BLOCK):
        si = i * BLOCK
        ei = min(si + BLOCK, N)
        qi = q[:, :, si:ei, :].contiguous()
        ki = k[:, :, si:ei, :].contiguous()
        vi = v[:, :, si:ei, :].contiguous()
        print(f"qi shape: {qi.shape}")

        qkv_none_diag = torch.matmul(qi, kv.to(qi.dtype)).to(torch.float32)

        print(f"qkv_none_diag: {qkv_none_diag[0, 0, 0:2, 0: 2]}")

        # diag
        qk = (
                torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32)
        )
        qkv_diag = torch.matmul(qk, vi.to(torch.float32))
        output[:, :, si:ei] = qkv_none_diag + qkv_diag
        print(f"output: {output[0, 0, si:si+2, 0: 2]}")
        new_kv = torch.matmul(ki.transpose(-1, -2).to(vi.dtype), vi).to(torch.float32)
        print(f"torch new_kv: {new_kv[0, 0, 0: 2, 0: 2]}")
        kv = kv + new_kv
        print(f"torch kv: {kv[0, 0, 0: 2, 0: 2]}")
    return output


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
        rtol=1e-3,
        atol=1e-2,
        msg="Lightning attention implementations produce different results",
    )

    print("✅ Two implementations match")

# index = block_off[:, None] - block_off[None, :]  # 相对位置 BLOCK x BLOCK
# s_index = -slope * index  # BLOCK * BLOCK
# s_index = tl.where(index >= 0, s_index, float("-inf"))
# diag_decay = tl.exp(s_index)
# o_intra = tl.dot(tl.dot(q, k_t) * diag_decay, v)
def test_intra_block_compute(q, k, v):
    pass

set_seed(10086)
B = 1
H = 1
N = 128
# NOTE: we only support d = 64!
d = 64

q = torch.randn(B, N, H, d).cuda().half()
k = torch.randn(B, N, H, d).cuda().half()
v = torch.randn(B, N, H, d).cuda().half()
q1 = q.transpose(1, 2).contiguous()
k1 = k.transpose(1, 2).contiguous()
v1 = v.transpose(1, 2).contiguous()

test_forward_without_decay(q1, k1, v1)
