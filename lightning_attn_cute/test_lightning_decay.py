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

    print(f"q_decay expend: {q_decay.expand(-1, -1, BLOCK)}")

    print(f"q_decay expend: {q_decay.expand(-1, -1, BLOCK)}")

    print(f"diag_decay squeeze: {diag_decay.squeeze(dim=0).shape}")

    print(f"block_decay expend: {block_decay.expand(-1, BLOCK, BLOCK).shape}")


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
    # myflash = load(name='myflash',
    #                sources=[
    #                    'main.cpp',
    #                    'light_attn_decay.cu',
    #                ],
    #                extra_cuda_cflags=[
    #                    '-O2',
    #                    '-lcublas',
    #                    '-lcublasLt',
    #                    '-std=c++17',
    #                    '-I/root/cutlass/include',
    #                    '-I/root/cutlass/tools/util/include',
    #                ],
    #                )


    set_seed(10086)
    B = 128
    H = 64
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
