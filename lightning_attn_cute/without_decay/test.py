import os 
import math
import torch
from torch.utils.cpp_extension import load
import random
import numpy as np
# from flashinfer import single_prefill_with_kv_cache
# from flash_attn import flash_attn_func

# Add a new environment variable  

def lightning_attn_no_decay(
        q, k, v, BLOCK = 64
) -> torch.Tensor:
    B, H, N, d = q.shape
    NUM_BLOCK = (N + BLOCK - 1) // BLOCK
    # kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)
    kv = torch.zeros(B, H, d, d).to(torch.float32).to(q.device)
    kv_output = torch.zeros(NUM_BLOCK, B, H, d, d).to(kv.dtype).to(q.device)

    o_inter_output = torch.zeros(NUM_BLOCK, B, H, BLOCK, d).to(torch.float16).to(q.device)
    o_intra_output = torch.zeros(NUM_BLOCK, B, H, BLOCK, d).to(torch.float16).to(q.device)
    output = torch.empty((B, H, N, d), dtype=torch.float16, device=q.device)

    for i in range(NUM_BLOCK):
        si = i * BLOCK
        ei = min(si + BLOCK, N)
        qi = q[:, :, si:ei].contiguous()
        ki = k[:, :, si:ei].contiguous()
        vi = v[:, :, si:ei].contiguous()
        qkv_none_diag = torch.matmul(qi, kv.to(torch.float16))
        o_inter_output[i] = qkv_none_diag.detach().clone()


        # diag
        qk = torch.matmul(qi, ki.transpose(-1, -2))

        qkv_diag = torch.matmul(qk, vi)
        o_intra_output[i] = qkv_diag.detach().clone()


        output[:, :, si:ei] = qkv_none_diag + qkv_diag

        new_kv = torch.matmul(ki.transpose(-1, -2), vi)
        print(f"new_kv shape: {new_kv.shape}")
        kv = kv + new_kv
        kv_output[i] = kv.detach().clone()
        print(f"data types. qi : {qi.dtype}, ki : {ki.dtype}, vi : {vi.dtype}, qkv_none_diag : {qkv_none_diag.dtype}, qk : {qk.dtype}, qkv_diag: {qkv_diag.dtype}, "
              f"output: {output.dtype}, new_kv: {new_kv.dtype}")


    return output, kv_output, o_inter_output, o_intra_output


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



def test_forward_without_decay_precision(q, k, v):
    torch_output, torch_kv_output, torch_o_inter_out, torch_o_intra_out = lightning_attn_no_decay(q, k, v)
    # cute_output, cute_kv_output, cute_o_inter_out, cute_o_intra_out = myflash.forward_without_decay_precision(q, k, v)
    #
    # BLOCK = 64
    # B, H, N, d = q.shape
    # num_block = (N + BLOCK - 1) // BLOCK
    #
    # print(f"num_block : {num_block}")
    #
    # for i in range(num_block):
    #     print(f"torch_kv_output shape: {torch_kv_output[i].shape}")
    #     print(f"cute_kv_output shape: {cute_kv_output[i].shape}")
    #     torch.testing.assert_close(
    #         torch_kv_output[i],
    #         cute_kv_output[i],
    #         rtol=1e-3,
    #         atol=1e-5,
    #         msg=f"block : {i}, KV results are different.torch_kv_output: {torch_kv_output[i]}. cute_kv_output: {cute_kv_output[i]}",
    #     )
    # print("✅ kv results match")
    #
    # for i in range(num_block):
    #     print(f"torch_o_intra_out shape: {torch_o_intra_out[i].shape}")
    #     print(f"cute_o_intra_out shape: {cute_o_intra_out[i].shape}")
    #     print(f"torch_o_intra_out dtype/device: {torch_o_intra_out[i].dtype} device: {torch_o_intra_out[i].device}")
    #     print(f"cute_o_intra_out dtype/device: {cute_o_intra_out[i].dtype}, device: {cute_o_intra_out[i].device}")
    #     torch.testing.assert_close(
    #         torch_o_intra_out[i],
    #         cute_o_intra_out[i],
    #         # rtol=1e-3,
    #         # atol=1e-2,
    #         msg=f"block : {i}, o_intra results are different.torch_o_intra_out: {torch_o_intra_out[i]}, cute_o_intra_out: {cute_o_intra_out[i]}",
    #     )
    # print("✅ o intra result maches")
    #
    # for i in range(num_block):
    #     print(f"torch_o_inter_out shape: {torch_o_inter_out[i].shape}")
    #     print(f"cute_o_inter_out shape: {cute_o_inter_out[i].shape}")
    #     torch.testing.assert_close(
    #         torch_o_inter_out[i],
    #         cute_o_inter_out[i],
    #         # rtol=1e-3,
    #         # atol=1e-2,
    #         msg=f"block : {i},  o inter results are different. torch_o_inter_out: {torch_o_inter_out[i]}, cute_o_inter_out: {cute_o_inter_out[i]}",
    #     )
    # print("✅ o inter result matches")
    #
    #
    #
    #
    # for i in range(num_block):
    #     b_torch_output = torch_output[:, :, i * BLOCK : (i + 1) * BLOCK]
    #     b_cute_output =  cute_output[:, :, i * BLOCK : (i + 1) * BLOCK]
    #
    #     print(f"block: {i}, b_torch_output shape: {b_torch_output.shape}. value: {b_torch_output}")
    #     print(f"block: {i}, b_cute_output shape: {b_cute_output.shape}. value: {b_cute_output}")
    #
    #     torch.testing.assert_close(
    #         b_torch_output,
    #         b_cute_output,
    #         # rtol=1e-3,
    #         # atol=1e-2,
    #         msg=f"block: {i}, Lightning attention implementations produce different results",
    #     )
    #
    # print("✅ Two implementations match")


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
                       'light_attn_without_decay.cu',
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
    B = 2
    H = 64
    N = 1024
    # NOTE: we only support d = 96!
    d = 96


    q = torch.randn(B, N, H, d).cuda().half()
    k = torch.randn(B, N, H, d).cuda().half()
    v = torch.randn(B, N, H, d).cuda().half()
    q1 = q.transpose(1, 2).contiguous()
    k1 = k.transpose(1, 2).contiguous()
    v1 = v.transpose(1, 2).contiguous()

    test_forward_without_decay_precision(q1, k1, v1)