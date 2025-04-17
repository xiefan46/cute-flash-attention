
import math


import torch
import torch.nn.functional as F
import triton
import triton
import triton.language as tl

@triton.jit
def fwd_kernel_v4(
        Q,
        K,
        V,
        O,
        S,  # log lambda
        b: tl.constexpr,
        h: tl.constexpr,
        n: tl.constexpr,
        d: tl.constexpr,
        e: tl.constexpr,
        BLOCK: tl.constexpr,
        NUM_BLOCK: tl.constexpr,
        BLOCK_MODEL: tl.constexpr,
):
    bx = tl.program_id(0)
    by = tl.program_id(1)

    block_off = tl.arange(0, BLOCK)
    qk_dim_off = tl.arange(0, d)
    vo_dim_off = tl.arange(0, BLOCK_MODEL) + by * BLOCK_MODEL
    k_row_off = tl.arange(0, d)
    # decay
    head_off = bx % h
    slope = tl.load(S + head_off).to(tl.float32)
    q_decay = tl.exp(-slope * block_off[:, None]).to(tl.float16)
    k_decay = tl.exp(-slope * (BLOCK - block_off[None, :])).to(tl.float16)
    block_decay = tl.exp(-slope * BLOCK)
    index = block_off[:, None] - block_off[None, :]  # 相对位置 BLOCK x BLOCK
    s_index = -slope * index  # BLOCK * BLOCK
    s_index = tl.where(index >= 0, s_index, float("-inf"))
    diag_decay = tl.exp(s_index).to(tl.float16)

    kv = tl.zeros((d, BLOCK_MODEL), dtype=tl.float32)

    Q_start = Q + bx * n * d + qk_dim_off[None, :]
    K_start = K + bx * n * d + k_row_off[:, None]
    V_start = V + bx * n * e + vo_dim_off[None, :]
    O_start = O + bx * n * e + vo_dim_off[None, :]

    for i in range(NUM_BLOCK):
        q_off = block_off[:, None] * d
        q = tl.load(Q_start + q_off, mask=block_off[:, None] < n, other=0.0).to(tl.float16)

        k_off = block_off[None, :] * d
        k_t = tl.load(K_start + k_off, mask=block_off[None, :] < n, other=0.0).to(tl.float16)

        vo_off = block_off[:, None] * e
        v = tl.load(V_start + vo_off, mask=block_off[:, None] < n, other=0.0).to(tl.float16)


        qk = tl.dot(q, k_t).to(tl.float16)
        o_intra = tl.dot((qk * diag_decay).to(tl.float16), v).to(tl.float16)

        kv_f16 = kv.to(tl.float16)
        qkv_inter = tl.dot(q, kv_f16).to(tl.float16)
        o_inter = (qkv_inter * q_decay).to(tl.float16)
        o = o_intra + o_inter

        tl.store(O_start + vo_off, o.to(O.dtype.element_ty), mask=block_off[:, None] < n)

        new_kv = tl.dot(k_t * k_decay, v).to(tl.float16)
        kv = kv * block_decay + new_kv.to(tl.float32)

        block_off += BLOCK

def lightning_attn2(q, k, v, s, kernel_impl, BLOCK):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    b, h, n, d = q.shape
    e = v.shape[-1]

    # Pad d to next power of 2
    d_padded = next_power_of_2(d)
    if d_padded != d:
        q_padded = F.pad(q, (0, d_padded - d))
        k_padded = F.pad(k, (0, d_padded - d))
    else:
        q_padded = q
        k_padded = k

    # Pad e to next power of 2
    e_padded = next_power_of_2(e)
    if e_padded != e:
        v_padded = F.pad(v, (0, e_padded - e))
    else:
        v_padded = v

    o_padded = torch.empty((b, h, n, e_padded), dtype=q.dtype, device=q.device)

    # output debug info
    q_decay_out = torch.empty((b, h, BLOCK), dtype=q.dtype, device=q.device)
    k_decay_out = torch.empty((b, h, BLOCK), dtype=q.dtype, device=q.device)
    diag_decay_out = torch.empty((b, h, BLOCK, BLOCK), dtype=q.dtype, device=q.device)
    block_decay_out = torch.empty((b, h), dtype=q.dtype, device=q.device)

    NUM_BLOCK = triton.cdiv(q.shape[2], BLOCK)
    # parallel over channel
    BLOCK_MODEL = min(triton.next_power_of_2(e_padded), 32)



    grid = (b * h, triton.cdiv(e_padded, BLOCK_MODEL))

    # print(f"kernel_impl={kernel_impl}, grid: {grid}")

    kernel_impl[grid](
        q_padded,
        k_padded,
        v_padded,
        o_padded,
        s,
        b,
        h,
        n,
        d_padded,
        e_padded,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        BLOCK_MODEL=BLOCK_MODEL,
    )

    # Remove padding from output
    if e_padded != e:
        o = o_padded[..., :e]
    else:
        o = o_padded

    return o


def is_support(dim):
    return 16 % dim


def next_power_of_2(n):
    return 2 ** (int(math.ceil(math.log(n, 2))))


def lightning_attn_func(q, k, v, s, kernel_impl, BLOCK):
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert d == e
    assert is_support(d) and is_support(e)

    # pad v's feature dim to power of 2
    e_pad = next_power_of_2(e)
    need_pad = e_pad != e
    if need_pad:
        v = F.pad(v, (0, e_pad - e))

    if d > 128:
        # split over head
        if d % 64 == 0:
            m = 64
        elif d % 32 == 0:
            m = 32
        elif d % 16 == 0:
            m = 16
        arr = [m * i for i in range(d // m + 1)]
        if arr[-1] != d:
            arr.append(d)
        n = len(arr)
        o = 0
        for i in range(n - 1):
            start = arr[i]
            end = arr[i + 1]
            q1 = q[..., start:end]
            k1 = k[..., start:end]
            o += lightning_attn2(q1, k1, v, s, kernel_impl, BLOCK)
    else:
        o = lightning_attn2(q, k, v, s, kernel_impl, BLOCK)

    if need_pad:
        o = o[:, :, :, :e]

    return o