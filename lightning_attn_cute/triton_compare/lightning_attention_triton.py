
import math


import torch
import torch.nn.functional as F
import triton

def lightning_attn2(q, k, v, s, kernel_impl):
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

    BLOCK = 16
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


def lightning_attn_func(q, k, v, s, kernel_impl):
    b, h, n, d = q.shape
    e = v.shape[-1]
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
            o += lightning_attn2(q1, k1, v, s, kernel_impl)
    else:
        o = lightning_attn2(q, k, v, s, kernel_impl)

    if need_pad:
        o = o[:, :, :, :e]

    return o