import triton
import triton.language as tl


@triton.jit
def fwd_kernel_v2(
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

    Q += bx * n * d
    K += bx * n * d
    V += bx * n * e
    O += bx * n * e

    block_off = tl.arange(0, BLOCK)
    qk_dim_off = tl.arange(0, d)
    vo_dim_off = tl.arange(0, BLOCK_MODEL) + by * BLOCK_MODEL

    # decay
    head_off = bx % h
    slope = tl.load(S + head_off).to(tl.float32)
    q_decay = tl.exp(-slope * block_off[:, None])
    k_decay = tl.exp(-slope * (BLOCK - block_off[None, :]))
    block_decay = tl.exp(-slope * BLOCK)
    index = block_off[:, None] - block_off[None, :]  # 相对位置 BLOCK x BLOCK
    s_index = -slope * index  # BLOCK * BLOCK
    s_index = tl.where(index >= 0, s_index, float("-inf"))
    diag_decay = tl.exp(s_index)


    kv = tl.zeros((d, BLOCK_MODEL), dtype=tl.float32)

    for i in range(NUM_BLOCK):
        q_off = block_off[:, None] * d + qk_dim_off[None, :]
        q = tl.load(Q + q_off, mask=block_off[:, None] < n, other=0.0).to(tl.float32)

        k_row_off = tl.arange(0, d)
        k_off = k_row_off[:, None] + block_off[None, :] * d
        k_t = tl.load(K + k_off, mask=block_off[None, :] < n, other=0.0).to(tl.float32)

        v_off = block_off[:, None] * e + vo_dim_off[None, :]
        v = tl.load(V + v_off, mask=block_off[:, None] < n, other=0.0).to(tl.float32)

        o_intra = tl.dot(tl.dot(q, k_t) * diag_decay, v)
        o_inter = tl.dot(q * q_decay, kv)
        o = o_intra + o_inter

        o_off = block_off[:, None] * e + vo_dim_off[None, :]
        tl.store(O + o_off, o.to(O.dtype.element_ty), mask=block_off[:, None] < n)

        new_kv = tl.dot(k_t * k_decay, v)
        kv = kv * block_decay + new_kv

        block_off += BLOCK
