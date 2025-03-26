import torch
import itertools

from benchmarks.benchmark_triton_impl.model.lightning_attention_pytorch import MiniMaxText01LightningAttention
from benchmarks.benchmark_triton_impl.model.lightning_attention_triton import lightning_attn_func
from benchmarks.benchmark_triton_impl.ops.fwd_kernel_v2 import fwd_kernel_v2
from benchmarks.benchmark_triton_impl.ops.fwd_kernel_v0 import fwd_kernel_v0
from benchmarks.benchmark_triton_impl.ops.fwd_kernel_v1 import fwd_kernel_v1
from benchmarks.benchmark_triton_impl.ops.fwd_kernel_v3 import fwd_kernel_v3
from benchmarks.benchmark_triton_impl.util import _build_slope_tensor
import triton
import triton.language as tl

def test_lightning_attention_implementations(model_params):
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 1024
    dtype = torch.bfloat16
    # dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_states = torch.randn(
        batch_size, seq_len, model_params["hidden_size"], dtype=dtype, device=device
    )

    attention_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)

    slope_rate = _build_slope_tensor(model_params["num_attention_heads"]).to(device)

    model_attn = MiniMaxText01LightningAttention(**model_params).to(dtype).to(device)
    model_attn.eval()

    # Run official pytorch impl
    with torch.no_grad():
        model_output, _, _ = model_attn.inference(
            hidden_states, attn_mask=attention_mask, slope_rate=slope_rate
        )


    # Custom implementations
    qkv = model_attn.act(model_attn.qkv_proj(hidden_states))
    new_shape = qkv.size()[:-1] + (model_attn.num_heads, -1)
    qkv = qkv.view(*new_shape)
    q, k, v = torch.split(qkv, [model_attn.head_dim] * 3, dim=-1)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    for kernel_impl in [fwd_kernel_v0, fwd_kernel_v1, fwd_kernel_v2, fwd_kernel_v3]:
        print(f"Check correctness of kernel: {kernel_impl.__name__}")
        lib_output = lightning_attn_func(q, k, v, slope_rate, kernel_impl)

        print(f"Finish kernel launch: {kernel_impl.__name__}")

        lib_output = lib_output.transpose(1, 2).contiguous()
        lib_output = lib_output.view(batch_size, seq_len, -1)
        lib_output = model_attn.norm(lib_output)
        lib_output = torch.sigmoid(model_attn.output_gate(hidden_states)) * lib_output
        lib_output = model_attn.out_proj(lib_output)

        torch.testing.assert_close(
            model_output,
            lib_output,
            rtol=1e-3,
            atol=1e-2,
            msg="Lightning attention implementations produce different results",
        )

        print("✅ Two implementations match")


def get_benchmark():
    batch_size_range = [2 ** i for i in range(0, 7)]  # max 64
    seq_length_range = [256, 512, 1024]
    configs = list(itertools.product(batch_size_range, seq_length_range))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            # "MiniMax-Text-01",
            line_vals=["kernel_v0", "kernel_v3"],
            line_names=[
                "kernel_v0",
                "kernel_v3",
            ],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="lightning-attention-prefill-performance",
            args={},
        )
    )
    def benchmark(batch_size, seq_len, provider):
        dtype = torch.bfloat16
        device = torch.device("cuda")

        params = {
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "head_dim": 96,
            "hidden_act": "gelu",
        }

        hidden_states = torch.randn(
            batch_size, seq_len, params["hidden_size"], dtype=dtype, device=device
        )

        attention_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)

        slope_rate = _build_slope_tensor(params["num_attention_heads"]).to(device)
        model_attn = MiniMaxText01LightningAttention(**params).to(dtype).to(device)
        model_attn.eval()

        quantiles = [0.5, 0.2, 0.8]
        if provider == "MiniMax-Text-01":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: model_attn.inference(
                    hidden_states, attn_mask=attention_mask, slope_rate=slope_rate
                ),
                quantiles=quantiles,
            )
        else:

            if provider == "kernel_v0":
                kernel_impl = fwd_kernel_v0
            elif provider == "kernel_v1":
                kernel_impl = fwd_kernel_v1
            elif provider == "kernel_v2":
                kernel_impl = fwd_kernel_v2
            elif provider == "kernel_v3":
                kernel_impl = fwd_kernel_v3
            else:
                raise ValueError("Unknown provider")

            def run_lib():
                qkv = model_attn.act(model_attn.qkv_proj(hidden_states))
                new_shape = qkv.size()[:-1] + (model_attn.num_heads, -1)
                qkv = qkv.view(*new_shape)
                q, k, v = torch.split(qkv, [model_attn.head_dim] * 3, dim=-1)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                lib_output = lightning_attn_func(q, k, v, slope_rate, kernel_impl)
                lib_output = lib_output.transpose(1, 2).contiguous()
                lib_output = lib_output.view(batch_size, seq_len, -1)
                lib_output = model_attn.norm(lib_output)
                lib_output = (
                        torch.sigmoid(model_attn.output_gate(hidden_states)) * lib_output
                )
                return model_attn.out_proj(lib_output)

            ms, min_ms, max_ms = triton.testing.do_bench(
                run_lib,
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/lightning_attention_prefill/",
        help="Path to save lightning attention prefill benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test first
    # Adapted from https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json
    params = {
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "head_dim": 96,
        "hidden_act": "silu",
    }
    test_lightning_attention_implementations(params)

    # Run performance benchmark
    benchmark = get_benchmark()
    benchmark.run(print_data=True, save_path=args.save_path)
