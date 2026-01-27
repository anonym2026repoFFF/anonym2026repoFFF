import torch

def bench_matmul(A, B, iters=200, warmup=20):
    for _ in range(warmup):
        _ = A @ B
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = A @ B
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def flops_gemm(m, k, n):
    return 2 * m * k * n


def fmt_flops(flops):
    if flops >= 1e12:
        return f"{flops/1e12:.3f} TFLOPs"
    if flops >= 1e9:
        return f"{flops/1e9:.3f} GFLOPs"
    if flops >= 1e6:
        return f"{flops/1e6:.3f} MFLOPs"
    if flops >= 1e3:
        return f"{flops/1e3:.3f} KFLOPs"
    return f"{flops} FLOPs"


def fmt_rate(flops_per_s):
    # FLOP/s -> nice unit string
    if flops_per_s >= 1e12:
        return f"{flops_per_s/1e12:.2f} TFLOP/s"
    if flops_per_s >= 1e9:
        return f"{flops_per_s/1e9:.2f} GFLOP/s"
    if flops_per_s >= 1e6:
        return f"{flops_per_s/1e6:.2f} MFLOP/s"
    return f"{flops_per_s:.2f} FLOP/s"


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print()

    # ===== USER PARAMS (NO LOOP) =====
    d = 4 #11         # number of sequential calls to simulate
    batch = 4
    seq = 1024
    # =================================

    # OPT-1.3B sizes
    k = 2048
    n_ffn = 8192

    # Your rule: n_seq = int(8192 / (2**d - 1))
    denom = (2**d - 1)
    n_seq = int(n_ffn / denom)
    n_seq = max(1, n_seq)

    m = batch * seq
    dtype = torch.float16

    X = torch.randn(m, k, device=device, dtype=dtype)

    # Fixed weights like real parameters
    W_ffn = torch.randn(k, n_ffn, device=device, dtype=dtype)
    W_seq = torch.randn(k, n_seq, device=device, dtype=dtype)

    # Iterations (avoid huge runtime)
    iters_ffn = 100
    iters_seq = 200

    # Timings (per single matmul call)
    t_ffn_ms = bench_matmul(X, W_ffn, iters=iters_ffn, warmup=20)
    t_seq_ms = bench_matmul(X, W_seq, iters=iters_seq, warmup=20)

    # FLOPs per call
    flops_ffn = flops_gemm(m, k, n_ffn)
    flops_seq = flops_gemm(m, k, n_seq)

    # Throughput per call (FLOP/s)
    thr_ffn = flops_ffn / (t_ffn_ms / 1000.0)
    thr_seq = flops_seq / (t_seq_ms / 1000.0)

    # Sequentially-scaled totals (simulate d sequential calls)
    t_seq_total_ms = d * t_seq_ms
    flops_seq_total = d * flops_seq
    thr_seq_total = flops_seq_total / (t_seq_total_ms / 1000.0)  # equals thr_seq, but kept explicit

    # Ratios (Variant / Baseline)
    time_ratio_total = t_seq_total_ms / t_ffn_ms if t_ffn_ms > 0 else float("inf")
    flops_ratio_total = flops_seq_total / flops_ffn if flops_ffn > 0 else float("inf")
    thr_ratio_per_call = thr_seq / thr_ffn if thr_ffn > 0 else float("inf")
    thr_ratio_total = thr_seq_total / thr_ffn if thr_ffn > 0 else float("inf")

    # "Runtime speed" relative to FFN baseline (higher is faster)
    speed_seq_vs_ffn = t_ffn_ms / t_seq_total_ms if t_seq_total_ms > 0 else float("inf")

    print("=== Shapes ===")
    print(f"Baseline FFN : (m x k) @ (k x n) = ({m} x {k}) @ ({k} x {n_ffn})")
    print(f"Seq variant  : (m x k) @ (k x n) = ({m} x {k}) @ ({k} x {n_seq})")
    print(f"Simulating d={d} sequential calls for the variant")
    print()

    print("=== Baseline: FFN (single call) ===")
    print(f"FLOPs/call   : {fmt_flops(flops_ffn)}")
    print(f"time/call    : {t_ffn_ms:.6f} ms")
    print(f"throughput   : {fmt_rate(thr_ffn)}")
    print()

    print("=== Variant: Sequential (single call) ===")
    print(f"FLOPs/call   : {fmt_flops(flops_seq)}")
    print(f"time/call    : {t_seq_ms:.6f} ms")
    print(f"throughput   : {fmt_rate(thr_seq)}")
    print()

    print("=== Variant: Sequential (TOTAL over d calls) ===")
    print(f"FLOPs/total  : {fmt_flops(flops_seq_total)}")
    print(f"time/total   : {t_seq_total_ms:.6f} ms")
    print(f"throughput   : {fmt_rate(thr_seq_total)}")
    print()

    print("=== Ratios (Variant TOTAL / Baseline) ===")
    print(f"FLOPs ratio      : {flops_ratio_total:.4f}×")
    print(f"Time ratio       : {time_ratio_total:.4f}×")
    print(f"Throughput ratio : {thr_ratio_total:.4f}×")
    print()

    print("=== Simulated Runtime Speed (higher is faster) ===")
    print("Baseline FFN: 1×")
    print(f"Variant (d={d}): {speed_seq_vs_ffn:.4f}×")

if __name__ == "__main__":
    main()
