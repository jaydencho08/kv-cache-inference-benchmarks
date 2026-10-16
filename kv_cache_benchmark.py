# %% [markdown]
# # KV-cache reuse on a Colab NVIDIA T4
# This is a controlled latency/memory experiment, not a model-quality benchmark.
# Recorded T4 results are in results/20261011-174734; rerun to measure your environment.
# Three modes: no_cache, fresh_cache, and prefix_reuse.
# Prefix reuse copies a resident GPU cache; copy cost is INCLUDED in request latency.
# This is not a paged/shared-block serving engine such as vLLM.

# %% [markdown]
# ## 1. Install in a fresh Colab GPU runtime
# Run this cell first. If Transformers was already imported, restart the session
# after installing, then continue from cell 2. Keep Colab's supplied CUDA PyTorch.
# The pinned Transformers release is intentional: cache APIs change over time.

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "transformers==4.57.1", "accelerate==1.11.0",
                       "pandas", "matplotlib"])

# %% [markdown]
# ## 2. Configuration and hardware record
# Choose Runtime > Change runtime type > GPU. Confirm the printed GPU is a T4.
# Start with quick. For your final study use research, and repeat in independent
# sessions. FP16 is the default for a T4; do not switch this experiment to BF16.
# Change one factor at a time. Model and precision changes require a fresh session.

# %%
import copy, gc, hashlib, importlib.metadata, json, math, platform, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

assert torch.cuda.is_available(), "Select a GPU runtime in Colab, then rerun."
assert transformers.__version__ == "4.57.1", "Restart the session after installation."
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = None  # Paste environment.json's model_revision to reproduce a prior run.
PROFILE = "quick"  # quick: pipeline check; research: more settings and repetitions
SEED = 42
DTYPE = torch.float16
DEVICE = torch.device("cuda:0")
WARMUPS = 2
REPEATS = 3 if PROFILE == "quick" else 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

OUT = Path("kv_results") / time.strftime("%Y%m%d-%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)
REVISION = REVISION or model_info(MODEL_ID).sha
gpu = torch.cuda.get_device_properties(0)
environment = {
    "model_id": MODEL_ID, "model_revision": REVISION,
    "python": platform.python_version(), "torch": torch.__version__,
    "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
    "gpu": gpu.name, "gpu_total_GiB": gpu.total_memory / 2**30,
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "dtype": str(DTYPE), "attention": "sdpa", "seed": SEED,
    "profile": PROFILE, "warmups": WARMUPS, "repeats": REPEATS,
}
print(json.dumps(environment, indent=2))
(OUT / "environment.json").write_text(json.dumps(environment, indent=2))
(OUT / "pip-freeze.txt").write_text(subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze"], text=True))
try:
    smi = subprocess.check_output(["nvidia-smi"], text=True)
    print(smi)
    (OUT / "nvidia-smi.txt").write_text(smi)
except (FileNotFoundError, subprocess.CalledProcessError):
    print("nvidia-smi unavailable; PyTorch hardware record is still saved.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=REVISION, dtype=DTYPE, attn_implementation="sdpa"
).to(DEVICE).eval()
assert model.config.model_type == "qwen2", "This notebook is scoped to Qwen2/2.5."
assert not getattr(model.config, "use_sliding_window", False)
(OUT / "model-config.json").write_text(model.config.to_json_string())
print("Model loaded. Allocated GiB:", round(torch.cuda.memory_allocated() / 2**30, 3))

# %% [markdown]
# ## 3. Exact-length inputs and benchmark functions
# Prompts are controlled token sequences derived from text. They are deliberately
# not a task-accuracy dataset. Prefixes match at the TOKEN level. Every mode in a
# paired trial sees identical input IDs. Suffixes vary between trials/batch rows.
# No padding, beam search, sampling, quantization, compilation, or CPU offload.
# Generate exactly G tokens even if EOS appears, so every mode does equal work.
# The full context is never shortened when a prefix is reused.

# %%
CORPUS = (
    "A research team measures transformer inference on a graphics processor. "
    "They keep the model fixed, record the hardware, and repeat each experiment. "
    "Attention uses keys and values from earlier tokens to predict the next token. "
    "A shared document can appear before different questions in many requests. "
)
SUFFIX_CORPUS = (
    "Explain the observations using latency, memory, and throughput. "
    "Compare the conditions and identify sources of uncertainty. "
    "Describe how the answer changes as the context becomes longer. "
)
base_tokens = tokenizer.encode(CORPUS * 1000, add_special_tokens=False)
suffix_tokens = tokenizer.encode(SUFFIX_CORPUS * 1000, add_special_tokens=False)

def make_ids(length, prefix_length, batch=1, sample=0):
    assert 0 <= prefix_length < length
    prefix = base_tokens[:prefix_length]
    rows = []
    for b in range(batch):
        offset = (sample * 17 + b * 29) % 1000
        suffix = suffix_tokens[offset:offset + length - prefix_length]
        row = prefix + suffix
        assert len(row) == length
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=DEVICE)

def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)

def cache_bytes(cache):
    if cache is None:
        return 0
    return sum(t.numel() * t.element_size() for layer in cache.layers
               for t in (layer.keys, layer.values) if t is not None)

def expected_cache_bytes(batch, tokens):
    c = model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return (2 * c.num_hidden_layers * batch * tokens * c.num_key_value_heads
            * head_dim * next(model.parameters()).element_size())

@torch.inference_mode()
def build_prefix(ids, prefix_length):
    """Separate setup cost, not included in warm-hit request timings."""
    assert 0 < prefix_length < ids.shape[1]
    mask = torch.ones_like(ids[:, :prefix_length])
    positions = torch.arange(prefix_length, device=DEVICE)
    sync()
    start = time.perf_counter()
    cache = DynamicCache(config=model.config)
    out = model(input_ids=ids[:, :prefix_length], attention_mask=mask,
                cache_position=positions, past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    sync()
    elapsed_ms = 1000 * (time.perf_counter() - start)
    cache = out.past_key_values
    del out
    assert cache.get_seq_length() == prefix_length
    return cache, elapsed_ms

@torch.inference_mode()
def request(ids, new_tokens, mode, master=None, prefix_length=0):
    """Synchronized host wall time, including Python dispatch and cache copying.

    TTFT starts with GPU-resident input IDs, and ends at the first GPU token ID.
    Tokenization, network, queueing, input transfer, and CPU text decoding excluded.
    There is one synchronization at the prefill/decode boundary, not per token.
    """
    assert mode in ("no_cache", "fresh_cache", "prefix_reuse")
    batch, length = ids.shape
    assert new_tokens >= 2
    assert length + new_tokens <= model.config.max_position_embeddings
    if mode == "prefix_reuse" and prefix_length:
        assert master is not None and master.get_seq_length() == prefix_length
    # Prepare identical bookkeeping buffers outside the timed region.
    tokens = torch.empty((batch, length + new_tokens), dtype=torch.long, device=DEVICE)
    tokens[:, :length] = ids
    mask = torch.ones_like(tokens)
    positions = torch.arange(length + new_tokens, device=DEVICE)
    gc.collect()
    sync()
    is_cuda = DEVICE.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        base_alloc = torch.cuda.memory_allocated(DEVICE)
    else:
        base_alloc = 0  # CPU path is for local correctness tests, not results.

    start = time.perf_counter()
    cached = mode != "no_cache"
    cache = None
    reused = prefix_length if mode == "prefix_reuse" else 0
    copy_ms = 0.0
    if reused:
        # A forward pass mutates DynamicCache, so preserve the master for next use.
        copy_start = time.perf_counter()
        cache = copy.deepcopy(master)
        sync()
        copy_ms = 1000 * (time.perf_counter() - copy_start)
    elif cached:
        cache = DynamicCache(config=model.config)
    out = model(input_ids=tokens[:, reused:length], attention_mask=mask[:, :length],
                cache_position=positions[reused:length], past_key_values=cache,
                use_cache=cached, logits_to_keep=1)
    cache = out.past_key_values if cached else None
    first_logits = out.logits[:, -1, :]
    tokens[:, length] = first_logits.argmax(dim=-1)
    del out
    sync()
    first_done = time.perf_counter()

    for i in range(1, new_tokens):
        end = length + i
        begin = end - 1 if cached else 0
        out = model(input_ids=tokens[:, begin:end], attention_mask=mask[:, :end],
                    cache_position=positions[begin:end], past_key_values=cache,
                    use_cache=cached, logits_to_keep=1)
        cache = out.past_key_values if cached else None
        tokens[:, end] = out.logits[:, -1, :].argmax(dim=-1)
        del out
    sync()
    finished = time.perf_counter()
    ttft = 1000 * (first_done - start)
    decode = 1000 * (finished - first_done)
    total = 1000 * (finished - start)
    peak = torch.cuda.max_memory_allocated(DEVICE) if is_cuda else 0
    metrics = {
        "ttft_ms": ttft, "decode_ms": decode, "total_ms": total,
        "mean_tpot_ms": decode / (new_tokens - 1),
        "decode_tokens_per_s": batch * (new_tokens - 1) * 1000 / decode,
        "e2e_tokens_per_s": batch * new_tokens * 1000 / total,
        "copy_ms": copy_ms, "peak_alloc_MiB": peak / 2**20,
        "base_alloc_MiB": base_alloc / 2**20,
        "incremental_peak_MiB": (peak - base_alloc) / 2**20,
        "peak_reserved_MiB": (torch.cuda.max_memory_reserved(DEVICE) / 2**20
                              if is_cuda else 0),
        "request_kv_MiB": cache_bytes(cache) / 2**20,
        # G tokens emitted, but the last token has not been fed into the model.
        "expected_request_kv_MiB": (expected_cache_bytes(batch, length + new_tokens - 1)
                                    / 2**20 if cached else 0),
    }
    if cached:
        assert cache.get_seq_length() == length + new_tokens - 1
        assert cache_bytes(cache) == expected_cache_bytes(batch, length + new_tokens - 1)
    result = (metrics, tokens[:, length:].cpu(), first_logits.float().cpu())
    del cache, tokens, first_logits
    return result

# %% [markdown]
# ## 4. Correctness gate before timing sweeps
# Compare logits and generated tokens on the same prompt. Floating-point attention
# kernels can differ slightly between full prefill and split prefill. Record those
# differences. Large differences indicate an invalid experiment. A rare greedy
# token divergence can occur near a tie; it needs inspection, not a speed claim.

# %%
ids = make_ids(128, 64, sample=0)
master, _ = build_prefix(ids, 64)
checks = {mode: request(ids, 8, mode, master, 64)
          for mode in ("no_cache", "fresh_cache", "prefix_reuse")}
reference = checks["fresh_cache"]
correctness = []
for mode, (_, generated, logits) in checks.items():
    diff = logits - reference[2]
    relative_l2 = (diff.norm() / reference[2].norm().clamp_min(1e-8)).item()
    correctness.append({"mode": mode, "relative_logit_l2": relative_l2,
                        "max_abs_logit_difference": diff.abs().max().item(),
                        "same_generated_tokens": torch.equal(generated, reference[1])})
    assert relative_l2 < 0.01, "Large logit discrepancy: investigate before benchmarking."
assert master.get_seq_length() == 64, "The reusable master cache was mutated."
print(pd.DataFrame(correctness).to_string(index=False))
pd.DataFrame(correctness).to_csv(OUT / "correctness.csv", index=False)
del checks, reference, master, ids
gc.collect()
torch.cuda.empty_cache()  # Only here, before warmup; never inside timed requests.

# %% [markdown]
# ## 5. Run paired experiments
# A: ordinary KV reuse during decoding, no_cache vs fresh_cache.
# B: prefix reuse, fresh_cache vs prefix_reuse, fixed total input length.
# C: batch-size scaling, same total input length and prefix ratio.
# Each case is warmed up; mode order and case order are randomized. Trials use
# different suffixes. Prefix cache creation is separately repeated and recorded.
# Quick has 3 repeats and only checks the pipeline; research has 10 repeats.
# Use 20-30 repeats at selected settings for final claims. Do not claim p99 here.
# OOM is recorded as a result. Unexpected exceptions stop the experiment.

# %%
if PROFILE == "quick":
    a_lengths, a_outputs = [256, 1024], [32]
    b_length, b_ratios, b_outputs = 1024, [0.0, 0.5, 0.9], [32]
    c_batches = [1, 2]
else:
    a_lengths, a_outputs = [128, 512, 2048], [16, 64]
    b_length, b_ratios, b_outputs = 2048, [0.0, 0.25, 0.5, 0.75, 0.9], [16, 128]
    c_batches = [1, 2, 4]

cases = []
for length in a_lengths:
    for generated in a_outputs:
        cases.append(("A", length, generated, 1, 0, ("no_cache", "fresh_cache")))
for ratio in b_ratios:
    for generated in b_outputs:
        cases.append(("B", b_length, generated, 1, int(b_length * ratio),
                      ("fresh_cache", "prefix_reuse")))
for batch in c_batches:
    cases.append(("C", 1024, 32, batch, 768, ("fresh_cache", "prefix_reuse")))
rng = random.Random(SEED)
rng.shuffle(cases)
rows, build_rows, failures = [], [], []

def run_case(case, case_id):
    experiment, length, generated, batch, p, modes = case
    meta = dict(experiment=experiment, case_id=case_id, input_tokens=length,
                output_tokens=generated, batch=batch, prefix_tokens=p,
                prefix_fraction=p / length)
    master = None
    ids = make_ids(length, p, batch)
    if p:
        for _ in range(WARMUPS):
            temp, _ = build_prefix(ids, p)
            del temp
        for rep in range(REPEATS):
            temp, cost = build_prefix(ids, p)
            build_rows.append({**meta, "rep": rep, "build_ms": cost,
                               "prefix_kv_MiB": cache_bytes(temp) / 2**20})
            del temp
        master, _ = build_prefix(ids, p)
    pool_mib = cache_bytes(master) / 2**20
    # The SAME resident master is present for both modes in B and C.
    # Thus raw peak_alloc includes a controlled, constant prefix pool in both.
    for mode in modes:
        for _ in range(WARMUPS):
            request(ids, generated, mode, master, p)
    for rep in range(REPEATS):
        ids = make_ids(length, p, batch, sample=rep + 1)
        digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
        order = list(modes)
        rng.shuffle(order)
        paired = {}
        for order_index, mode in enumerate(order):
            paired[mode] = request(ids, generated, mode, master, p)
            metrics = paired[mode][0]
            rows.append({**meta, "rep": rep, "mode": mode, "order": order_index,
                         "input_sha256": digest, "resident_prefix_MiB": pool_mib,
                         **metrics})
            pd.DataFrame(rows).to_csv(OUT / "raw.csv", index=False)
        ref_tokens, ref_logits = paired["fresh_cache"][1:]
        for row in rows[-len(modes):]:
            _, generated_ids, logits = paired[row["mode"]]
            row["same_generated_tokens"] = torch.equal(generated_ids, ref_tokens)
            row["relative_logit_l2"] = ((logits - ref_logits).norm()
                                        / ref_logits.norm().clamp_min(1e-8)).item()
            assert row["relative_logit_l2"] < 0.01, "Cache correctness failed."
        if master is not None:
            assert master.get_seq_length() == p
    del master, ids

for case_id, case in enumerate(cases):
    print(f"Case {case_id + 1}/{len(cases)}: {case[:5]}", flush=True)
    try:
        run_case(case, case_id)
    except torch.cuda.OutOfMemoryError:
        # Exception traceback/frame references are released after this block.
        failures.append({"case_id": case_id, "settings": str(case), "error": "CUDA OOM"})
        print("OOM recorded. Reduce length/batch/model size for a subsequent run.")
    gc.collect()
    if failures and failures[-1]["case_id"] == case_id:
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT / "raw.csv", index=False)
    pd.DataFrame(build_rows).to_csv(OUT / "prefix_build.csv", index=False)
    pd.DataFrame(failures, columns=["case_id", "settings", "error"]).to_csv(
        OUT / "failures.csv", index=False)

# %% [markdown]
# ## 6. Summaries, paired speedups, break-even, and figures
# Report medians and IQR for each setting. Bootstrap intervals below are within
# this session, not uncertainty across all T4s or all Colab sessions.
# A speedup >1 favors caching. An interval spanning 1 is inconclusive here.
# Prefix build cost is excluded from a warm hit and included in amortization:
# N * T_fresh versus T_build + N * T_reuse. These are model-based projections
# from measured request times, not measurements of a concurrent serving workload.

# %%
df = pd.DataFrame(rows)
assert len(df), "No completed measurements. Check the failures file."
groups = ["experiment", "case_id", "input_tokens", "output_tokens", "batch",
          "prefix_tokens", "prefix_fraction", "mode"]
summary = df.groupby(groups).agg(
    n=("total_ms", "size"), ttft_median_ms=("ttft_ms", "median"),
    ttft_q25_ms=("ttft_ms", lambda x: x.quantile(.25)),
    ttft_q75_ms=("ttft_ms", lambda x: x.quantile(.75)),
    total_median_ms=("total_ms", "median"),
    total_q25_ms=("total_ms", lambda x: x.quantile(.25)),
    total_q75_ms=("total_ms", lambda x: x.quantile(.75)),
    tpot_median_ms=("mean_tpot_ms", "median"),
    decode_tps_median=("decode_tokens_per_s", "median"),
    e2e_tps_median=("e2e_tokens_per_s", "median"),
    copy_median_ms=("copy_ms", "median"),
    peak_alloc_median_MiB=("peak_alloc_MiB", "median"),
    request_kv_MiB=("request_kv_MiB", "median"),
    resident_prefix_MiB=("resident_prefix_MiB", "median"),
).reset_index()
summary.to_csv(OUT / "summary.csv", index=False)
display(summary)

bootstrap_rng = np.random.default_rng(SEED)
speedups, amortization = [], []
for case_id, group in df.groupby("case_id"):
    meta = group.iloc[0][groups[:-1]].to_dict()
    control = "no_cache" if meta["experiment"] == "A" else "fresh_cache"
    treatment = "fresh_cache" if meta["experiment"] == "A" else "prefix_reuse"
    # Exclude partial trials left by an OOM or interrupted run.
    for metric in ("ttft_ms", "total_ms", "mean_tpot_ms"):
        paired = group.pivot(index="rep", columns="mode", values=metric)
        if control not in paired or treatment not in paired:
            continue
        paired = paired[[control, treatment]].dropna()
        if paired.empty:
            continue
        ratios = (paired[control] / paired[treatment]).to_numpy()
        boot = np.median(bootstrap_rng.choice(ratios, (2000, len(ratios))), axis=1)
        speedups.append({**meta, "metric": metric, "control": control,
                         "treatment": treatment, "paired_n": len(ratios),
                         "median_paired_speedup": np.median(ratios),
                         "bootstrap_low": np.quantile(boot, .025),
                         "bootstrap_high": np.quantile(boot, .975)})
    if meta["experiment"] in ("B", "C"):
        paired = group.pivot(index="rep", columns="mode", values="total_ms")
        if not {"fresh_cache", "prefix_reuse"}.issubset(paired.columns):
            continue
        paired = paired[["fresh_cache", "prefix_reuse"]].dropna()
        if paired.empty:
            continue
        fresh, hit = paired.median()[["fresh_cache", "prefix_reuse"]]
        costs = [r["build_ms"] for r in build_rows if r["case_id"] == case_id]
        build = float(np.median(costs)) if costs else 0.0
        saving = fresh - hit
        n_break = math.floor(build / saving) + 1 if saving > 0 else None
        for n in (1, 2, 4, 8, 16, 32):
            amortization.append({**meta, "requests_or_batches": n,
                "prefix_build_ms": build, "fresh_median_ms": fresh,
                "warm_hit_median_ms": hit, "break_even_n": n_break,
                "projected_cold_inclusive_speedup": n * fresh / (build + n * hit)})

speedups = pd.DataFrame(speedups)
amortization = pd.DataFrame(amortization)
speedups.to_csv(OUT / "paired_speedups.csv", index=False)
amortization.to_csv(OUT / "amortization.csv", index=False)
display(speedups)
display(amortization)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for (g, mode), block in summary[summary.experiment == "B"].groupby(["output_tokens", "mode"]):
    block = block.sort_values("prefix_fraction")
    axes[0].plot(block.prefix_fraction, block.ttft_median_ms, "o-", label=f"{mode}, G={g}")
axes[0].set(xlabel="Cached prefix / total prompt tokens", ylabel="Median TTFT (ms)",
            title="Prefix reuse: first-token latency")
for (g, mode), block in summary[summary.experiment == "A"].groupby(["output_tokens", "mode"]):
    block = block.sort_values("input_tokens")
    axes[1].plot(block.input_tokens, block.tpot_median_ms, "o-", label=f"{mode}, G={g}")
axes[1].set(xlabel="Prompt tokens", ylabel="Mean TPOT, median over trials (ms)",
            title="Ordinary KV cache: decoding")
for mode, block in summary[summary.experiment == "C"].groupby("mode"):
    block = block.sort_values("batch")
    axes[2].plot(block.batch, block.e2e_tps_median, "o-", label=mode)
axes[2].set(xlabel="Batch size", ylabel="Aggregate output tokens / second",
            title="Batch throughput")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(OUT / "results.png", dpi=180)
plt.show()
print("Results folder:", OUT)
print("Greedy-token mismatches:", int((df.same_generated_tokens == False).sum()))
print("OOM cases:", len(failures))

# %% [markdown]
# ## 7. Save before Colab disconnects
# Download the zip. It includes raw trials, summary tables, configuration, package
# versions, correctness checks, prefix costs, failures, and the figure. Keep this
# notebook alongside it. Timing results belong only to the recorded environment.

# %%
import shutil
archive = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
from google.colab import files
files.download(archive)

# %% [markdown]
# ## Interpretation and next experiments
# - Prefix reuse should mostly improve TTFT. Ordinary KV caching should mostly
#   improve decode speed versus full recomputation at every token.
# - At the same total context length, fresh_cache and prefix_reuse still attend
#   over the same history during decoding. Do not expect a large TPOT improvement.
# - Longer generated answers dilute the end-to-end percentage gain from prefix reuse.
# - Short prefixes may lose: Python/cache-copy overhead can exceed saved compute.
# - The 0% prefix case is a negative control: fresh_cache and prefix_reuse execute
#   the same algorithm and should be similar within measurement variation.
# - request_kv_MiB measures tensor payload. peak_alloc_MiB includes weights,
#   temporary tensors, input buffers, and (in B/C) a resident master in BOTH modes.
#   peak_reserved_MiB measures allocator reservation and depends on earlier cases.
#   Never call total GPU peak memory 'the KV cache size'.
# - This implementation clones the prefix for safety. It does not physically share
#   prefix blocks across requests, so it cannot demonstrate paged-cache memory savings.
# - Final cache length is L+G-1: the Gth emitted token has not been fed back yet.
# - For full-attention layers, KV bytes = 2 * layers * batch * tokens * KV_heads
#   * head_dim * bytes_per_element. Use KV heads, not query heads, for GQA models.
# - Here N in amortization means sequential requests (batch=1) or sequential batches.
#   The setup computes a standalone prefix; a serving engine can populate its cache
#   while serving the first request and will have different cold-start accounting.
# - Next, vary cache-hit fraction h. Approximate average request time as
#   h*T_hit + (1-h)*T_miss, then validate with an explicit mixed workload. Prefix
#   fraction and hit fraction are different variables. This notebook measures
#   warm hits and a fresh baseline; it does not simulate eviction or lookup misses.
# - Keep reusable documents/system text first and variable questions last. Exact
#   prefix tokens, model weights, positions, masks, and adapters must match. A
#   matching passage after different preceding text is not an exact reusable prefix.
# - Repeat selected settings in at least 3 independent Colab sessions. Use 20-30
#   trials per setting for final medians/IQR. Between-session variation is separate
#   from the within-session bootstrap interval. Do not infer p95/p99 from tiny n.
# - Extend to Qwen2.5-1.5B-Instruct or 4096-token prompts in a fresh session after
#   the smoke test succeeds. Record OOM; do not silently offload or quantize.
# - Test a paged serving engine separately if the research question includes real
#   request scheduling, prefix eviction, concurrency, or shared-block memory.
#
# Sources checked when authoring:
# https://huggingface.co/docs/transformers/v4.57.1/en/kv_cache
# https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation
# https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2
# https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json
# https://docs.pytorch.org/docs/stable/notes/cuda.html
# https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/
