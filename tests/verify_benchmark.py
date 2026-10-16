"""Validate real cache semantics using a tiny random Qwen2 on CPU, no model download."""
import ast
import copy
import gc
import time
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache

torch.set_num_threads(2)
torch.manual_seed(42)
DEVICE = torch.device("cpu")
config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=256, use_sliding_window=False,
                    attention_dropout=0.0, eos_token_id=None, pad_token_id=0)
config._attn_implementation = "sdpa"
model = Qwen2ForCausalLM(config).eval()
path = Path(__file__).resolve().parents[1] / "kv_cache_benchmark.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
names = {"sync", "cache_bytes", "expected_cache_bytes", "build_prefix", "request"}
functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name in names], type_ignores=[])
exec(compile(functions, str(path), "exec"))
checked = 0
for batch, length, prefix_length, generated in [(1, 16, 8, 8), (2, 24, 23, 2),
                                                (2, 32, 16, 8), (1, 24, 0, 8)]:
    ids = torch.randint(1, 64, (batch, length))
    master, _ = build_prefix(ids, prefix_length) if prefix_length else (None, 0)
    snapshots = [(layer.keys.clone(), layer.values.clone()) for layer in master.layers] if master else []
    modes = {mode: request(ids, generated, mode, master, prefix_length)
             for mode in ("no_cache", "fresh_cache", "prefix_reuse")}
    with torch.inference_mode():
        official = model.generate(ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=generated, do_sample=False,
                                  use_cache=True, pad_token_id=0)[:, length:]
    for mode, (metrics, tokens, logits) in modes.items():
        torch.testing.assert_close(logits, modes["fresh_cache"][2], atol=1e-6, rtol=1e-5)
        assert torch.equal(tokens, official), (batch, length, mode, "generation differs")
        assert metrics["ttft_ms"] > 0 and metrics["decode_ms"] > 0
        assert abs(metrics["total_ms"] - metrics["ttft_ms"] - metrics["decode_ms"]) < 1e-6
        assert metrics["request_kv_MiB"] == metrics["expected_request_kv_MiB"]
        checked += 1
    if master:
        assert master.get_seq_length() == prefix_length
        for layer, (k, v) in zip(master.layers, snapshots):
            assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
    print(f"PASS batch={batch}, prompt={length}, prefix={prefix_length}, output={generated}")
print(f"PASS: {checked} mode/shape checks against Hugging Face generate; master cache unchanged.")

# Exercise the actual runner and analysis/export cell with the tiny CPU model.
# These CPU timings are temporary validation artifacts, not GPU measurements.
import hashlib
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, WARMUPS, REPEATS = 42, 1, 2
OUT = Path(__file__).resolve().parent / "validation_results"
OUT.mkdir(exist_ok=True)
rng = random.Random(SEED)
rows, build_rows, failures = [], [], []
def make_ids(length, prefix_length, batch=1, sample=0):
    prefix = torch.arange(prefix_length) % 63 + 1
    suffix = torch.arange(length - prefix_length) % 63 + 1
    result = torch.cat([prefix, (suffix + sample) % 63 + 1])
    return result.repeat(batch, 1)
runner = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "run_case"], type_ignores=[])
exec(compile(runner, str(path), "exec"))
test_cases = [("A", 32, 8, 1, 0, ("no_cache", "fresh_cache")),
              ("B", 32, 8, 1, 0, ("fresh_cache", "prefix_reuse")),
              ("B", 32, 8, 1, 16, ("fresh_cache", "prefix_reuse")),
              ("C", 32, 8, 2, 16, ("fresh_cache", "prefix_reuse"))]
for i, case in enumerate(test_cases):
    run_case(case, i)
notebook = json.loads(path.with_name("kv_cache_t4_colab.ipynb").read_text())
analysis = next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and "".join(c["source"]).startswith("df ="))
display = lambda value: None
exec(compile(analysis, "analysis_cell", "exec"))
assert len(df) == 16
assert df.same_generated_tokens.all()
assert len(speedups) == 12
assert len(amortization) == 18
assert (OUT / "results.png").is_file()
assert len(list(OUT.glob("*.csv"))) >= 4
print("PASS: paired runner, CSV summaries, bootstrap, amortization and figure generation.")
