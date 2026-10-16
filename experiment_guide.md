# AI inference and KV-cache reuse on a Colab T4

Use `kv_cache_t4_colab.ipynb` as the executable experiment. The companion
`kv_cache_benchmark.py` contains exactly the same code and cell explanations.
Recorded T4 measurements are included in `results/20261011-174734/`.
The notebook produces a new measurement set on your runtime.

Authoring validation: every notebook code cell was syntax-checked. The core
generation functions passed 12 mode/shape checks against Hugging Face `generate`
using a tiny randomly initialized Qwen2 model on CPU, including batched requests,
zero-prefix reuse, cache byte counts, and master-cache immutability. The actual
summary/bootstrap/amortization code passed synthetic-fixture checks. The included results came from a completed Colab T4 session. Repeating the
pretrained-model CUDA execution on your own runtime is required to obtain new
GPU measurements; CPU checks do not establish GPU performance.

## Research questions

Separate two mechanisms before drawing conclusions:

1. **Within-request KV caching:** retain the keys and values of tokens already
   processed while generating a single answer. Compare `no_cache` with
   `fresh_cache`. Both process the complete prompt initially; only the former
   recomputes the entire growing sequence at each generation step.
2. **Across-request prefix reuse:** retain a common prompt prefix between requests.
   Compare `fresh_cache` with `prefix_reuse`. Both use normal caching during
   decoding; the latter avoids recomputing the shared prefix during prefill.

A useful concrete workload is a fixed document followed by different questions.
Reuse requires identical prefix **tokens**, model configuration/weights, positions,
masks, and adapters. Identical text found after different preceding context is not
generally interchangeable. Tokenization boundaries also matter.

Hugging Face documents cache handling and positions in its
[caching explanation](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
and [cache strategies guide](https://huggingface.co/docs/transformers/v4.57.1/en/kv_cache).

## Steps to obtain results

1. Download the notebook, open [Google Colab](https://colab.research.google.com/),
   and upload it using **File > Upload notebook**.
2. Select **Runtime > Change runtime type > GPU**. Select a T4 if available.
   Confirm the actual GPU printed in cell 2. Colab hardware and availability vary;
   see its [resource FAQ](https://research.google.com/colaboratory/faq.html).
3. Run the installation cell in a fresh runtime. It pins Transformers to 4.57.1
   and Accelerate to 1.11.0, retaining Colab's CUDA-enabled PyTorch. If Transformers
   was already imported, restart the session, then resume from cell 2.
4. Start with `PROFILE = "quick"`, the default Qwen2.5-0.5B-Instruct, and FP16.
   The notebook prints the GPU, package versions, model revision, and configuration.
   Downloading/loading the model is outside all inference measurements.
5. Run the functions and correctness gate. Check the displayed logit differences
   and generated-token agreement. Significant logit differences stop the notebook.
   Near-tied greedy choices can diverge through floating-point rounding; investigate
   mismatches before using that configuration for a claim.
6. Run the sweeps. Quick uses 3 trials and verifies that the pipeline works.
   Each condition has 2 unmeasured warmups. The notebook randomizes mode and case
   order and uses paired inputs, protecting against some order/time effects.
7. Inspect the tables and three plots. Make sure the 0% prefix negative control
   shows similar times for fresh and reused modes. Check OOMs and incomplete cases.
8. Download the generated zip before the runtime disconnects. It preserves raw
   trials, environment details, model config, package versions, correctness checks,
   prefix creation costs, failures, summaries, paired speedups, and plots.
9. Change to `PROFILE = "research"` and rerun the experiment. This expands lengths,
   prefix ratios, output lengths, and batch sizes with 10 trials per condition.
   For your final selected configurations, increase `REPEATS` to 20–30 and rerun
   in at least 3 independent sessions. Report within- and between-session variation.
10. Only after this works, repeat selected settings with 4096-token prompts or
    Qwen2.5-1.5B-Instruct. Use a fresh session for model/precision changes and retain
    the model revision. Reduce batch size if needed. Record OOM outcomes instead
    of silently changing precision, moving to CPU, or changing the model.

This is a latency study using controlled text-derived token sequences. It does
not evaluate answer usefulness or task accuracy. Fixed output lengths deliberately
ignore EOS, eliminating unequal generation length as a confound.

## Manipulated variables and hypotheses

| Variable | Research settings | Hold fixed | Hypothesis |
|---|---|---|---|
| Within-request cache | Off / fresh dynamic cache | Prompt IDs, output length, batch, model | Cache improves decode latency; benefit usually grows with context |
| Prompt tokens L | 128, 512, 2048 in A; optionally 4096 | Output length, batch, mode | More context increases work and KV storage |
| Cached prefix P/L | 0%, 25%, 50%, 75%, 90% in B | **Total** prompt length = 2048 | Larger reusable prefixes generally reduce warm TTFT, subject to copy overhead |
| Output tokens G | 16 and 64 in A; 16 and 128 in B | Prompt/prefix/batch | Long answers dilute the end-to-end percentage improvement from prefix reuse |
| Batch size B | 1, 2, 4 in C | L=1024, P=768, G=32 | Aggregate throughput may rise, while latency and KV memory also rise |
| Number of reuses N | 1, 2, 4, 8, 16, 32 projected | Measured build and request times | More hits amortize one-time cache creation |

The notebook tests one set of factors at a time rather than running every possible
combination. Research settings are in cell 5; modify those lists to expand a sweep.
`N` is a projection from measured costs, not an additional executed workload.

Optional follow-up: vary request-level hit rate at 0%, 25%, 50%, 75%, 100% using an
explicit sequence of requests and a cache lookup policy. This is different from
prefix fraction: a 90%-cached prompt can still occur in a workload with only 10%
cache hits. The supplied notebook isolates warm hits; it does not simulate eviction,
cache lookup, CPU/disk offload, or production concurrency.

## Metrics and formulas

- **TTFT:** synchronized elapsed time from GPU-resident inputs to the first
  selected token on the GPU. Includes prefix-copy time for reuse. Excludes
  tokenization, request queueing, network transfer, and CPU text decoding.
- **Mean TPOT:** decode time divided by G−1. The first token is produced by prefill.
- **Decode throughput:** B(G−1)/decode_seconds, aggregated across the batch.
- **End-to-end throughput:** BG/total_seconds, for this model-execution boundary.
- **Total latency:** TTFT plus decode time.
- **Speedup:** control latency / treatment latency. Above 1 favors treatment.
  The exported estimate is the median of paired trial ratios.
- **KV payload:** bytes stored in the request's actual key/value tensors.
- **Peak allocated memory:** peak live PyTorch tensor allocation. Also report
  peak reserved memory separately; reserved memory reflects allocator history.

CUDA execution is asynchronous, so the code synchronizes timing boundaries.
It does not synchronize after each output token. These host wall timings include
Python and dispatch overhead and are not isolated kernel timings. See
[PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

For a full-attention model with uniform layers:

    KV bytes = 2 × layers × batch × cached_tokens × KV_heads × head_dim × bytes_per_element

The factor 2 accounts for keys and values. Use KV heads, not query heads, for
grouped-query attention. For this Qwen checkpoint, its
[configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)
gives 24 layers, 2 KV heads, and head_dim = 896/14 = 64. At FP16 this is
12,288 bytes per cached token per sequence. A 4096-token KV cache therefore has
48 MiB of tensor payload, excluding weights, allocator overhead, and temporary
tensors. This small model's cache is intentionally modest; do not extrapolate its
absolute memory consumption to a larger model.

The final request cache has L+G−1 tokens because the last emitted token has not
been fed back through the model. The code checks measured payload against that
formula. Peak memory without caching need not always be lower: repeated full
sequence forwards can have large temporary allocations.

## What different outcomes would mean

**Lower TTFT with similar TPOT:** the expected prefix-reuse signature. Prefill
work was avoided, but decoding still attends to the complete history. vLLM's
[prefix caching documentation](https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/)
describes the same distinction between prefill and decoding.

**Much faster decoding for fresh cache versus no cache:** evidence for ordinary
autoregressive KV caching, not evidence of reuse across requests.

**Little benefit or a slowdown for short prefixes:** a plausible result if cache
copying and Python overhead exceed the saved prefill computation. Inspect `copy_ms`
and run longer contexts before drawing a general conclusion.

**Large TTFT improvement but modest total-latency improvement:** prefill is only
part of the request. Longer answers spend more time in decoding.

**More batch throughput but higher latency:** useful hardware utilization improved,
but individual users do not necessarily receive faster answers. Report both.

**Memory grows linearly in batch/context:** expected for this dense full-attention
KV payload. Total peak memory can behave differently because temporary allocations
and attention backends also change.

**Different generated outputs:** first confirm prefix token equality, mask length,
positions, and cache immutability. Then examine numerical differences near greedy
ties. Reusing an exact full-precision prefix should not intentionally change the
model's computation; approximate/compressed caches are a different experiment.

## Cache-build amortization

Let C be one-time prefix creation time, F be fresh request time, H be warm reused
request time, and N be the number of subsequent requests using the prefix:

    without reuse = N × F
    with reuse    = C + N × H
    break-even    = N > C / (F − H), provided F > H

For illustration only, if C=100 ms, F=80 ms, H=50 ms, then reuse wins after 4
requests. These numbers are invented to explain the formula, not benchmark results.
If F≤H, no number of reuses pays back this implementation's build cost.

The notebook measures standalone prefix creation. A serving engine may build its
cache while processing the first real request, so its cold-start accounting differs.
For B>1, N refers to sequential batches in the exported projection.

## Limits to state in a report

The reusable GPU master is copied before each hit to prevent its mutation. Copy
time is included, and the master remains resident in both B/C comparison modes.
Thus raw peak allocation in those comparisons includes a constant cache pool in
both modes. The implementation does **not** share physical prefix blocks between
requests; it cannot demonstrate the memory savings of a paged serving system.

The benchmark uses eager Transformers execution and a small model on one GPU.
Do not describe the results as the maximum T4 serving throughput. Tiny-model
latency can be heavily influenced by host/Python overhead. Bootstrap intervals
describe trials within a session and are not guarantees across hardware/sessions.
Three quick trials are not enough for final claims, and neither quick nor research
defaults justify tail-latency claims such as p99.

A useful reporting sentence is:

> On [recorded GPU] using [model revision, dtype, software], for L=[...], G=[...],
> B=[...], and prefix fraction=[...], median paired TTFT speedup was [...] with
> within-session bootstrap interval [...]. Total latency changed from [...] to
> [...] ms. The resident prefix occupied [...] MiB, and copying cost [...] ms.
> Across [...] independent sessions, the observed speedup range was [...].
