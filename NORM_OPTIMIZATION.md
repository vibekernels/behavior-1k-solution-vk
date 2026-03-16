# Norm Stats Computation Optimization

## Baseline
- **Script**: `scripts/compute_norm_stats.py`
- **Dataset**: 10,000 episodes, ~17K rows each, 128 CPUs, 512 GB RAM
- **Baseline speed**: ~3.3 eps/sec (32 workers), ~50 min estimated for 10K episodes
- **Baseline problem**: OOM — accumulated all raw chunk data (~110 GB) in memory before aggregation

## Key Findings from Prior Session
- More workers helped (32 → 64 showed improvement)
- PyArrow column selection didn't work because `observation.state` and `action` columns contain nested arrays (object dtype)
- The original code already had: vectorized numpy (no iterrows), column selection, sampled chunks (10%)

## Root Causes of OOM
1. **Worker results too large**: Each worker returned a (960, 960) float64 outer product matrix (~7.4 MB) for streaming correlation. With 10K episodes, IPC serialization alone was ~74 GB.
2. **Fork inherited heavy imports**: Workers forked from main process inherited JAX/torch/omnigibson memory. With 64 workers, this multiplied the ~4 GB import footprint.
3. **Reservoir as Python list**: Growing list of 1M numpy arrays had significant Python object overhead.

## Optimizations Applied

### Round 1: Streaming Aggregation (eliminated ~110 GB accumulation)
- Workers return compact stats (count/sum/sum_sq/min/max) instead of raw chunk arrays
- Main process accumulates incrementally — O(1) memory regardless of episode count
- Reservoir sampling for quantile computation instead of collecting all data
- **Result**: 100 eps in 4.6s (21.8 eps/sec), up from 30s. But still OOMed at 10K.

### Round 2: Memory-Safe Design (eliminated remaining OOM sources)
- **Removed outer product from workers**: Correlation computed from reservoir at the end, not streamed. Worker results dropped from ~8.5 MB to ~100 KB each.
- **Inlined `extract_state_from_proprio`**: Workers are pure numpy/pandas — no JAX/torch/omnigibson imports. PROPRIOCEPTION_INDICES extracted as plain dict of ints in main, passed to workers.
- **`forkserver` start method**: Workers fork from a clean process, not the bloated main process with heavy imports.
- **Pre-allocated float32 reservoir**: Single numpy array (500K × 30 × 23 × 4B = 1.38 GB max), no Python list growth.
- **Result**: 10K episodes in 161s (62.1 eps/sec), peak memory ~3 GB. No OOM.

## Results

| Optimization | Episodes/sec | Total time (10K) | Peak memory | Kept? |
|---|---|---|---|---|
| Baseline (original, 32 workers) | ~3.3 | ~50 min (est.) | OOM | - |
| Round 1: streaming stats (32 workers) | 21.8 | OOM at scale | ~110 GB+ | Replaced |
| Round 2: memory-safe (64 workers) | **62.1** | **2.7 min** | **~3 GB** | Yes |

## Architecture (Current)

```
Main process (heavy imports: JAX, torch, omnigibson, openpi)
  │
  ├── Extract config values, delta_mask, proprio_slices as plain Python types
  │
  └── forkserver ProcessPoolExecutor (64 workers)
        │
        Workers (only import: numpy, pandas)
          ├── Read parquet (2 columns)
          ├── Inlined state extraction (numpy slicing)
          ├── Vectorized delta transform
          ├── Compute compact stats (count/sum/sum_sq/min/max)
          ├── Sample reservoir chunks (200 per episode, float32)
          └── Return ~100 KB result dict
        │
  Main loop: accumulate stats + reservoir sampling (500K cap)
  │
  Post-processing:
    ├── Quantiles from reservoir (np.percentile)
    ├── Correlation from reservoir (X.T @ X on 500K×960 matrix)
    ├── Cholesky decomposition
    └── Save norm_stats.json
```

## Possible Further Optimizations (not yet needed)
- Increase workers to 96-128 (currently 64, have 128 CPUs)
- Use `pyarrow.compute` for faster parquet reads if nested array issue is resolved
- Batch reservoir insertion instead of per-chunk loop
- Overlap correlation computation with episode processing using a background thread
