# nettrace

**[English]** | [中文文档 README_zh.md](README_zh.md)

A **batteries-included, zero-dependency** network bandwidth trace loader designed for ABR (Adaptive Bitrate) streaming research and network simulation.

**Key features:**
- **Bundled ABRBench datasets** — 12 trace collections (3G / 4G+ / Wi-Fi) and 2 video chunk-size datasets, all shipped inside the package
- **Reproducible sampling** — file list sorted by name + `random.Random(seed)`, so the same seed always returns the same trace
- **Pure standard library** — no third-party runtime dependencies; just `pip install`

---

## Installation

**Install directly from GitHub (recommended):**

```bash
pip install git+https://github.com/Systems-Intelligent-Lab/nettrace.git
```

**Clone and install locally (for development):**

```bash
git clone https://github.com/Systems-Intelligent-Lab/nettrace.git
cd nettrace
pip install -e .          # runtime only
pip install -e ".[dev]"   # with test dependencies
```

> **Note:** If you switch between Python environments (e.g. conda envs), run the install command inside each environment and use that environment's `python` to run scripts.

---

## Quick Start

```python
from nettrace import list_trace_sets, list_trace_files, load_trace_file, sample_trace, load_video_sizes

# List all available trace collections
print(list_trace_sets("ABRBench-4G+"))
# ['Ghent', 'Lab', 'Lumos4G', 'Lumos5G', 'Norway3G', 'SolisWi-Fi']

# Sample one trace reproducibly
trace = sample_trace("SolisWi-Fi", suite="ABRBench-4G+", split="train", seed=42)
print(trace.path.name)       # wifi_lab_231114-162511.txt  (fixed for seed=42)
print(len(trace.times))      # 200
print(trace.bandwidths[:3])  # (30.6, 13.4, 13.4)

# Load video chunk sizes for each bitrate level
sizes = load_video_sizes("big_buck_bunny", bitrate_levels=6)
print([round(sum(sizes[i]) / len(sizes[i]) / 1024) for i in range(6)])
# [483, 1210, 2412, 3867, 7734, 19272]  KB — higher bitrate → larger chunks
```

Run the bundled example:

```bash
python examples/quickstart.py
```

---

## Datasets

### Network Traces

| Suite | Collection | Split | Files |
|---|---|---|---|
| ABRBench-3G | FCC-16 | train / test | 269 |
| ABRBench-3G | FCC-18 | train / test | 400 |
| ABRBench-3G | HSR | all | 34 |
| ABRBench-3G | Oboe | train / test | 428 |
| ABRBench-3G | Puffer-21 | train / test | 600 |
| ABRBench-3G | Puffer-22 | train / test | 600 |
| ABRBench-4G+ | Ghent | all | 40 |
| ABRBench-4G+ | Lab | all | 61 |
| ABRBench-4G+ | Lumos4G | train / test | 175 |
| ABRBench-4G+ | Lumos5G | train / test | 121 |
| ABRBench-4G+ | Norway3G | train / test | 134 |
| ABRBench-4G+ | SolisWi-Fi | train / test | 80 |

> HSR, Ghent, and Lab have no train/test split — use `split="all"` for these.

Each trace file contains two whitespace-separated columns:

```
0.0  25.5
1.0   7.71
2.0   7.97
...
```

Column 1: timestamp (seconds) · Column 2: bandwidth (Mbps)

### Video Chunk Metadata

| Video | Bitrate levels | Chunks per level |
|---|---|---|
| `big_buck_bunny` | 6 | 49 |
| `envivio_3g` | 6 | 49 |

Bitrate ladder: 300 / 750 / 1200 / 1850 / 2850 / 4300 Kbps

---

## API Reference

### `list_trace_sets(suite) -> list[str]`

Return a sorted list of all collection names under the given suite.

```python
list_trace_sets("ABRBench-3G")   # ['FCC-16', 'FCC-18', 'HSR', ...]
list_trace_sets("ABRBench-4G+")  # ['Ghent', 'Lab', 'Lumos4G', ...]
```

### `list_trace_files(trace_set, *, suite, split) -> list[Path]`

Return all trace file paths in a collection/split, **sorted by filename** for a stable, deterministic order.

```python
files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
```

### `load_trace_file(path) -> Trace`

Parse a single two-column trace file and return a `Trace` object.

### `sample_trace(trace_set, *, suite, split, seed) -> Trace`

Reproducibly sample and load one trace from a collection.

```python
trace = sample_trace("FCC-16", suite="ABRBench-3G", split="test", seed=0)
```

### `load_video_sizes(video_type, *, bitrate_levels) -> dict[int, tuple[int, ...]]`

Load per-chunk byte sizes for every bitrate level.

```python
sizes = load_video_sizes("envivio_3g", bitrate_levels=6)
sizes[0]   # chunk sizes (bytes) for level 0
```

### `Trace` (frozen dataclass)

| Field | Type | Description |
|---|---|---|
| `times` | `tuple[float, ...]` | Timestamp sequence (seconds) |
| `bandwidths` | `tuple[float, ...]` | Bandwidth sequence (Mbps) |
| `path` | `pathlib.Path` | Absolute path to the source file |

---

## Reproducibility

`sample_trace(..., seed=SEED)` guarantees reproducibility via two invariants:

1. `list_trace_files` always returns files **sorted alphabetically by filename** — independent of filesystem ordering
2. A `random.Random(seed)` instance is used to pick the index from that sorted list

As long as the dataset content is unchanged, **the same seed always selects the same file and produces the same trace**.

```python
t1 = sample_trace("SolisWi-Fi", suite="ABRBench-4G+", split="train", seed=42)
t2 = sample_trace("SolisWi-Fi", suite="ABRBench-4G+", split="train", seed=42)
assert t1.path == t2.path            # same file
assert t1.bandwidths == t2.bandwidths  # same content
```

---

## Example: Using nettrace in a Gym Environment

`examples/env.py` provides a complete ABR video-streaming Gym environment that uses `nettrace` as its data backend:

```python
from nettrace import list_trace_files, load_trace_file, load_video_sizes

# Load all traces for an episode pool
files = list_trace_files(trace_name, suite=suite, split=data_split)
for f in files:
    trace = load_trace_file(f)
    time_seqs.append(list(trace.times))
    bw_seqs.append(list(trace.bandwidths))

# Load video chunk sizes
sizes = load_video_sizes(video_type, bitrate_levels=6)
```

Run the debug demo (requires `gymnasium` and `numpy` in your environment):

```bash
python examples/env.py
```

Sample output:

```
[reset  ]  delay=  227.1ms  rebuf= 227.1ms  buf=  4.0s  remain=47  chunk=  134KB
[step  1]  delay= 2233.8ms  rebuf=   0.0ms  buf=  5.8s  remain=46  chunk= 1822KB  reward=+0.7500
[step  2]  delay= 3239.1ms  rebuf=   0.0ms  buf=  6.5s  remain=45  chunk= 2748KB  reward=+1.2000
...
Episode ended — total reward: 55.95  |  rebuffer: 0.000 s
✓ Reproducibility check passed
```

---

## Tests

```bash
pytest
```

Test coverage (27 tests):

| Area | What is verified |
|---|---|
| `list_trace_sets` | Both suites return expected collections; result is sorted |
| `list_trace_files` | Returns `Path` objects; sorted by name; stable across calls; test split works; all-only sets use root dir |
| `load_trace_file` | Two-column parse; timestamps ≥ 0; bandwidths > 0; `path` stored correctly; every file in FCC-16/train loads without error |
| Bulk loading | All files in a set load into parallel lists with equal length |
| `load_video_sizes` | All 6 levels present; all sizes positive; chunk count consistent across levels; higher bitrate → larger average chunk |
| Reproducibility | Same seed → same file path and content; different seeds → different files; `reset(seed=X)` resets to identical first observation |

---

## Project Structure

```
nettrace/
├── nettrace/
│   ├── __init__.py       # Public API exports
│   ├── core.py           # Trace, list_trace_sets, load_trace_file, sample_trace, load_video_sizes
│   ├── utils.py          # DATA_ROOT constant
│   └── datasets/
│       ├── trace/
│       │   ├── ABRBench-3G/   (FCC-16, FCC-18, HSR, Oboe, Puffer-21, Puffer-22)
│       │   └── ABRBench-4G+/  (Ghent, Lab, Lumos4G, Lumos5G, Norway3G, SolisWi-Fi)
│       └── video/
│           ├── big_buck_bunny/
│           └── envivio_3g/
├── examples/
│   ├── quickstart.py     # Minimal usage demo
│   └── env.py            # ABR Gym environment demo (requires gymnasium + numpy)
├── tests/
│   ├── test_reproducible_sampling.py
│   └── test_env_dependencies.py
├── pyproject.toml
└── MANIFEST.in
```
