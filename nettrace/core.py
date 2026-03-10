from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Literal

from .utils import DATA_ROOT

TraceSuite = Literal["ABRBench-3G", "ABRBench-4G+"]
TraceSplit = Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class Trace:
    """A single network trace.

    Attributes:
        times: Time stamps (seconds).
        bandwidths: Bandwidth values (unit depends on dataset; typically Mbps).
        path: Absolute path to the underlying trace file.
    """

    times: tuple[float, ...]
    bandwidths: tuple[float, ...]
    path: Path


def _trace_root(suite: TraceSuite) -> Path:
    return DATA_ROOT / "trace" / suite


def list_trace_sets(suite: TraceSuite = "ABRBench-4G+") -> list[str]:
    """List available trace set names under a suite."""
    root = _trace_root(suite)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_trace_files(
    trace_set: str,
    *,
    suite: TraceSuite = "ABRBench-4G+",
    split: TraceSplit = "train",
) -> list[Path]:
    """List trace files (sorted) for stable reproducible sampling."""
    trace_dir = _trace_root(suite) / trace_set / split
    if not trace_dir.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")
    return sorted([p for p in trace_dir.iterdir() if p.is_file()], key=lambda p: p.name)


def load_trace_file(path: Path) -> Trace:
    """Load a trace file that contains 2 columns: time and bandwidth."""
    times: list[float] = []
    bandwidths: list[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            times.append(float(parts[0]))
            bandwidths.append(float(parts[1]))

    if not times or not bandwidths:
        raise ValueError(f"Empty or invalid trace file: {path}")
    return Trace(times=tuple(times), bandwidths=tuple(bandwidths), path=path.resolve())


def sample_trace(
    trace_set: str,
    *,
    suite: TraceSuite = "ABRBench-4G+",
    split: TraceSplit = "train",
    seed: int | None = None,
) -> Trace:
    """Sample (deterministically with seed) and load one trace.

    Reproducibility guarantee:
    - The candidate file list is sorted by filename.
    - Sampling uses `random.Random(seed)` so the same seed yields the same file.
    """
    files = list_trace_files(trace_set, suite=suite, split=split)
    if not files:
        raise FileNotFoundError(f"No trace files found in {suite}/{trace_set}/{split}")

    rng = random.Random(seed)
    chosen = files[rng.randrange(len(files))]
    return load_trace_file(chosen)


def load_video_sizes(
    video_type: str,
    *,
    bitrate_levels: int = 6,
) -> dict[int, tuple[int, ...]]:
    """Load video chunk sizes for each bitrate level.

    Expects files named `video_size_0..video_size_{bitrate_levels-1}` under
    `nettrace/datasets/video/<video_type>/`.
    """
    video_dir = DATA_ROOT / "video" / video_type
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    out: dict[int, tuple[int, ...]] = {}
    for level in range(bitrate_levels):
        path = video_dir / f"video_size_{level}"
        sizes: list[int] = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                sizes.append(int(line.split()[0]))
        out[level] = tuple(sizes)
    return out