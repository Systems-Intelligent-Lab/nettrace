#!/usr/bin/env python3
from __future__ import annotations

from nettrace import list_trace_sets, sample_trace, load_video_sizes


def main() -> None:
    suites = ["ABRBench-3G", "ABRBench-4G+"]
    for suite in suites:
        sets_ = list_trace_sets(suite)  # type: ignore[arg-type]
        print(f"{suite}: {len(sets_)} trace sets")

    trace = sample_trace("SolisWi-Fi", suite="ABRBench-4G+", split="train", seed=123)
    print(f"sampled trace file: {trace.path.name}")
    print(f"points: {len(trace.times)}")
    print(f"first 3 bw: {trace.bandwidths[:3]}")

    sizes = load_video_sizes("big_buck_bunny", bitrate_levels=6)
    print(f"video sizes levels: {list(sizes.keys())}")
    print(f"level 0 first 3 sizes: {sizes[0][:3]}")


if __name__ == "__main__":
    main()
