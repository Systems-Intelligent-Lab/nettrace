from __future__ import annotations

import random

from nettrace import list_trace_files, load_trace_file, sample_trace


def test_same_seed_same_trace_path_and_content() -> None:
    t1 = sample_trace("SolisWi-Fi", suite="ABRBench-4G+", split="train", seed=42)
    t2 = sample_trace("SolisWi-Fi", suite="ABRBench-4G+", split="train", seed=42)

    print(f"\n  seed=42 选中文件: {t1.path.name}")
    print(f"  两次路径相同: {t1.path == t2.path}")
    print(f"  前5个时间戳: {t1.times[:5]}")
    print(f"  前5个带宽值: {t1.bandwidths[:5]}")

    assert t1.path == t2.path
    assert t1.times[:10] == t2.times[:10]
    assert t1.bandwidths[:10] == t2.bandwidths[:10]


def test_sampling_matches_sorted_list_and_rng() -> None:
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    assert files, "expected bundled dataset files"

    seed = 123
    rng = random.Random(seed)
    expected = files[rng.randrange(len(files))]
    got = sample_trace("SolisWi-Fi", suite="ABRBench-4G+", split="train", seed=seed)

    print(f"\n  候选文件共 {len(files)} 个 (已排序)")
    print(f"  seed={seed} 手动计算应选: {expected.name}")
    print(f"  sample_trace 实际选中:    {got.path.name}")
    print(f"  结果一致: {got.path.name == expected.name}")

    assert got.path.name == expected.name


def test_load_trace_file_parses_two_columns() -> None:
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    tr = load_trace_file(files[0])

    print(f"\n  文件: {files[0].name}")
    print(f"  数据点数: {len(tr.times)}")
    print(f"  时间列[:3]: {tr.times[:3]}")
    print(f"  带宽列[:3]: {tr.bandwidths[:3]}")
    print(f"  时间列与带宽列长度相同: {len(tr.times) == len(tr.bandwidths)}")

    assert len(tr.times) == len(tr.bandwidths)
    assert len(tr.times) > 0
