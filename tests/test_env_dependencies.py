"""Verify that all data-loading capabilities required by VideoStreamingEnv
work correctly through the nettrace package API.

These tests do NOT import the env itself; they test only what the package
provides so that any external env can rely on these guarantees.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nettrace import (
    list_trace_sets,
    list_trace_files,
    load_trace_file,
    load_video_sizes,
)
from nettrace.utils import DATA_ROOT

_ALL_ONLY = {"HSR", "Ghent", "Lab"}
BITRATE_LEVELS = 6
VALID_VIDEOS = ("big_buck_bunny", "envivio_3g")


# ── list_trace_sets ──────────────────────────────────────────────────────────

def test_list_trace_sets_3g():
    sets = list_trace_sets("ABRBench-3G")
    print(f"\n  ABRBench-3G trace sets ({len(sets)}个): {sets}")
    assert set(sets) >= {"FCC-16", "FCC-18", "HSR", "Oboe", "Puffer-21", "Puffer-22"}


def test_list_trace_sets_4g():
    sets = list_trace_sets("ABRBench-4G+")
    print(f"\n  ABRBench-4G+ trace sets ({len(sets)}个): {sets}")
    assert set(sets) >= {"Ghent", "Lab", "Lumos4G", "Lumos5G", "Norway3G", "SolisWi-Fi"}


def test_list_trace_sets_returns_sorted():
    for suite in ("ABRBench-3G", "ABRBench-4G+"):
        sets = list_trace_sets(suite)  # type: ignore[arg-type]
        print(f"\n  {suite} 顺序: {sets}")
        assert sets == sorted(sets), f"{suite} list is not sorted"


# ── list_trace_files ─────────────────────────────────────────────────────────

def test_list_trace_files_returns_paths():
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    print(f"\n  SolisWi-Fi/train 共 {len(files)} 个文件")
    print(f"  前3个: {[p.name for p in files[:3]]}")
    assert len(files) > 0
    assert all(isinstance(p, Path) for p in files)


def test_list_trace_files_sorted_by_name():
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    names = [p.name for p in files]
    print(f"\n  文件名前3: {names[:3]}")
    print(f"  文件名后3: {names[-3:]}")
    print(f"  已按字母排序: {names == sorted(names)}")
    assert names == sorted(names)


def test_list_trace_files_stable_across_calls():
    a = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    b = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    same = [p.name for p in a] == [p.name for p in b]
    print(f"\n  两次调用结果完全一致: {same}  (共{len(a)}个文件)")
    assert same


def test_list_trace_files_test_split():
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="test")
    print(f"\n  SolisWi-Fi/test 共 {len(files)} 个文件")
    assert len(files) > 0


@pytest.mark.parametrize("trace_set", ["HSR", "Ghent", "Lab"])
def test_all_only_sets_use_root_dir(trace_set: str):
    suite = "ABRBench-3G" if trace_set == "HSR" else "ABRBench-4G+"
    root = DATA_ROOT / "trace" / suite / trace_set
    direct_files = sorted(
        [p for p in root.iterdir() if p.is_file() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )
    print(f"\n  {trace_set} 根目录直接包含 {len(direct_files)} 个 trace 文件")
    print(f"  前2个: {[p.name for p in direct_files[:2]]}")
    assert len(direct_files) > 0


# ── load_trace_file ──────────────────────────────────────────────────────────

def test_load_trace_file_two_columns():
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    trace = load_trace_file(files[0])
    print(f"\n  文件: {files[0].name}")
    print(f"  数据点数: {len(trace.times)}")
    print(f"  时间列与带宽列长度相同: {len(trace.times) == len(trace.bandwidths)}")
    assert len(trace.times) == len(trace.bandwidths)
    assert len(trace.times) > 0


def test_load_trace_file_times_non_negative():
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    trace = load_trace_file(files[0])
    min_t = min(trace.times)
    print(f"\n  文件: {files[0].name}")
    print(f"  时间戳范围: [{min_t:.3f}, {max(trace.times):.3f}] 秒")
    assert min_t >= 0


def test_load_trace_file_bandwidths_positive():
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    trace = load_trace_file(files[0])
    min_bw, max_bw = min(trace.bandwidths), max(trace.bandwidths)
    avg_bw = sum(trace.bandwidths) / len(trace.bandwidths)
    print(f"\n  文件: {files[0].name}")
    print(f"  带宽范围: [{min_bw:.3f}, {max_bw:.3f}]  均值: {avg_bw:.3f}")
    assert min_bw > 0


def test_load_trace_file_path_stored():
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    trace = load_trace_file(files[0])
    print(f"\n  输入路径: {files[0]}")
    print(f"  trace.path: {trace.path}")
    print(f"  两者相同: {trace.path == files[0].resolve()}")
    assert trace.path == files[0].resolve()


def test_load_all_traces_in_set():
    files = list_trace_files("FCC-16", suite="ABRBench-3G", split="train")
    total_pts = 0
    for f in files:
        tr = load_trace_file(f)
        total_pts += len(tr.times)
        assert len(tr.times) > 0
    print(f"\n  FCC-16/train: {len(files)} 个文件全部解析成功，合计 {total_pts} 个数据点")


# ── env needs: load all files as parallel lists ──────────────────────────────

def test_bulk_load_returns_parallel_lists():
    """模拟 VideoStreamingEnv 批量加载所有 trace 为 time_sequences / bandwidth_sequences。"""
    files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
    time_seqs, bw_seqs = [], []
    for f in files:
        tr = load_trace_file(f)
        time_seqs.append(tr.times)
        bw_seqs.append(tr.bandwidths)

    lens = [len(t) for t in time_seqs]
    print(f"\n  加载 {len(time_seqs)} 条 trace")
    print(f"  各 trace 数据点数: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.1f}")
    print(f"  time_seqs 与 bw_seqs 数量一致: {len(time_seqs) == len(bw_seqs)}")
    assert len(time_seqs) == len(bw_seqs)
    for t, b in zip(time_seqs, bw_seqs):
        assert len(t) == len(b) and len(t) > 0


def test_bulk_load_order_is_deterministic():
    files = list_trace_files("Lumos4G", suite="ABRBench-4G+", split="train")
    tr_a = load_trace_file(files[0])
    tr_b = load_trace_file(files[0])
    print(f"\n  文件: {files[0].name}")
    print(f"  两次加载带宽前3值相同: {tr_a.bandwidths[:3] == tr_b.bandwidths[:3]}")
    print(f"  带宽前3: {tr_a.bandwidths[:3]}")
    assert tr_a.bandwidths == tr_b.bandwidths


# ── load_video_sizes ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("video_type", VALID_VIDEOS)
def test_load_video_sizes_all_levels(video_type: str):
    sizes = load_video_sizes(video_type, bitrate_levels=BITRATE_LEVELS)
    print(f"\n  [{video_type}] 共 {len(sizes)} 个码率层: {sorted(sizes.keys())}")
    assert set(sizes.keys()) == set(range(BITRATE_LEVELS))


@pytest.mark.parametrize("video_type", VALID_VIDEOS)
def test_load_video_sizes_positive(video_type: str):
    sizes = load_video_sizes(video_type, bitrate_levels=BITRATE_LEVELS)
    for level, chunks in sizes.items():
        avg = sum(chunks) / len(chunks)
        print(f"\n  [{video_type}] level={level}: {len(chunks)} chunks, 均值大小={avg/1024:.1f} KB")
        assert len(chunks) > 0
        assert all(s > 0 for s in chunks)


@pytest.mark.parametrize("video_type", VALID_VIDEOS)
def test_load_video_sizes_consistent_chunk_count(video_type: str):
    sizes = load_video_sizes(video_type, bitrate_levels=BITRATE_LEVELS)
    counts = [len(v) for v in sizes.values()]
    print(f"\n  [{video_type}] 所有码率层 chunk 数: {counts}")
    assert len(set(counts)) == 1, f"Inconsistent chunk counts: {counts}"


def test_load_video_sizes_higher_bitrate_larger():
    sizes = load_video_sizes("big_buck_bunny", bitrate_levels=BITRATE_LEVELS)
    avgs = [sum(sizes[lvl]) / len(sizes[lvl]) for lvl in range(BITRATE_LEVELS)]
    print("\n  big_buck_bunny 各码率层平均 chunk 大小 (KB):")
    for lvl, avg in enumerate(avgs):
        print(f"    level {lvl}: {avg/1024:.1f} KB")
    assert avgs[-1] > avgs[0], "最高码率层应有更大的平均 chunk 大小"
