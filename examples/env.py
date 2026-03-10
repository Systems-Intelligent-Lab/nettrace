from __future__ import annotations

import copy
import math
import random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from nettrace import list_trace_files, load_trace_file, load_video_sizes

# --------------------------------------------------------------------------- #
#  QoE coefficients
# --------------------------------------------------------------------------- #
_BETA = 6
QoE_Param_Type: dict[str, dict[str, float]] = {
    "livestreams":   {"μ1": 1,     "μ2": 1,     "μ3": _BETA},
    "documentaries": {"μ1": 1,     "μ2": _BETA, "μ3": 1},
    "news":          {"μ1": _BETA, "μ2": 1,     "μ3": 1},
    "normal":        {"μ1": 1,     "μ2": 1,     "μ3": 4.3},
}

# --------------------------------------------------------------------------- #
#  Network / playback constants
# --------------------------------------------------------------------------- #
LINK_RTT               = 80           # ms
NOISE_LOW              = 0.9
NOISE_HIGH             = 1.1
M_IN_K                 = 1000.0
BITRATE_LEVELS         = 6
BUFFER_NORM_FACTOR     = 10.0
MILLISECONDS_IN_SECOND = 1000.0
DRAIN_BUFFER_SLEEP_TIME = 500.0       # ms
BUFFER_THRESH          = 60.0 * MILLISECONDS_IN_SECOND  # ms  (60 s)
VIDEO_CHUNCK_LEN       = 4000.0       # ms per video chunk

# 3G trace sets (live under ABRBench-3G)
_3G_SETS = {"FCC-16", "FCC-18", "HSR", "Oboe", "Puffer-21", "Puffer-22"}
# Sets whose files live directly in the set root (no train/test sub-dir)
_ALL_ONLY_SETS = {"HSR", "Ghent", "Lab"}

VALID_TRACE_NAMES = {
    "FCC-16", "FCC-18", "HSR", "Oboe", "Puffer-21", "Puffer-22",
    "Ghent", "Lab", "Lumos4G", "Lumos5G", "Norway3G", "SolisWi-Fi",
}
VALID_SPLITS   = {"train", "test", "all"}
VALID_VIDEOS   = {"big_buck_bunny", "envivio_3g"}
VALID_QOE_TYPES = set(QoE_Param_Type.keys())


class VideoStreamingEnv(gym.Env):
    """Adaptive Bitrate (ABR) video streaming simulation gym environment.

    Reproducibility:
    - Pass ``seed`` to the constructor (or ``reset(seed=…)``).
    - Trace-file lists are sorted by filename before random selection, so the
      same seed always picks the same trace.

    Parameters
    ----------
    trace_name:
        One of ``VALID_TRACE_NAMES``.
    data_split:
        ``"train"``, ``"test"``, or ``"all"``.  Sets in ``_ALL_ONLY_SETS``
        require ``"all"``.
    video_type:
        ``"big_buck_bunny"`` or ``"envivio_3g"``.
    qoe_type:
        One of ``"livestreams"``, ``"documentaries"``, ``"news"``, ``"normal"``.
    seed:
        Integer seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace_name: str,
        data_split: str,
        video_type: str,
        qoe_type: str,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if trace_name not in VALID_TRACE_NAMES:
            raise ValueError(f"Invalid trace name: {trace_name!r}. Choose from {sorted(VALID_TRACE_NAMES)}")
        if data_split not in VALID_SPLITS:
            raise ValueError(f"Invalid data_split: {data_split!r}. Choose from {sorted(VALID_SPLITS)}")
        if trace_name in _ALL_ONLY_SETS and data_split != "all":
            raise ValueError(f"Trace {trace_name!r} only supports split='all', got {data_split!r}")
        if video_type not in VALID_VIDEOS:
            raise ValueError(f"Invalid video_type: {video_type!r}. Choose from {sorted(VALID_VIDEOS)}")
        if qoe_type not in VALID_QOE_TYPES:
            raise ValueError(f"Invalid qoe_type: {qoe_type!r}. Choose from {sorted(VALID_QOE_TYPES)}")

        self.video_type = video_type
        self.VIDEO_BIT_RATE = np.array([300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0])  # Kbps
        self.TOTAL_VIDEO_CHUNCK = 48
        self.SMOOTH_PENALTY = QoE_Param_Type[qoe_type]["μ2"]
        self.REBUF_PENALTY  = QoE_Param_Type[qoe_type]["μ3"]

        # Load all traces and video sizes once at construction time.
        self.time_traces, self.bandwidth_traces = self._load_bandwidth_trace(trace_name, data_split)
        self.video_chunk_sizes = self._load_video_sizes_by_bitrate()

        self.action_space = spaces.Discrete(BITRATE_LEVELS)
        self.observation_space = spaces.Dict(
            {
                "delay_ms":                      spaces.Box(low=0.0, high=1e8, shape=(), dtype=np.float32),
                "sleep_time_ms":                 spaces.Box(low=0.0, high=1e8, shape=(), dtype=np.float32),
                "buffer_size_ms":                spaces.Box(low=0.0, high=1e8, shape=(), dtype=np.float32),
                "rebuffer_ms":                   spaces.Box(low=0.0, high=1e8, shape=(), dtype=np.float32),
                "selected_video_chunk_size_bytes": spaces.Box(low=0, high=1e8, shape=(), dtype=np.int32),
                "is_done_bool":                  spaces.Discrete(2),
                "remain_chunk":                  spaces.Box(low=0, high=self.TOTAL_VIDEO_CHUNCK, shape=(), dtype=np.int32),
                "next_video_chunk_sizes":        spaces.Box(low=0, high=1e8, shape=(BITRATE_LEVELS,), dtype=np.int32),
            }
        )

        # RNG state is set during reset() / _apply_seed()
        self._np_rng: np.random.Generator = np.random.default_rng(seed)
        self._py_rng: random.Random = random.Random(seed)

        # Episode state (initialised in reset)
        self.time_stamp: float = 0.0
        self.client_buffer_size: float = 0.0
        self.video_chunk_cnt: int = 0
        self.last_select_bitrate: int = 1
        self.trace_index: int = 0
        self.current_trace_times: list[float] = []
        self.current_bandwidth: list[float] = []
        self.bandwidth_ptr: int = 1
        self.last_bandwidth_time: float = 0.0

    # ---------------------------------------------------------------------- #
    #  Gym API
    # ---------------------------------------------------------------------- #

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._apply_seed(seed)

        self.time_stamp        = 0.0
        self.client_buffer_size = 0.0
        self.video_chunk_cnt   = 0
        self.last_select_bitrate = 1

        self._pick_trace()

        obs = self._get_video_chunk(self.last_select_bitrate)
        assert obs["remain_chunk"] < self.TOTAL_VIDEO_CHUNCK, "Video chunk count error on reset!"
        return copy.deepcopy(obs), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        bitrate = int(action)
        obs = self._get_video_chunk(bitrate)

        bw_reward     = self.VIDEO_BIT_RATE[bitrate] / M_IN_K
        rebuf_penalty = self.REBUF_PENALTY * obs["rebuffer_ms"] / MILLISECONDS_IN_SECOND
        smooth_penalty = self.SMOOTH_PENALTY * (
            np.abs(self.VIDEO_BIT_RATE[bitrate] - self.VIDEO_BIT_RATE[self.last_select_bitrate]) / M_IN_K
        )
        reward = float(bw_reward - rebuf_penalty - smooth_penalty)

        self.last_select_bitrate = bitrate
        terminated = bool(obs["is_done_bool"])

        return (
            copy.deepcopy(obs),
            reward,
            terminated,
            False,
            {
                "bitrate_reward":         float(bw_reward),
                "rebuffer_time_reward":   float(-rebuf_penalty),
                "smooth_penalty_reward":  float(-smooth_penalty),
            },
        )

    # ---------------------------------------------------------------------- #
    #  Internal helpers
    # ---------------------------------------------------------------------- #

    def _apply_seed(self, seed: int) -> None:
        self._np_rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def _pick_trace(self) -> None:
        """Select a trace at random (using current RNG state)."""
        n = len(self.time_traces)
        self.trace_index = int(self._np_rng.integers(0, n))
        self.current_trace_times = self.time_traces[self.trace_index]
        self.current_bandwidth   = self.bandwidth_traces[self.trace_index]
        max_ptr = len(self.current_bandwidth) - 1
        self.bandwidth_ptr = int(self._np_rng.integers(1, max(2, max_ptr)))
        self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr - 1]

    def _get_video_chunk(self, quality: int) -> dict:
        assert 0 <= quality < BITRATE_LEVELS
        selected_chunk_size = self.video_chunk_sizes[quality][self.video_chunk_cnt]

        BITS_IN_BYTE           = 8.0
        B_IN_MB                = 1_000_000.0
        PACKET_PAYLOAD_PORTION = 0.95

        # ---- Simulate chunk download ---- #
        delay: float = 0.0
        downloaded  = 0.0
        while True:
            throughput = self.current_bandwidth[self.bandwidth_ptr] * B_IN_MB / BITS_IN_BYTE
            duration   = self.current_trace_times[self.bandwidth_ptr] - self.last_bandwidth_time
            assert duration >= 0, f"Negative duration: {duration}"

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION
            if downloaded + packet_payload > selected_chunk_size:
                frac_time = (selected_chunk_size - downloaded) / throughput / PACKET_PAYLOAD_PORTION
                delay += frac_time
                self.last_bandwidth_time += frac_time
                break

            downloaded += packet_payload
            delay      += duration
            self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr]
            self.bandwidth_ptr += 1
            if self.bandwidth_ptr >= len(self.current_bandwidth):
                self.bandwidth_ptr = 1
                self.last_bandwidth_time = 0.0

        # ---- Add RTT + multiplicative noise ---- #
        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT
        delay *= self._np_rng.uniform(NOISE_LOW, NOISE_HIGH)

        # ---- Buffer accounting ---- #
        wait_rebuf_time         = max(delay - self.client_buffer_size, 0.0)
        self.client_buffer_size = max(self.client_buffer_size - delay, 0.0)
        self.client_buffer_size += VIDEO_CHUNCK_LEN

        # ---- Drain buffer if too large ---- #
        sleep_time: float = 0.0
        if self.client_buffer_size > BUFFER_THRESH:
            drain          = self.client_buffer_size - BUFFER_THRESH
            sleep_time     = math.ceil(drain / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME
            self.client_buffer_size -= sleep_time
            remaining_sleep = sleep_time
            while True:
                seg_dur = self.current_trace_times[self.bandwidth_ptr] - self.last_bandwidth_time
                if seg_dur > remaining_sleep / MILLISECONDS_IN_SECOND:
                    self.last_bandwidth_time += remaining_sleep / MILLISECONDS_IN_SECOND
                    break
                remaining_sleep -= seg_dur * MILLISECONDS_IN_SECOND
                self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr]
                self.bandwidth_ptr += 1
                if self.bandwidth_ptr >= len(self.current_bandwidth):
                    self.bandwidth_ptr = 1
                    self.last_bandwidth_time = self.current_trace_times[self.bandwidth_ptr - 1]

        # ---- Advance chunk counter ---- #
        self.video_chunk_cnt += 1
        video_chunk_remain = self.TOTAL_VIDEO_CHUNCK - self.video_chunk_cnt
        end_of_video = self.video_chunk_cnt >= self.TOTAL_VIDEO_CHUNCK

        if end_of_video:
            self.client_buffer_size = 0.0
            self.video_chunk_cnt    = 0
            self._pick_trace()

        next_sizes = np.array(
            [self.video_chunk_sizes[lvl][self.video_chunk_cnt] for lvl in range(BITRATE_LEVELS)],
            dtype=np.int32,
        )
        return {
            "delay_ms":                       np.float32(delay),
            "sleep_time_ms":                  np.float32(sleep_time),
            "buffer_size_ms":                 np.float32(self.client_buffer_size),
            "rebuffer_ms":                    np.float32(wait_rebuf_time),
            "selected_video_chunk_size_bytes": np.int32(selected_chunk_size),
            "is_done_bool":                   end_of_video,
            "remain_chunk":                   np.int32(video_chunk_remain),
            "next_video_chunk_sizes":         next_sizes,
        }

    # ---------------------------------------------------------------------- #
    #  Data loading (uses bundled datasets via DATA_ROOT)
    # ---------------------------------------------------------------------- #

    def _load_bandwidth_trace(
        self, trace_name: str, data_split: str
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Load all trace files via nettrace package."""
        suite = "ABRBench-3G" if trace_name in _3G_SETS else "ABRBench-4G+"
        # "all" 模式的集合（HSR/Ghent/Lab）文件直接在 set 根目录，
        # list_trace_files 的 split 参数对应子目录，不传 split 时用 "all"
        # 用 nettrace.list_trace_files 的 split 参数支持 train/test；
        # "all" 时手动列出根目录所有文件。
        from nettrace.utils import DATA_ROOT as _DATA_ROOT

        if data_split == "all":
            root = _DATA_ROOT / "trace" / suite / trace_name
            files = sorted(
                [p for p in root.iterdir() if p.is_file() and not p.name.startswith(".")],
                key=lambda p: p.name,
            )
        else:
            files = list_trace_files(trace_name, suite=suite, split=data_split)  # type: ignore[arg-type]

        time_seqs: list[list[float]] = []
        bw_seqs:   list[list[float]] = []
        for f in files:
            trace = load_trace_file(f)
            time_seqs.append(list(trace.times))
            bw_seqs.append(list(trace.bandwidths))

        return time_seqs, bw_seqs

    def _load_video_sizes_by_bitrate(self) -> dict[int, list[int]]:
        """Load video chunk sizes via nettrace package."""
        sizes = load_video_sizes(self.video_type, bitrate_levels=BITRATE_LEVELS)
        return {level: list(chunks) for level, chunks in sizes.items()}


# --------------------------------------------------------------------------- #
#  main: 用 nettrace 数据集跑一个完整 episode，逐 chunk 打印网络模拟结果
# --------------------------------------------------------------------------- #

def main() -> None:
    SEED       = 42
    TRACE_NAME = "SolisWi-Fi"
    DATA_SPLIT = "train"
    VIDEO_TYPE = "big_buck_bunny"
    QOE_TYPE   = "normal"
    # 固定码率策略：始终选 level 2（1200 Kbps），方便 debug
    FIXED_ACTION = 2

    print("=" * 60)
    print("VideoStreamingEnv — 网络带宽模拟 debug 运行")
    print("=" * 60)
    print(f"  trace:      {TRACE_NAME} / {DATA_SPLIT}")
    print(f"  video:      {VIDEO_TYPE}")
    print(f"  qoe_type:   {QOE_TYPE}")
    print(f"  seed:       {SEED}")
    print(f"  action:     固定 level={FIXED_ACTION}  "
          f"({[300,750,1200,1850,2850,4300][FIXED_ACTION]} Kbps)")
    print("=" * 60)

    env = VideoStreamingEnv(TRACE_NAME, DATA_SPLIT, VIDEO_TYPE, QOE_TYPE, seed=SEED)

    # ── reset ──────────────────────────────────────────────────────────────── #
    obs, _ = env.reset(seed=SEED)
    print(f"\n[reset]  选中 trace index={env.trace_index}"
          f"  bw_ptr={env.bandwidth_ptr}"
          f"  trace 长度={len(env.current_bandwidth)} 点")
    print(f"  当前带宽点: {env.current_bandwidth[env.bandwidth_ptr]:.3f} Mbps")
    _print_obs(0, obs, reward=None, info=None)

    # ── episode loop ───────────────────────────────────────────────────────── #
    total_reward      = 0.0
    total_rebuf_s     = 0.0
    bitrate_history   = []
    step = 0
    while True:
        step += 1
        obs, reward, terminated, _, info = env.step(FIXED_ACTION)
        total_reward  += reward
        total_rebuf_s += float(obs["rebuffer_ms"]) / 1000.0
        bitrate_history.append(FIXED_ACTION)
        _print_obs(step, obs, reward, info)
        if terminated:
            break

    # ── summary ───────────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("Episode 结束统计")
    print("=" * 60)
    print(f"  总步数 (step 数):   {step}")
    print(f"  累计 reward:        {total_reward:.4f}")
    print(f"  累计卡顿时长:       {total_rebuf_s:.3f} s")
    avg_bw = sum(env.current_bandwidth) / len(env.current_bandwidth)
    print(f"  本 episode 使用 trace 的平均带宽: {avg_bw:.3f} Mbps")

    # ── 可复现性验证 ───────────────────────────────────────────────────────── #
    print("\n[可复现性验证] 相同 seed 再跑一次，首个 obs 应完全一致 ...")
    env2 = VideoStreamingEnv(TRACE_NAME, DATA_SPLIT, VIDEO_TYPE, QOE_TYPE, seed=SEED)
    obs2, _ = env2.reset(seed=SEED)
    match = (
        float(obs2["delay_ms"])    == float(obs["delay_ms"])    and  # noqa
        int(obs2["remain_chunk"]) == int(obs["remain_chunk"])
    )

    # obs 是 episode 最后一步，obs2 是第二次 reset 的第一步，不应比较，
    # 我们重新取 reset obs 做对比
    env_a = VideoStreamingEnv(TRACE_NAME, DATA_SPLIT, VIDEO_TYPE, QOE_TYPE, seed=SEED)
    obs_a, _ = env_a.reset(seed=SEED)
    env_b = VideoStreamingEnv(TRACE_NAME, DATA_SPLIT, VIDEO_TYPE, QOE_TYPE, seed=SEED)
    obs_b, _ = env_b.reset(seed=SEED)
    same_delay  = float(obs_a["delay_ms"])   == float(obs_b["delay_ms"])
    same_remain = int(obs_a["remain_chunk"]) == int(obs_b["remain_chunk"])
    same_bw     = (obs_a["next_video_chunk_sizes"] == obs_b["next_video_chunk_sizes"]).all()
    print(f"  delay_ms 相同:              {same_delay}  "
          f"({float(obs_a['delay_ms']):.2f} ms)")
    print(f"  remain_chunk 相同:          {same_remain}  "
          f"({int(obs_a['remain_chunk'])})")
    print(f"  next_video_chunk_sizes 相同: {same_bw}")
    if same_delay and same_remain and same_bw:
        print("  ✓ 可复现性验证通过")
    else:
        print("  ✗ 可复现性验证失败！请检查 seed 设置")


def _print_obs(
    step: int,
    obs: dict,
    reward: float | None,
    info: dict | None,
) -> None:
    tag = f"[step {step:2d}]" if step > 0 else "[reset  ]"
    bw_str = "/".join(f"{v:6d}" for v in obs["next_video_chunk_sizes"].tolist())
    reward_str = f"  reward={reward:+.4f}" if reward is not None else ""
    rebuf_str  = ""
    if info is not None:
        rebuf_str = (f"  (bw={info['bitrate_reward']:+.3f}"
                     f"  rebuf={info['rebuffer_time_reward']:+.3f}"
                     f"  smooth={info['smooth_penalty_reward']:+.3f})")
    print(
        f"{tag}"
        f"  delay={float(obs['delay_ms']):7.1f}ms"
        f"  rebuf={float(obs['rebuffer_ms']):6.1f}ms"
        f"  buf={float(obs['buffer_size_ms'])/1000:5.1f}s"
        f"  sleep={float(obs['sleep_time_ms']):5.0f}ms"
        f"  remain={int(obs['remain_chunk']):2d}"
        f"  chunk={int(obs['selected_video_chunk_size_bytes'])//1024:5d}KB"
        f"  next(KB)=[{bw_str}]"
        + reward_str + rebuf_str
    )


if __name__ == "__main__":
    main()
