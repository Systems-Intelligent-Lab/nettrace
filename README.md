# nettrace

`nettrace` 是一个**自带数据集、零外部依赖**的网络带宽 trace 加载工具包，专为 ABR 自适应码率、网络仿真等研究场景设计。

核心特性：
- **内置 ABRBench 数据集**：12 个 trace 集合（3G/4G+/WiFi），2 种视频 chunk 元数据
- **可复现采样**：文件列表按文件名排序 + `random.Random(seed)`，seed 相同则每次拿到同一条 trace
- **纯标准库**：运行时无任何第三方依赖，直接 `pip install`

---

## 安装

```bash
# 本地开发（可编辑安装，推荐）
pip install -e .

# 含测试依赖
pip install -e ".[dev]"
```

> **注意**：如果你在多个 Python 环境（如 conda env）之间切换，需要在每个环境里分别执行安装命令，然后用该环境的 `python` 运行脚本。

---

## 快速开始

```python
from nettrace import list_trace_sets, list_trace_files, load_trace_file, sample_trace, load_video_sizes

# 查看所有可用 trace 集合
print(list_trace_sets("ABRBench-4G+"))
# ['Ghent', 'Lab', 'Lumos4G', 'Lumos5G', 'Norway3G', 'SolisWi-Fi']

# 随机采样一条 trace（seed 保证可复现）
trace = sample_trace("SolisWi-Fi", suite="ABRBench-4G+", split="train", seed=42)
print(trace.path.name)          # wifi_lab_231114-162511.txt（seed=42 固定）
print(len(trace.times))         # 200
print(trace.bandwidths[:3])     # (30.6, 13.4, 13.4)

# 加载视频 chunk 大小（每个码率层）
sizes = load_video_sizes("big_buck_bunny", bitrate_levels=6)
print([round(sum(sizes[i]) / len(sizes[i]) / 1024) for i in range(6)])
# [483, 1210, 2412, 3867, 7734, 19272]  (KB，码率越高 chunk 越大)
```

运行完整示例：

```bash
python examples/quickstart.py
```

---

## 数据集

### Trace 数据集

| Suite | 集合 | Split | 文件数 |
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

> HSR、Ghent、Lab 无 train/test 划分，使用时 `split="all"`。

每条 trace 文件格式（空格分隔的两列）：

```
0.0  25.5
1.0   7.71
2.0   7.97
...
```

第一列为时间戳（秒），第二列为带宽（Mbps）。

### 视频元数据

| 视频 | 码率层数 | Chunk 数 |
|---|---|---|
| `big_buck_bunny` | 6 | 49 |
| `envivio_3g` | 6 | 49 |

码率档位：300 / 750 / 1200 / 1850 / 2850 / 4300 Kbps

---

## API

### `list_trace_sets(suite) -> list[str]`

列出指定 suite 下所有 trace 集合名（已排序）。

```python
list_trace_sets("ABRBench-3G")   # ['FCC-16', 'FCC-18', 'HSR', ...]
list_trace_sets("ABRBench-4G+")  # ['Ghent', 'Lab', 'Lumos4G', ...]
```

### `list_trace_files(trace_set, *, suite, split) -> list[Path]`

列出某个集合下某个 split 的所有 trace 文件路径（**按文件名排序**，顺序稳定）。

```python
files = list_trace_files("SolisWi-Fi", suite="ABRBench-4G+", split="train")
```

### `load_trace_file(path) -> Trace`

解析单个 trace 文件，返回 `Trace` 对象。

### `sample_trace(trace_set, *, suite, split, seed) -> Trace`

从 trace 集合中**可复现地随机采样**一条 trace 并加载。

```python
trace = sample_trace("FCC-16", suite="ABRBench-3G", split="test", seed=0)
```

### `load_video_sizes(video_type, *, bitrate_levels) -> dict[int, tuple[int, ...]]`

加载视频 chunk 大小（字节），按码率层索引。

```python
sizes = load_video_sizes("envivio_3g", bitrate_levels=6)
sizes[0]  # level 0 的所有 chunk 大小
```

### `Trace`（dataclass）

| 字段 | 类型 | 说明 |
|---|---|---|
| `times` | `tuple[float, ...]` | 时间戳序列（秒） |
| `bandwidths` | `tuple[float, ...]` | 带宽序列（Mbps） |
| `path` | `pathlib.Path` | 原始文件绝对路径 |

---

## 可复现性

`sample_trace(..., seed=SEED)` 的复现保证：

1. `list_trace_files` 返回的文件列表**按文件名字母序排列**（不依赖文件系统顺序）
2. 用 `random.Random(seed)` 从该列表中取索引

因此只要数据集内容不变，**同一 seed 始终选中同一文件、读出同一 trace**。

---

## 示例：在 gym 环境中使用

`examples/env.py` 提供了一个完整的 ABR 视频流 gym 环境示例，展示如何将 `nettrace` 作为数据后端：

```python
# examples/env.py 中的加载方式
from nettrace import list_trace_files, load_trace_file, load_video_sizes

# 批量加载所有 trace（env 内部使用）
files = list_trace_files(trace_name, suite=suite, split=data_split)
for f in files:
    trace = load_trace_file(f)
    time_seqs.append(list(trace.times))
    bw_seqs.append(list(trace.bandwidths))

# 加载视频 chunk 大小
sizes = load_video_sizes(video_type, bitrate_levels=6)
```

运行 debug 示例（需在对应 conda 环境中安装 `gymnasium` 和 `numpy`）：

```bash
python examples/env.py
```

---

## 测试

```bash
pytest
```

测试覆盖：
- 所有 trace 集合可列出且顺序稳定
- 文件列表跨调用结果一致
- trace 文件正确解析（时间非负、带宽为正、双列等长）
- 批量加载所有文件无报错
- video sizes 各层 chunk 数一致、值为正、高码率层均值更大
- seed 相同选中同一文件、内容一致
- seed 不同选中不同文件

---

## 项目结构

```
nettrace/
├── nettrace/
│   ├── __init__.py       # 对外公共 API
│   ├── core.py           # Trace、list_trace_sets、load_trace_file 等实现
│   ├── utils.py          # DATA_ROOT 路径常量
│   └── datasets/
│       ├── trace/
│       │   ├── ABRBench-3G/   (FCC-16, FCC-18, HSR, Oboe, Puffer-21, Puffer-22)
│       │   └── ABRBench-4G+/  (Ghent, Lab, Lumos4G, Lumos5G, Norway3G, SolisWi-Fi)
│       └── video/
│           ├── big_buck_bunny/
│           └── envivio_3g/
├── examples/
│   ├── quickstart.py     # 最简使用示例
│   └── env.py            # ABR gym 环境示例（需要 gymnasium + numpy）
├── tests/
│   ├── test_reproducible_sampling.py
│   └── test_env_dependencies.py
├── pyproject.toml
└── MANIFEST.in
```
