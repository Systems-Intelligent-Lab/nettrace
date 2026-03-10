from pathlib import Path

# 获取包安装后的数据集绝对路径
DATA_ROOT = Path(__file__).parent / "datasets"

def get_trace_root() -> Path:
    return DATA_ROOT / "trace"


def get_video_root() -> Path:
    return DATA_ROOT / "video"


def get_trace_dir(suite: str, trace_set: str, split: str) -> Path:
    """Get a trace directory path like: datasets/trace/<suite>/<trace_set>/<split>."""
    return get_trace_root() / suite / trace_set / split