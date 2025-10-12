# CritiQ/push_config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Any, Dict
import json
import os
import pathlib

try:
    import tomllib  # py311+
except Exception:  # pragma: no cover
    tomllib = None
try:
    import tomli  # py310
except Exception:  # pragma: no cover
    tomli = None

# yaml(Optional)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class TrainConfig:
    teacher_path: str = field(
        default="saved_models/teacher_push.zip",
        metadata={"help": "Pretrained teacher model (with full observability)"},
    )
    student_path: str = field(
        default="n",
        metadata={"help": 'Pretrained student model path, or "n" to train from scratch'},
    )
    discriminator_path: str = field(
        default="n",
        metadata={"help": 'Pretrained discriminator model path, or "n" to train from scratch'},
    )
    save_path: str = field(
        default="./critiq_test", metadata={"help": "Directory to save trained models and logs"}
    )
    save_video: bool = field(default=True, metadata={"help": "Save teacher augmentation and validation videos"})
    buffer_path: str = field(
        default="./buffer/buffer_1k.pkl", metadata={"help": "Pre-collected expert demonstration data"}
    )
    new_buffer_path: Optional[str] = field(default=None, metadata={"help": "Where to save new buffer (None = don't save)"})

    num_traj: int = field(default=20000, metadata={"help": "Trajectories to collect if collect_data is True"})
    num_epoch: int = field(default=100, metadata={"help": "Training epochs per round"})
    max_episode_steps: int = field(default=200, metadata={"help": "Max steps per episode"})
    eval_its: int = field(default=20, metadata={"help": "Number of evaluation episodes"})
    num_cpu: int = field(default=16, metadata={"help": "Number of CPU workers for data collection"})

    collect_data: bool = field(default=False, metadata={"help": "Collect teacher demonstrations before training"})
    collect_more_data: bool = field(default=False, metadata={"help": "Collect additional expert data on top of existing buffer"})
    data_aug_ep: int = field(default=20, metadata={"help": "Episodes per data augmentation round"})
    seq_len: int = field(default=20, metadata={"help": "Sequence length used by discriminator"})
    cnt_threshold: int = field(default=7, metadata={"help": "Consecutive low-probability steps threshold for failure"})
    aug_traj_mul: int = field(default=2, metadata={"help": "Multiplier applied to number of failed states for teacher data collection"})

    env_id: str = field(default="singlepush", metadata={"help": "Environment id"})

    device: Optional[str] = field(default=None, metadata={"help": "Device override (cpu/cuda)"})


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TestConfig:
    teacher_path: str = field(default="saved_models/teacher_push.zip", metadata={"help": "Teacher model for testing"})
    student_path: str = field(default="saved_models/student_push.pth", metadata={"help": "Student model for testing"})
    discriminator_path: str = field(default="saved_models/discriminator_push.pth", metadata={"help": "Discriminator model for testing"})
    save_path: str = field(default="test_critiq", metadata={"help": "Directory to save test data (video)"})
    num_episodes: int = field(default=20, metadata={"help": "Number of episodes to run for testing"})
    max_episode_steps: int = field(default=200, metadata={"help": "Max steps per test episode"})
    difficulty: int = field(default=3, metadata={"help": "Environment difficulty setting"})
    seq_len: int = field(default=20, metadata={"help": "Sequence length for test-time discriminator input"})
    env_id: str = field(default="singlepush", metadata={"help": "Environment id for testing"})
    device: Optional[str] = field(default=None, metadata={"help": "Device override (cpu/cuda)"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_config_schema() -> Dict[str, Dict[str, Any]]:
    """Return a dict describing TrainConfig fields: default and help text.

    Useful for generating help or documentation programmatically.
    """
    import dataclasses

    schema: Dict[str, Dict[str, Any]] = {}
    for f in dataclasses.fields(TrainConfig):
        schema[f.name] = {"default": f.default, "help": f.metadata.get("help", "")}
    return schema

def _load_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f: return json.load(f)

def _load_yaml(p: str) -> Dict[str, Any]:
    if yaml is None: raise ImportError("PyYAML not installed: run `pip install pyyaml` or use JSON/TOML config files")
    with open(p, "r") as f: return yaml.safe_load(f) or {}

def _load_toml(p: str) -> Dict[str, Any]:
    if tomllib is not None:
        with open(p, "rb") as f: return tomllib.load(f)
    if tomli is not None:
        with open(p, "rb") as f: return tomli.load(f)
    raise ImportError("No TOML loader available: use Python 3.11+ (tomllib) or install the 'tomli' package")

def _load_any(cfg_path: str) -> Dict[str, Any]:
    path = pathlib.Path(cfg_path)
    if not path.exists(): raise FileNotFoundError(f"Config file not found: {cfg_path}")
    ext = path.suffix.lower()
    if ext == ".json": return _load_json(str(path))
    if ext in (".yml", ".yaml"): return _load_yaml(str(path))
    if ext == ".toml": return _load_toml(str(path))
    raise ValueError(f"Unsupported file extension: {ext} (supported: json, yaml, toml)")

def load_config(cfg_path: str) -> TrainConfig:
    raw = _load_any(cfg_path)
    if "trained_id" in raw: raw["trained_id"] = int(raw["trained_id"])
    return TrainConfig(**raw)


def load_config_with_raw(cfg_path: str) -> tuple[TrainConfig, Dict[str, Any]]:
    """Load config and also return the raw dict loaded from the file.

    This is useful when you want to distinguish fields that were explicitly provided
    in the config file vs. fields that come from dataclass defaults.
    Returns: (TrainConfig, raw_dict)
    """
    raw = _load_any(cfg_path)
    if "trained_id" in raw:
        raw["trained_id"] = int(raw["trained_id"])
    cfg = TrainConfig(**raw)
    return cfg, raw


def save_provided_config(raw_cfg: Dict[str, Any], out_dir: str, filename_base: str = "config_provided") -> None:
    """Save only the raw (user-provided) configuration dict to the output directory.

    This avoids writing back dataclass defaults into a file intended to show only
    the user's explicit overrides.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{filename_base}.json"), "w") as f:
        json.dump(raw_cfg, f, indent=2)
def load_test_config(cfg_path: str) -> TestConfig:
    raw = _load_any(cfg_path)
    return TestConfig(**raw)

def save_config(cfg: TrainConfig, out_dir: str, filename_base: str = "config_used") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{filename_base}.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)