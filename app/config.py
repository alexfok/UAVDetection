from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoConfig:
    source: str = "0"
    camera_config: str = "data_store/system_config/cameras.yaml"
    resize_width: int = 1920
    resize_height: int = 1080
    frame_skip: int = 0
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0
    buffer_size: int = 1


@dataclass
class DetectorConfig:
    model_path: str = "data_store/models/trained/yolov8n_drone_best.pt"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.45
    image_size: int = 960
    device: str = ""
    target_classes: list[str] = field(default_factory=lambda: ["drone"])
    label_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "airplane": "drone",
            "kite": "drone",
        }
    )


@dataclass
class TrackerConfig:
    iou_match_threshold: float = 0.25
    max_track_age_sec: float = 2.0
    min_box_area: int = 8


@dataclass
class AlertConfig:
    confidence_threshold: float = 0.3
    persistence_frames: int = 5
    window_seconds: float = 2.0
    cooldown_seconds: float = 1.0


@dataclass
class UIConfig:
    window_name: str = "Fast Drone Detection PoC"
    show_window: bool = True
    fullscreen: bool = False
    draw_all_tracks: bool = True
    draw_status_bar: bool = True
    save_output: bool = False
    output_path: str = "videos/annotated_output.mp4"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key: {key}")

        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path = "configs/config.yaml") -> AppConfig:
    import yaml

    config = AppConfig()
    config_path = Path(path)

    if not config_path.exists():
        return config

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return _merge_dataclass(config, data)
