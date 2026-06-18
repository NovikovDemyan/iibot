from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaptureConfig:
    # None = весь основной экран.
    region: dict | None = None

    # Размер окна предпросмотра.
    preview_scale: float = 0.75

    # mss стабильнее для первого запуска.
    prefer_dxcam: bool = False


@dataclass
class DatasetConfig:
    raw_dir: Path = Path("dataset/raw_frames")
    captures_dir: Path = Path("captures")

    # Сохранять каждый N-й кадр.
    # 3 = примерно каждый третий кадр.
    save_every_n_frames: int = 3

    jpeg_quality: int = 95


CAPTURE = CaptureConfig()
DATASET = DatasetConfig()
