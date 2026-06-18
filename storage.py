from pathlib import Path
import time

import cv2


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_frame(frame_bgr, out_dir: Path, prefix: str = "frame", quality: int = 95) -> Path:
    ensure_dir(out_dir)

    ts_ms = int(time.time() * 1000)
    out_path = out_dir / f"{prefix}_{ts_ms}.jpg"

    ok = cv2.imwrite(
        str(out_path),
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )

    if not ok:
        raise RuntimeError(f"Failed to save frame: {out_path}")

    return out_path
