import argparse
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(src: Path):
    return sorted([p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("dataset/raw_frames"))
    parser.add_argument("--dst", type=Path, default=Path("dataset/labeling_frames"))
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    imgs = list_images(args.src)
    if not imgs:
        raise SystemExit(f"No images found in {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)

    n = min(args.count, len(imgs))
    if n <= 0:
        raise SystemExit("count must be positive")

    # Равномерная выборка по всей записи, чтобы не брать 500 почти одинаковых кадров подряд.
    if n == 1:
        selected = [imgs[0]]
    else:
        idxs = [round(i * (len(imgs) - 1) / (n - 1)) for i in range(n)]
        selected = [imgs[i] for i in idxs]

    copied = 0
    for i, src_path in enumerate(selected, start=1):
        dst_name = f"sample_{i:05d}{src_path.suffix.lower()}"
        dst_path = args.dst / dst_name
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"Source frames: {len(imgs)}")
    print(f"Copied frames: {copied}")
    print(f"Output folder: {args.dst}")


if __name__ == "__main__":
    main()
