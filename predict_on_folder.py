import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path):
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("dataset/labeling_frames"))
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    imgs = list_images(args.source)
    if not imgs:
        raise SystemExit(f"No images found: {args.source}")

    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))

    saved = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(img, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        out_path = args.out / img_path.name
        cv2.imwrite(str(out_path), annotated)
        saved += 1

        if saved % 50 == 0:
            print(f"Saved predictions: {saved}")

    print(f"Done. Saved: {saved}")
    print(f"Output folder: {args.out}")


if __name__ == "__main__":
    main()
