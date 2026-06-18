import argparse
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("dataset/raw_frames"))
    parser.add_argument("--dst", type=Path, default=Path("dataset/labeling_frames_extra"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--low-conf-thr", type=float, default=0.45)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    imgs = list_images(args.source)
    if not imgs:
        raise SystemExit(f"No images found: {args.source}")

    args.dst.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))

    low_conf = []
    no_det = []
    high_conf = []

    print(f"Scanning {len(imgs)} raw frames...")

    for idx, img_path in enumerate(imgs, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            no_det.append((img_path, 0.0))
        else:
            confs = boxes.conf.cpu().numpy().tolist()
            max_conf = max(confs)
            if max_conf < args.low_conf_thr:
                low_conf.append((img_path, max_conf))
            else:
                high_conf.append((img_path, max_conf))

        if idx % 250 == 0:
            print(f"Processed: {idx}/{len(imgs)}")

    low_conf.sort(key=lambda x: x[1])

    # Берём сначала сомнительные low_conf, потом no_det.
    selected = []
    selected.extend(low_conf[: args.count])

    if len(selected) < args.count:
        need = args.count - len(selected)
        selected.extend(no_det[:need])

    copied = 0
    for i, (src_path, conf) in enumerate(selected, start=1):
        # Имя делаем коротким и уникальным, чтобы labels точно совпадали с images.
        dst_name = f"extra_{i:05d}_conf_{conf:.3f}_{src_path.name}"
        dst_path = args.dst / dst_name
        shutil.copy2(src_path, dst_path)
        copied += 1

    print("\nDone.")
    print(f"low_conf found: {len(low_conf)}")
    print(f"no_det found:   {len(no_det)}")
    print(f"high_conf found:{len(high_conf)}")
    print(f"Copied clean originals: {copied}")
    print(f"Output folder: {args.dst}")
    print("\nNext command:")
    print(f'.\\.venv\\Scripts\\python.exe label_enemy.py --images "{args.dst}" --labels dataset/labels_extra')


if __name__ == "__main__":
    main()
