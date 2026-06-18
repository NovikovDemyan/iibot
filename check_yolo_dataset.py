import argparse
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def count_labels(label_dir: Path):
    label_files = sorted(label_dir.rglob("*.txt"))
    images_with_boxes = 0
    total_boxes = 0
    empty_labels = 0
    malformed = 0

    for label in label_files:
        lines = [ln.strip() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            empty_labels += 1
            continue

        images_with_boxes += 1
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                malformed += 1
                continue
            total_boxes += 1

            try:
                cls, xc, yc, w, h = parts
                vals = [float(xc), float(yc), float(w), float(h)]
                if any(v < 0 or v > 1 for v in vals):
                    malformed += 1
            except ValueError:
                malformed += 1

    return {
        "label_files": len(label_files),
        "images_with_boxes": images_with_boxes,
        "empty_labels": empty_labels,
        "total_boxes": total_boxes,
        "malformed": malformed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/yolo_enemy/data.yaml"))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"data.yaml not found: {args.data}")

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(data["path"])

    train_img_dir = root / data["train"]
    val_img_dir = root / data["val"]

    train_lbl_dir = root / "labels" / "train"
    val_lbl_dir = root / "labels" / "val"

    train_imgs = list_images(train_img_dir)
    val_imgs = list_images(val_img_dir)

    print(f"Dataset root: {root}")
    print(f"Train images: {len(train_imgs)}")
    print(f"Val images: {len(val_imgs)}")

    tr = count_labels(train_lbl_dir)
    va = count_labels(val_lbl_dir)

    print("\nTrain labels:")
    for k, v in tr.items():
        print(f"  {k}: {v}")

    print("\nVal labels:")
    for k, v in va.items():
        print(f"  {k}: {v}")

    if tr["malformed"] or va["malformed"]:
        print("\nWARNING: malformed labels found. Fix them before training.")
    else:
        print("\nOK: labels look structurally valid.")


if __name__ == "__main__":
    main()
