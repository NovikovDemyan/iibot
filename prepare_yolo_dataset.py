import argparse
import random
import shutil
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def copy_pair(img_path: Path, label_path: Path, out_img_dir: Path, out_lbl_dir: Path):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    dst_img = out_img_dir / img_path.name
    dst_lbl = out_lbl_dir / f"{img_path.stem}.txt"

    shutil.copy2(img_path, dst_img)
    shutil.copy2(label_path, dst_lbl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("dataset/labeling_frames"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--out", type=Path, default=Path("dataset/yolo_enemy"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    imgs = list_images(args.images)

    pairs = []
    for img in imgs:
        label = args.labels / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))

    if not pairs:
        raise SystemExit("No labeled image/label pairs found. Run label_enemy.py first.")

    random.seed(args.seed)
    random.shuffle(pairs)

    val_n = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:val_n]
    train_pairs = pairs[val_n:]

    if args.out.exists():
        shutil.rmtree(args.out)

    for img, label in train_pairs:
        copy_pair(
            img,
            label,
            args.out / "images" / "train",
            args.out / "labels" / "train",
        )

    for img, label in val_pairs:
        copy_pair(
            img,
            label,
            args.out / "images" / "val",
            args.out / "labels" / "val",
        )

    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "enemy"},
    }

    (args.out / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"YOLO dataset created: {args.out}")
    print(f"Train images: {len(train_pairs)}")
    print(f"Val images: {len(val_pairs)}")
    print(f"Config: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
