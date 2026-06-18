import argparse
import random
import shutil
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def collect_pairs(img_dir: Path, lbl_dir: Path):
    pairs = []
    for img in list_images(img_dir):
        label = lbl_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
    return pairs


def copy_pair(img_path: Path, label_path: Path, out_img_dir: Path, out_lbl_dir: Path, prefix: str):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{prefix}_{img_path.stem}"
    dst_img = out_img_dir / f"{stem}{img_path.suffix.lower()}"
    dst_lbl = out_lbl_dir / f"{stem}.txt"

    shutil.copy2(img_path, dst_img)
    shutil.copy2(label_path, dst_lbl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Pairs of image_dir label_dir. Example: --pairs dataset/labeling_frames dataset/labels dataset/labeling_frames_extra dataset/labels_extra",
    )
    parser.add_argument("--out", type=Path, default=Path("dataset/yolo_enemy_v2"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        raise SystemExit("--pairs must contain even number of paths: image_dir label_dir ...")

    all_pairs = []

    for idx in range(0, len(args.pairs), 2):
        img_dir = Path(args.pairs[idx])
        lbl_dir = Path(args.pairs[idx + 1])

        if not img_dir.exists():
            raise SystemExit(f"Image dir not found: {img_dir}")
        if not lbl_dir.exists():
            raise SystemExit(f"Label dir not found: {lbl_dir}")

        pairs = collect_pairs(img_dir, lbl_dir)
        prefix = f"set{idx // 2 + 1}"
        all_pairs.extend([(prefix, img, lbl) for img, lbl in pairs])
        print(f"{img_dir} + {lbl_dir}: {len(pairs)} labeled pairs")

    if not all_pairs:
        raise SystemExit("No labeled pairs found.")

    random.seed(args.seed)
    random.shuffle(all_pairs)

    val_n = max(1, int(len(all_pairs) * args.val_ratio))
    val_pairs = all_pairs[:val_n]
    train_pairs = all_pairs[val_n:]

    if args.out.exists():
        shutil.rmtree(args.out)

    for prefix, img, label in train_pairs:
        copy_pair(
            img,
            label,
            args.out / "images" / "train",
            args.out / "labels" / "train",
            prefix,
        )

    for prefix, img, label in val_pairs:
        copy_pair(
            img,
            label,
            args.out / "images" / "val",
            args.out / "labels" / "val",
            prefix,
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

    print("\nYOLO dataset created.")
    print(f"Output: {args.out}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")
    print(f"Config: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
