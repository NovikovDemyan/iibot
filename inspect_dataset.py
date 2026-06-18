import argparse
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("dataset/labeling_frames"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    args = parser.parse_args()

    imgs = list_images(args.images)
    labeled = 0
    positive = 0
    negative = 0
    boxes = 0
    missing = 0

    for img in imgs:
        label = args.labels / f"{img.stem}.txt"
        if not label.exists():
            missing += 1
            continue

        labeled += 1
        lines = [ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            positive += 1
            boxes += len(lines)
        else:
            negative += 1

    print(f"Images: {len(imgs)}")
    print(f"Labeled: {labeled}")
    print(f"Missing labels: {missing}")
    print(f"Positive images: {positive}")
    print(f"Negative images: {negative}")
    print(f"Total boxes: {boxes}")

    if labeled:
        print(f"Avg boxes per labeled image: {boxes / labeled:.2f}")
    if positive:
        print(f"Avg boxes per positive image: {boxes / positive:.2f}")


if __name__ == "__main__":
    main()
