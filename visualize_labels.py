import argparse
from pathlib import Path
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def read_yolo_boxes(label_path: Path, w: int, h: int):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = parts
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        boxes.append((x1, y1, x2, y2, cls))
    return boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("dataset/labeling_frames"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--out", type=Path, default=Path("gt_preview"))
    args = parser.parse_args()

    imgs = list_images(args.images)
    if not imgs:
        raise SystemExit(f"No images found: {args.images}")

    args.out.mkdir(parents=True, exist_ok=True)
    saved = 0

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = args.labels / f"{img_path.stem}.txt"
        boxes = read_yolo_boxes(label_path, w, h)

        for x1, y1, x2, y2, cls in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(img, f"gt:{cls}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(args.out / img_path.name), img)
        saved += 1

    print(f"Saved GT previews: {saved}")
    print(f"Output: {args.out}")

if __name__ == "__main__":
    main()
