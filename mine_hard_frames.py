import argparse
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def save_annotated(result, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = result.plot()
    cv2.imwrite(str(out_path), annotated)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("dataset/raw_frames"))
    parser.add_argument("--out", type=Path, default=Path("hard_frames"))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    imgs = list_images(args.source)
    if not imgs:
        raise SystemExit(f"No images found: {args.source}")

    model = YOLO(str(args.model))
    low_conf, high_conf, no_det = [], [], []

    print(f"Scanning {len(imgs)} frames...")

    for idx, img_path in enumerate(imgs, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            no_det.append((img_path, 0.0, result))
        else:
            confs = boxes.conf.cpu().numpy().tolist()
            max_conf = max(confs)
            if max_conf < 0.45:
                low_conf.append((img_path, max_conf, result))
            else:
                high_conf.append((img_path, max_conf, result))

        if idx % 250 == 0:
            print(f"Processed: {idx}/{len(imgs)}")

    low_conf.sort(key=lambda x: x[1])
    high_conf.sort(key=lambda x: -x[1])

    if args.out.exists():
        shutil.rmtree(args.out)

    for name, items in [
        ("low_conf", low_conf[:args.limit]),
        ("no_det", no_det[:args.limit]),
        ("high_conf", high_conf[:args.limit]),
    ]:
        for i, (img_path, conf, result) in enumerate(items, start=1):
            out_name = f"{i:05d}_conf_{conf:.3f}_{img_path.name}"
            save_annotated(result, args.out / name / out_name)

    print("\nDone.")
    print(f"low_conf:  {len(low_conf)} -> saved {min(len(low_conf), args.limit)}")
    print(f"no_det:    {len(no_det)} -> saved {min(len(no_det), args.limit)}")
    print(f"high_conf: {len(high_conf)} -> saved {min(len(high_conf), args.limit)}")
    print(f"Output: {args.out}")

    print("\nNext:")
    print("- low_conf: first candidates for relabeling")
    print("- no_det: find frames where an enemy exists but detector missed")
    print("- high_conf: check false positives")

if __name__ == "__main__":
    main()
