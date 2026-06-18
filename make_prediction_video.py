import argparse
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
    parser.add_argument("--out", type=Path, default=Path("detector_demo.mp4"))
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    imgs = list_images(args.source)
    if not imgs:
        raise SystemExit(f"No images found: {args.source}")

    imgs = imgs[:args.max_frames]
    model = YOLO(str(args.model))

    first = cv2.imread(str(imgs[0]))
    if first is None:
        raise SystemExit("Failed to read first frame.")

    if args.scale != 1.0:
        first = cv2.resize(first, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    written = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        annotated = result.plot()

        if args.scale != 1.0:
            annotated = cv2.resize(annotated, (w, h), interpolation=cv2.INTER_AREA)

        cv2.putText(annotated, f"CS2 enemy detector | conf={args.conf}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)
        written += 1

        if written % 100 == 0:
            print(f"Written frames: {written}")

    writer.release()
    print(f"Video saved: {args.out}")
    print(f"Frames written: {written}")

if __name__ == "__main__":
    main()
