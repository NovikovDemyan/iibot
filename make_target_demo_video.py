import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from target_logic import detections_from_yolo_result, choose_target
from draw_target import draw_detections

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("dataset/raw_frames"))
    parser.add_argument("--out", type=Path, default=Path("target_demo_v2.mp4"))
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    imgs = list_images(args.source)[: args.max_frames]
    if not imgs:
        raise SystemExit(f"No images found: {args.source}")

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
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = detections_from_yolo_result(result, min_conf=args.conf)
        choice = choose_target(dets, frame.shape[1], frame.shape[0])
        vis = draw_detections(frame.copy(), dets, choice)

        if args.scale != 1.0:
            vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(vis)
        written += 1
        if written % 100 == 0:
            print(f"Written: {written}")

    writer.release()
    print(f"Video saved: {args.out}")
    print(f"Frames written: {written}")


if __name__ == "__main__":
    main()
