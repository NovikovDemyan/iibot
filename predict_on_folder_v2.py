import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from target_logic import detections_from_yolo_result, choose_target
from draw_target import draw_detections

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path):
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("dataset/raw_frames"))
    parser.add_argument("--out", type=Path, default=Path("predictions_v2"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    imgs = list_images(args.source)[: args.limit]
    if not imgs:
        raise SystemExit(f"No images found: {args.source}")

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    saved = 0
    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = detections_from_yolo_result(result, min_conf=args.conf)
        choice = choose_target(dets, frame.shape[1], frame.shape[0])
        vis = draw_detections(frame.copy(), dets, choice)
        cv2.imwrite(str(args.out / img_path.name), vis)
        saved += 1

        if saved % 100 == 0:
            print(f"Saved: {saved}")

    print(f"Done. Saved: {saved}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
