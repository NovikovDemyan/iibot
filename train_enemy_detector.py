import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/yolo_enemy/data.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="Examples: 0 for GPU, cpu for CPU")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="cs2_enemy_detector")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"data.yaml not found: {args.data}")

    model = YOLO(args.model)

    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "patience": 8,
        "workers": 0,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("\nTraining finished.")
    print(f"Best model: {best}")


if __name__ == "__main__":
    main()
