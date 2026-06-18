import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("dataset/yolo_enemy/data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.data.exists():
        raise SystemExit(f"Data config not found: {args.data}")

    model = YOLO(str(args.model))
    kwargs = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "split": "val",
        "verbose": True,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    metrics = model.val(**kwargs)

    print("\nMain metrics:")
    try:
        print(f"mAP50:     {metrics.box.map50:.4f}")
        print(f"mAP50-95:  {metrics.box.map:.4f}")
        print(f"precision: {metrics.box.mp:.4f}")
        print(f"recall:    {metrics.box.mr:.4f}")
    except Exception:
        print(metrics)

if __name__ == "__main__":
    main()
