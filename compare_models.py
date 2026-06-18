import argparse
from pathlib import Path
from ultralytics import YOLO


def eval_one(model_path: Path, data: Path, imgsz: int, conf: float, device: str | None):
    model = YOLO(str(model_path))
    kwargs = {
        "data": str(data),
        "imgsz": imgsz,
        "conf": conf,
        "split": "val",
        "verbose": False,
    }
    if device:
        kwargs["device"] = device

    metrics = model.val(**kwargs)
    return {
        "model": str(model_path),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data config not found: {args.data}")

    rows = []
    for m in args.models:
        if not m.exists():
            print(f"SKIP, model not found: {m}")
            continue
        print(f"Evaluating: {m}")
        rows.append(eval_one(m, args.data, args.imgsz, args.conf, args.device))

    if not rows:
        raise SystemExit("No valid models evaluated.")

    print("\nComparison:")
    header = f"{'model':60} {'mAP50':>8} {'mAP50-95':>10} {'precision':>10} {'recall':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        name = r["model"][-60:]
        print(f"{name:60} {r['mAP50']:8.4f} {r['mAP50-95']:10.4f} {r['precision']:10.4f} {r['recall']:8.4f}")

    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        print("\nDelta last - first:")
        for key in ("mAP50", "mAP50-95", "precision", "recall"):
            print(f"{key}: {b[key] - a[key]:+.4f}")


if __name__ == "__main__":
    main()
