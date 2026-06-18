import argparse
import time
from pathlib import Path

import cv2
import mss
import numpy as np
from ultralytics import YOLO


WINDOW_NAME = "CS2 Enemy Detector Preview — no input, no control"


def grab_screen(sct, monitor):
    raw = np.array(sct.grab(monitor))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


def window_is_open(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--preview-scale", type=float, default=0.75)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    model = YOLO(str(args.model))

    sct = mss.mss()
    monitor = sct.monitors[1]

    print("Live detector preview started.")
    print("No keyboard/mouse input is sent.")
    print("Exit: q in window, close window, or Ctrl+C in terminal.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_t = time.perf_counter()
    fps = 0.0

    try:
        while True:
            frame = grab_screen(sct, monitor)

            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )

            annotated = results[0].plot()

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps

            if args.preview_scale != 1.0:
                annotated = cv2.resize(
                    annotated,
                    None,
                    fx=args.preview_scale,
                    fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.putText(
                annotated,
                f"detector preview | fps={fps:.1f} | conf={args.conf}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if not window_is_open(WINDOW_NAME):
                break

    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")

    finally:
        sct.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
