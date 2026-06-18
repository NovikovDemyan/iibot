import cv2

from capture import ScreenCapture
from config import CAPTURE, DATASET
from overlay import draw_preview_overlay
from storage import save_frame


WINDOW_NAME = "CS2 Vision Agent SAFE v3: preview only"


def window_is_open(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def main():
    cap = ScreenCapture(region=CAPTURE.region, prefer_dxcam=CAPTURE.prefer_dxcam)
    cap.start()

    print(f"Capture backend: {cap.backend}")
    print("Preview only. No keyboard/mouse input is sent.")
    print("Exit: q in preview window, close window, or Ctrl+C in terminal.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    saved = 0

    try:
        while True:
            pkt = cap.read()
            raw = pkt.frame_bgr
            frame = raw.copy()

            if CAPTURE.preview_scale != 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=CAPTURE.preview_scale,
                    fy=CAPTURE.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )

            draw_preview_overlay(frame, pkt.fps, cap.backend, recording=False, saved=saved)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                path = save_frame(raw, DATASET.captures_dir, "capture", DATASET.jpeg_quality)
                saved += 1
                print(f"Saved: {path}")

            if not window_is_open(WINDOW_NAME):
                break

    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")

    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
