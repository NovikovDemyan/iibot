import cv2

from capture import ScreenCapture
from config import CAPTURE, DATASET
from overlay import draw_preview_overlay
from storage import save_frame


WINDOW_NAME = "CS2 Vision Agent SAFE v3: AUTO RECORD"


def window_is_open(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def main():
    cap = ScreenCapture(region=CAPTURE.region, prefer_dxcam=CAPTURE.prefer_dxcam)
    cap.start()

    print(f"Capture backend: {cap.backend}")
    print("AUTO RECORD started immediately.")
    print("No keyboard/mouse input is sent.")
    print("Stop: Ctrl+C in terminal, close preview window, or press q in preview window.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx = 0
    saved = 0

    try:
        while True:
            pkt = cap.read()
            raw = pkt.frame_bgr
            frame_idx += 1

            if frame_idx % DATASET.save_every_n_frames == 0:
                path = save_frame(raw, DATASET.raw_dir, "raw", DATASET.jpeg_quality)
                saved += 1

                if saved % 25 == 0:
                    print(f"Saved frames: {saved}. Last: {path}")

            preview = raw.copy()
            if CAPTURE.preview_scale != 1.0:
                preview = cv2.resize(
                    preview,
                    None,
                    fx=CAPTURE.preview_scale,
                    fy=CAPTURE.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )

            draw_preview_overlay(preview, pkt.fps, cap.backend, recording=True, saved=saved)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if not window_is_open(WINDOW_NAME):
                break

    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")

    finally:
        cap.stop()
        cv2.destroyAllWindows()

    print(f"Finished. Total saved frames: {saved}")
    print(f"Folder: {DATASET.raw_dir}")


if __name__ == "__main__":
    main()
