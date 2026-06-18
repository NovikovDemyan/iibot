import cv2


def draw_preview_overlay(frame, fps: float, backend: str, recording: bool = False, saved: int = 0):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1)

    status = "REC AUTO" if recording else "preview"
    text = f"{status} | backend={backend} | fps={fps:.1f} | saved={saved} | size={w}x{h}"

    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "q: quit | close window: quit | Ctrl+C in terminal: stop",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
