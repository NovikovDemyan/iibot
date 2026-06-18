import argparse
from pathlib import Path

import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_ID = 0
WINDOW_NAME = "Enemy Labeler — drag LMB, s save, d no enemy, u undo, c clear, q quit"


def list_images(path: Path):
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def yolo_line_from_box(box, w, h):
    x1, y1, x2, y2 = box

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def save_yolo_label(label_path: Path, boxes, w, h):
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for box in boxes:
        line = yolo_line_from_box(box, w, h)
        if line is not None:
            lines.append(line)

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_existing_label(label_path: Path, w, h):
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = parts
        if cls != str(CLASS_ID):
            continue

        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        boxes.append((x1, y1, x2, y2))

    return boxes


class LabelState:
    def __init__(self, scale):
        self.scale = scale
        self.boxes = []
        self.drawing = False
        self.start = None
        self.current = None

    def mouse_cb(self, event, x, y, flags, param):
        # Координаты окна переводим в координаты исходного изображения.
        ix = int(x / self.scale)
        iy = int(y / self.scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (ix, iy)
            self.current = (ix, iy)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current = (ix, iy)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            end = (ix, iy)
            x1, y1 = self.start
            x2, y2 = end

            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            if abs(x2 - x1) >= 5 and abs(y2 - y1) >= 5:
                self.boxes.append((x1, y1, x2, y2))

            self.start = None
            self.current = None


def draw_boxes(img, boxes, scale, current_box=None):
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def sc_box(box):
        x1, y1, x2, y2 = box
        return int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)

    for box in boxes:
        x1, y1, x2, y2 = sc_box(box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

    if current_box is not None:
        x1, y1, x2, y2 = sc_box(current_box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("dataset/labeling_frames"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--scale", type=float, default=0.75)
    args = parser.parse_args()

    images = list_images(args.images)
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    print(f"Images: {len(images)}")
    print(f"Labels folder: {args.labels}")
    print("Controls: drag LMB=box | s=save+next | d=no enemy+next | u=undo | c=clear | n=next | p=prev | q=quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    i = 0
    while 0 <= i < len(images):
        img_path = images[i]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read: {img_path}")
            i += 1
            continue

        h, w = img.shape[:2]
        label_path = args.labels / f"{img_path.stem}.txt"

        state = LabelState(scale=args.scale)
        state.boxes = load_existing_label(label_path, w, h)

        cv2.setMouseCallback(WINDOW_NAME, state.mouse_cb)

        while True:
            current_box = None
            if state.drawing and state.start and state.current:
                x1, y1 = state.start
                x2, y2 = state.current
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                current_box = (x1, y1, x2, y2)

            vis = draw_boxes(img, state.boxes, args.scale, current_box)

            status = f"{i + 1}/{len(images)} | boxes={len(state.boxes)} | {img_path.name}"
            cv2.putText(
                vis,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return

            if key == ord("u"):
                if state.boxes:
                    state.boxes.pop()

            elif key == ord("c"):
                state.boxes.clear()

            elif key == ord("s"):
                save_yolo_label(label_path, state.boxes, w, h)
                print(f"Saved: {label_path} boxes={len(state.boxes)}")
                i += 1
                break

            elif key == ord("d"):
                save_yolo_label(label_path, [], w, h)
                print(f"Saved negative: {label_path}")
                i += 1
                break

            elif key == ord("n"):
                i += 1
                break

            elif key == ord("p"):
                i = max(0, i - 1)
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    return
            except cv2.error:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
