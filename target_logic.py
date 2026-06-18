from dataclasses import dataclass
import math


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


@dataclass
class TargetChoice:
    detection: Detection | None
    aim_x: int | None
    aim_y: int | None
    dx: int | None
    dy: int | None
    score: float


def aim_point(det: Detection, mode: str = "upper_body"):
    x = int((det.x1 + det.x2) / 2)
    if mode == "center":
        y = int((det.y1 + det.y2) / 2)
    else:
        y = int(det.y1 + 0.38 * (det.y2 - det.y1))
    return x, y


def choose_target(dets: list[Detection], frame_w: int, frame_h: int, mode: str = "upper_body") -> TargetChoice:
    if not dets:
        return TargetChoice(None, None, None, None, None, 0.0)

    cx = frame_w / 2
    cy = frame_h / 2
    diag = math.hypot(frame_w, frame_h)

    best = None
    best_score = -1e9
    best_aim = None

    for det in dets:
        ax, ay = aim_point(det, mode=mode)
        dist = math.hypot(ax - cx, ay - cy) / diag
        area = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
        area_norm = min(1.0, area / (frame_w * frame_h * 0.20))
        score = 1.20 * det.conf + 0.35 * area_norm - 0.90 * dist

        if score > best_score:
            best_score = score
            best = det
            best_aim = (ax, ay)

    ax, ay = best_aim
    return TargetChoice(best, ax, ay, int(ax - cx), int(ay - cy), best_score)


def detections_from_yolo_result(result, min_conf: float = 0.25) -> list[Detection]:
    dets = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dets

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    for box, conf in zip(xyxy, confs):
        if float(conf) < min_conf:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        dets.append(Detection(x1, y1, x2, y2, float(conf)))

    return dets
