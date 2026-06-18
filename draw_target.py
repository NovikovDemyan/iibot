import cv2


def draw_detections(frame, dets, choice):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1)

    selected = choice.detection if choice else None

    for det in dets:
        thickness = 3 if det is selected else 1
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (255, 255, 255), thickness)
        cv2.putText(frame, f"enemy {det.conf:.2f}", (det.x1, max(20, det.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    if selected is not None:
        ax, ay = choice.aim_x, choice.aim_y
        cv2.circle(frame, (ax, ay), 8, (255, 255, 255), 2)
        cv2.line(frame, (cx, cy), (ax, ay), (255, 255, 255), 2)
        cv2.putText(frame, f"selected | dx={choice.dx} dy={choice.dy} score={choice.score:.2f}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "no target", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return frame
