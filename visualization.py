"""Optional overlay for context-aware LaneNet driving."""
import cv2
import numpy as np
from class_labels import CLASS_NAMES


def render(output):
    im = output.frame_bgr.copy()
    colors = {1: (0, 0, 255), 2: (255, 255, 255), 3: (0, 255, 255)}
    color_mask = np.zeros_like(im)
    for cls_id, bgr in colors.items():
        color_mask[output.segmentation == cls_id] = bgr
    is_lane = output.segmentation != 0
    blended = cv2.addWeighted(im, 0.7, color_mask, 0.3, 0)
    im[is_lane] = blended[is_lane]
    for x1, y1, x2, y2, cls_id, confidence in output.detections:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        label = f"{CLASS_NAMES.get(int(cls_id), str(int(cls_id)))} {confidence:.2f}"
        cv2.putText(im, label, (int(x1), max(15, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 0), 1)
    err_text = "unavailable" if output.error_cm is None else f"{output.error_cm:+.2f} cm"
    cv2.putText(im, f"lane error={err_text} ({output.mode})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(im, f"v={output.velocity:.2f} m/s omega={output.omega:+.2f} rad/s",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return im
