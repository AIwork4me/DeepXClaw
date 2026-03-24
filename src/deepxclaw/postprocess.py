"""YOLOv26n post-processing: decode NPU output -> NMS -> bounding boxes.

NPU output tensor shape: [1, 300, 6]
  Each row: [cx, cy, w, h, confidence, class_id]
  cx, cy: center coordinates (in input pixel space)
  w, h: box width and height
  confidence: detection confidence
  class_id: COCO class index (float, cast to int)
"""

from __future__ import annotations

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """Non-Maximum Suppression. Returns indices of kept boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep


def decode_yolo26n(
    output: np.ndarray,
    input_size: tuple[int, int] = (640, 640),
    score_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict]:
    """Decode YOLOv26n NPU output into detection results.

    Args:
        output: NPU output tensor, shape [1, 300, 6] or [300, 6]
        input_size: Model input (H, W) = (640, 640)
        score_threshold: Min confidence to keep
        iou_threshold: NMS IoU threshold

    Returns:
        List of dicts: x1, y1, x2, y2, score, class_id, label
    """
    from .labels import COCO_LABELS

    if output is None or output.size == 0:
        return []

    # Squeeze to [300, 6]
    if output.ndim == 3:
        output = output.squeeze(0)

    # Filter by confidence
    conf = output[:, 4]
    mask = conf >= score_threshold
    if not mask.any():
        return []

    detections = output[mask]  # [N, 6]
    cx = detections[:, 0]
    cy = detections[:, 1]
    w = detections[:, 2]
    h = detections[:, 3]
    scores = detections[:, 4]
    class_ids = detections[:, 5].astype(int)

    # Convert cx,cy,w,h -> x1,y1,x2,y2
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Clip to input size
    x1 = np.clip(x1, 0, input_size[1])
    y1 = np.clip(y1, 0, input_size[0])
    x2 = np.clip(x2, 0, input_size[1])
    y2 = np.clip(y2, 0, input_size[0])

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Per-class NMS
    keep = []
    unique_classes = np.unique(class_ids)
    for cid in unique_classes:
        cmask = class_ids == cid
        if cmask.sum() == 0:
            continue
        indices = nms(boxes[cmask], scores[cmask], iou_threshold)
        for idx in indices:
            keep.append(np.where(cmask)[0][idx])

    results = []
    for i in keep:
        cls_id = int(class_ids[i])
        label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else f"class_{cls_id}"
        results.append({
            "x1": float(x1[i]),
            "y1": float(y1[i]),
            "x2": float(x2[i]),
            "y2": float(y2[i]),
            "score": float(scores[i]),
            "class_id": cls_id,
            "label": label,
        })

    return results