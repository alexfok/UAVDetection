from __future__ import annotations

import logging
import sys
from typing import Any

LOGGER = logging.getLogger(__name__)
_TORCHVISION_LIBRARY: Any | None = None
_INSTALLED = False


def install_torchvision_nms_fallback() -> None:
    """Patch torchvision NMS when Jetson wheels lack the compiled operator."""
    global _INSTALLED, _TORCHVISION_LIBRARY
    if _INSTALLED:
        return
    _INSTALLED = True

    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive import guard
        LOGGER.debug("Torch import failed; skipping torchvision NMS fallback: %s", exc)
        return

    try:
        import torchvision

        empty_boxes = torch.empty((0, 4))
        empty_scores = torch.empty((0,))
        torchvision.ops.nms(empty_boxes, empty_scores, 0.45)
        return
    except Exception as exc:
        native_error = exc

    for module_name in list(sys.modules):
        if module_name == "torchvision" or module_name.startswith("torchvision."):
            sys.modules.pop(module_name, None)

    try:
        _TORCHVISION_LIBRARY = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_LIBRARY.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        # The operator can already exist when torchvision was built with custom ops.
        pass

    try:
        import torchvision
    except Exception as exc:
        LOGGER.warning("Using pure PyTorch torchvision.ops.nms fallback: %s", native_error)
    else:
        try:
            empty_boxes = torch.empty((0, 4))
            empty_scores = torch.empty((0,))
            torchvision.ops.nms(empty_boxes, empty_scores, 0.45)
            return
        except Exception as exc:
            LOGGER.warning("Using pure PyTorch torchvision.ops.nms fallback: %s", exc)

    def nms(boxes: Any, scores: Any, iou_threshold: float) -> Any:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-9)
            order = rest[iou <= iou_threshold]

        if not keep:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        return torch.stack(keep).to(dtype=torch.long)

    torchvision.ops.nms = nms
