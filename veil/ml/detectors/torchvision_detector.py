"""Open-source torchvision detectors (COCO-pretrained).

Only models that ship with permissive licences and public weights are
registered. Weights are downloaded once into the torch hub cache; VEIL never
loads a user-supplied checkpoint (see docs/security.md).
"""

from __future__ import annotations

import torch

from veil.ml.detectors.base import Detection, Detector, DetectorInfo

# torchvision's COCO category list, index-aligned with model outputs.
COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

_MODELS = {
    "fasterrcnn-mobilenet-320": (
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
        "Fast, small Faster R-CNN. Default digital detector.",
    ),
    "fasterrcnn-resnet50": (
        "fasterrcnn_resnet50_fpn",
        "FasterRCNN_ResNet50_FPN_Weights",
        "Heavier, more accurate Faster R-CNN.",
    ),
    "retinanet-resnet50": (
        "retinanet_resnet50_fpn",
        "RetinaNet_ResNet50_FPN_Weights",
        "Single-stage detector; different failure modes to Faster R-CNN.",
    ),
}


class TorchvisionDetector(Detector):
    """Wraps a torchvision detection model behind the VEIL interface.

    Gradients: torchvision detection heads are differentiable, but box
    selection (NMS/top-k) is not. `score` therefore differentiates through
    the *confidences of the boxes that survived selection*, which is the
    standard formulation used by adversarial-patch work. The selection itself
    is treated as piecewise constant.
    """

    differentiable = True

    def __init__(self, model_id: str = "fasterrcnn-mobilenet-320", device: str | None = None):
        if model_id not in _MODELS:
            raise KeyError(f"unknown torchvision model: {model_id}")
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = None
        self._weights_name = ""

    def load(self) -> "TorchvisionDetector":
        if self._model is not None:
            return self
        import torchvision

        fn_name, weights_enum, _ = _MODELS[self.model_id]
        weights = getattr(torchvision.models.detection, weights_enum).DEFAULT
        self._weights_name = str(weights)
        model = getattr(torchvision.models.detection, fn_name)(weights=weights)
        self._model = model.eval().to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)  # we optimize the pattern, never the model
        return self

    def _forward(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        self.load()
        assert self._model is not None
        return self._model([img for img in images.to(self.device)])

    @torch.no_grad()
    def predict(self, images: torch.Tensor, threshold: float = 0.5) -> list[list[Detection]]:
        out = []
        for result in self._forward(images):
            dets = []
            for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
                s = float(score)
                if s < threshold:
                    continue
                idx = int(label)
                name = COCO_LABELS[idx] if idx < len(COCO_LABELS) else str(idx)
                dets.append(Detection(name, s, tuple(float(v) for v in box)))
            out.append(dets)
        return out

    def score(self, images: torch.Tensor, label: str) -> torch.Tensor:
        """Max confidence for `label` per image, differentiable w.r.t. input."""
        try:
            wanted = COCO_LABELS.index(label)
        except ValueError as exc:  # pragma: no cover - guarded by the API layer
            raise KeyError(f"{label!r} is not a COCO class") from exc
        scores = []
        for i, result in enumerate(self._forward(images)):
            match = result["scores"][result["labels"] == wanted]
            if match.numel() == 0:
                # Keep the graph connected so .backward() still works.
                scores.append(images[i].sum() * 0.0)
            else:
                scores.append(match.max())
        return torch.stack(scores)

    def metadata(self) -> DetectorInfo:
        _, _, notes = _MODELS[self.model_id]
        import torchvision

        return DetectorInfo(
            id=self.model_id,
            name=_MODELS[self.model_id][0],
            version=f"torchvision-{torchvision.__version__}:{self._weights_name or 'DEFAULT'}",
            labels=[l for l in COCO_LABELS if l not in ("N/A", "__background__")],
            differentiable=True,
            license="BSD-3-Clause (torchvision), COCO weights",
            source="https://pytorch.org/vision/stable/models.html",
            notes=notes,
        )
