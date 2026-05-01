import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
YOLO_CONFIG_DIR = PROJECT_ROOT / ".cache" / "ultralytics"

MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_DIR)

import cpuinfo
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import torch_utils as ultralytics_torch_utils

from app.schemas.annotate import AnnotateResponse
from app.schemas.predict import BoundingBox, DetectionItem, PredictResponse
from app.utils.image_utils import encode_image_to_base64

ORIGINAL_TORCH_LOAD = torch.load


def patch_torch_load_for_trusted_local_checkpoint() -> None:
    if getattr(torch.load, "_cloudeco_patch_applied", False):
        return

    def trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return ORIGINAL_TORCH_LOAD(*args, **kwargs)

    trusted_torch_load._cloudeco_patch_applied = True
    torch.load = trusted_torch_load


def patch_ultralytics_cpu_info() -> None:
    def safe_get_cpu_info() -> str:
        info = cpuinfo.get_cpu_info()
        return (
            info.get("brand_raw")
            or info.get("brand")
            or info.get("arch_string_raw")
            or "CPU"
        )

    ultralytics_torch_utils.get_cpu_info = safe_get_cpu_info


class YoloService:
    def __init__(
        self,
        model_path: str = "fire-models/fire_m.pt",
        device: str = "cpu",
    ) -> None:
        self.model_path = PROJECT_ROOT / model_path
        self.device = device
        self._model: YOLO | None = None

    def load_model(self) -> YOLO:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            patch_torch_load_for_trusted_local_checkpoint()
            patch_ultralytics_cpu_info()
            self._model = YOLO(str(self.model_path))
        return self._model

    def get_class_names(self) -> dict[int, str]:
        model = self.load_model()

        if isinstance(model.names, dict):
            return {int(class_id): name for class_id, name in model.names.items()}

        return {index: name for index, name in enumerate(model.names)}

    def predict(
        self,
        image: Any,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[Any]:
        model = self.load_model()
        return model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
        )

    def predict_from_pil_image(
        self,
        request_uuid: UUID,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> PredictResponse:
        results = self.predict(
            image=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        return self.build_predict_response(
            request_uuid=request_uuid,
            result=results[0],
        )

    def build_predict_response(
        self,
        request_uuid: UUID,
        result: Any,
    ) -> PredictResponse:
        detections: list[DetectionItem] = []
        class_names = self.get_class_names()

        if result.boxes is not None:
            boxes_xywh = result.boxes.xywh.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for box_xywh, class_id, confidence in zip(
                boxes_xywh,
                class_ids,
                confidences,
            ):
                center_x, center_y, width, height = box_xywh
                top_left_x = center_x - (width / 2)
                top_left_y = center_y - (height / 2)

                detections.append(
                    DetectionItem(
                        class_name=class_names[int(class_id)],
                        box=BoundingBox(
                            x=round(top_left_x, 2),
                            y=round(top_left_y, 2),
                            width=round(width, 2),
                            height=round(height, 2),
                            probability=round(confidence, 4),
                        ),
                    )
                )

        return PredictResponse(
            uuid=request_uuid,
            count=len(detections),
            detections=detections,
            speed_preprocess_ms=round(result.speed.get("preprocess", 0.0), 3),
            speed_inference_ms=round(result.speed.get("inference", 0.0), 3),
            speed_postprocess_ms=round(result.speed.get("postprocess", 0.0), 3),
        )

    def annotate_from_pil_image(
        self,
        request_uuid: UUID,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> AnnotateResponse:
        results = self.predict(
            image=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        return self.build_annotate_response(
            request_uuid=request_uuid,
            result=results[0],
        )

    def build_annotate_response(
        self,
        request_uuid: UUID,
        result: Any,
    ) -> AnnotateResponse:
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_image = Image.fromarray(annotated_rgb)

        return AnnotateResponse(
            uuid=request_uuid,
            image=encode_image_to_base64(annotated_image),
        )


@lru_cache(maxsize=1)
def get_yolo_service() -> YoloService:
    service = YoloService()
    service.load_model()
    return service
