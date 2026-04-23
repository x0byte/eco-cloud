import os
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
YOLO_CONFIG_DIR = PROJECT_ROOT / ".cache" / "ultralytics"

MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

import cv2
from PIL import Image
import cpuinfo
import torch
from ultralytics import YOLO
from ultralytics.utils import torch_utils as ultralytics_torch_utils

ORIGINAL_TORCH_LOAD = torch.load


def patch_torch_load_for_trusted_local_model() -> None:
    def trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return ORIGINAL_TORCH_LOAD(*args, **kwargs)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the wildfire YOLO model on one local image."
    )
    parser.add_argument(
        "--model",
        default="fire-models/fire_m.pt",
        help="Path to the YOLO model weights file.",
    )
    parser.add_argument(
        "--image",
        default="demo-images/image3.jpeg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/annotated-image3.jpg",
        help="Path to save the annotated output image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for non-max suppression.",
    )
    return parser


def to_detection_payload(result, class_names: dict[int, str]) -> dict:
    detections = []

    if result.boxes is not None:
        boxes_xywh = result.boxes.xywh.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for box_xywh, class_id, confidence in zip(boxes_xywh, class_ids, confidences):
            center_x, center_y, width, height = box_xywh
            top_left_x = center_x - (width / 2)
            top_left_y = center_y - (height / 2)

            detections.append(
                {
                    "class": class_names[int(class_id)],
                    "box": {
                        "x": round(top_left_x, 2),
                        "y": round(top_left_y, 2),
                        "width": round(width, 2),
                        "height": round(height, 2),
                        "probability": round(confidence, 4),
                    },
                }
            )

    return {
        "count": len(detections),
        "detections": detections,
        "speed_preprocess_ms": round(result.speed.get("preprocess", 0.0), 3),
        "speed_inference_ms": round(result.speed.get("inference", 0.0), 3),
        "speed_postprocess_ms": round(result.speed.get("postprocess", 0.0), 3),
    }


def save_annotated_image(result, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(annotated_rgb).save(output_path)


def main() -> None:
    args = build_parser().parse_args()
    patch_torch_load_for_trusted_local_model()
    patch_ultralytics_cpu_info()
    model = YOLO(args.model)
    results = model.predict(args.image, conf=args.conf, iou=args.iou, device="cpu")
    first_result = results[0]

    payload = to_detection_payload(first_result, model.names)
    save_annotated_image(first_result, Path(args.output))

    print("Model classes:")
    print(json.dumps(model.names, indent=2))
    print()
    print("Detection summary:")
    print(json.dumps(payload, indent=2))
    print()
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
