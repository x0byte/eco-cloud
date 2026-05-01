from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uuid: UUID = Field(
        ...,
        description="Unique request identifier supplied by the client.",
    )
    image: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded image string.",
    )


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., description="Top-left x-coordinate in pixels.")
    y: float = Field(..., description="Top-left y-coordinate in pixels.")
    width: float = Field(..., description="Bounding box width in pixels.")
    height: float = Field(..., description="Bounding box height in pixels.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")


class DetectionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str = Field(..., description="Detected object class name.")
    box: BoundingBox


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uuid: UUID = Field(..., description="Echo of the request identifier.")
    count: int = Field(..., ge=0, description="Number of detections returned.")
    detections: list[DetectionItem] = Field(
        default_factory=list,
        description="Detected objects and their bounding boxes.",
    )
    speed_preprocess_ms: float = Field(
        ...,
        ge=0.0,
        description="Image preprocessing time in milliseconds.",
    )
    speed_inference_ms: float = Field(
        ...,
        ge=0.0,
        description="Model inference time in milliseconds.",
    )
    speed_postprocess_ms: float = Field(
        ...,
        ge=0.0,
        description="Postprocessing time in milliseconds.",
    )
