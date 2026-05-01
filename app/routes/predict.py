from fastapi import APIRouter, Depends

from app.routes.helpers import decode_request_image_or_400
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.yolo_service import YoloService, get_yolo_service

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict_image(
    payload: PredictRequest,
    yolo_service: YoloService = Depends(get_yolo_service),
) -> PredictResponse:
    decoded_image = decode_request_image_or_400(payload.image)
    return yolo_service.predict_from_pil_image(
        request_uuid=payload.uuid,
        image=decoded_image,
    )
