from fastapi import APIRouter, Depends

from app.routes.helpers import decode_request_image_or_400
from app.schemas.annotate import AnnotateRequest, AnnotateResponse
from app.services.yolo_service import YoloService, get_yolo_service

router = APIRouter(prefix="/api", tags=["Annotation"])


@router.post("/annotate", response_model=AnnotateResponse)
def annotate_image(
    payload: AnnotateRequest,
    yolo_service: YoloService = Depends(get_yolo_service),
) -> AnnotateResponse:
    decoded_image = decode_request_image_or_400(payload.image)
    return yolo_service.annotate_from_pil_image(
        request_uuid=payload.uuid,
        image=decoded_image,
    )
