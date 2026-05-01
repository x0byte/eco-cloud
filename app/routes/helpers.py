from fastapi import HTTPException, status
from PIL import Image

from app.utils.image_utils import InvalidImageError, decode_base64_image


def decode_request_image_or_400(image_base64: str) -> Image.Image:
    try:
        return decode_base64_image(image_base64)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
