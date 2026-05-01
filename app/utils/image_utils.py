import base64
import binascii
import io

from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when the supplied base64 image cannot be decoded safely."""


def strip_data_url_prefix(image_base64: str) -> str:
    if "," in image_base64 and image_base64.lower().startswith("data:"):
        return image_base64.split(",", maxsplit=1)[1]

    return image_base64


def decode_base64_image(image_base64: str) -> Image.Image:
    normalized_image = strip_data_url_prefix(image_base64.strip())

    try:
        image_bytes = base64.b64decode(normalized_image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("The image field is not valid base64 data.") from exc

    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidImageError("The decoded data is not a supported image.") from exc


def encode_image_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    image_buffer = io.BytesIO()
    image.save(image_buffer, format=image_format)
    return base64.b64encode(image_buffer.getvalue()).decode("utf-8")
