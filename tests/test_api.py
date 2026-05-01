import base64
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

SAMPLE_UUID = "123e4567-e89b-12d3-a456-426614174000"
SAMPLE_IMAGE_PATH = Path(__file__).resolve().parents[1] / "demo-images" / "image3.jpeg"


def build_sample_image_base64() -> str:
    return base64.b64encode(SAMPLE_IMAGE_PATH.read_bytes()).decode("utf-8")


def test_health_returns_200() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_valid_sample_image_returns_200() -> None:
    client = TestClient(app)
    payload = {
        "uuid": SAMPLE_UUID,
        "image": build_sample_image_base64(),
    }

    response = client.post("/api/predict", json=payload)
    response_body = response.json()

    assert response.status_code == 200
    assert response_body["uuid"] == SAMPLE_UUID
    assert response_body["count"] >= 0
    assert isinstance(response_body["detections"], list)
    assert "speed_preprocess_ms" in response_body
    assert "speed_inference_ms" in response_body
    assert "speed_postprocess_ms" in response_body


def test_predict_with_invalid_base64_returns_400() -> None:
    client = TestClient(app)
    payload = {
        "uuid": SAMPLE_UUID,
        "image": "not-base64",
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 400
    assert response.json() == {"detail": "The image field is not valid base64 data."}


def test_annotate_with_valid_sample_image_returns_200() -> None:
    client = TestClient(app)
    payload = {
        "uuid": SAMPLE_UUID,
        "image": build_sample_image_base64(),
    }

    response = client.post("/api/annotate", json=payload)
    response_body = response.json()

    assert response.status_code == 200
    assert response_body["uuid"] == SAMPLE_UUID
    assert isinstance(response_body["image"], str)
    assert len(response_body["image"]) > 0


def test_annotate_with_invalid_base64_returns_400() -> None:
    client = TestClient(app)
    payload = {
        "uuid": SAMPLE_UUID,
        "image": "not-base64",
    }

    response = client.post("/api/annotate", json=payload)

    assert response.status_code == 400
    assert response.json() == {"detail": "The image field is not valid base64 data."}
