import base64
import json
from pathlib import Path

from locust import HttpUser, task, between


IMAGE_PATH = Path("demo-images/image3.jpeg")
IMAGE_B64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")


class WildfireUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        payload = {
            "uuid": "123e4567-e89b-12d3-a456-426614174000",
            "image": IMAGE_B64,
        }

        with self.client.post(
            "/api/predict",
            json=payload,
            name="/api/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status: {response.status_code}")
                return

            try:
                data = response.json()
            except json.JSONDecodeError:
                response.failure("Response was not valid JSON")
                return

            if "count" not in data or "detections" not in data:
                response.failure("Missing expected fields in response")
                return

            response.success()