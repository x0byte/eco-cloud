# CloudEco Wildfire and Smoke Detection Service

This project is a cloud-deployed machine learning inference service for **wildfire and smoke detection** using a pretrained YOLO model. It was built for **FIT5225 Assignment 1**.

The system provides two main FastAPI endpoints:

- `POST /api/predict` — returns structured detection results in JSON
- `POST /api/annotate` — returns the input image with detections drawn on it as a base64-encoded image

The application was containerised with Docker and deployed to a Kubernetes cluster on Oracle Cloud Infrastructure (OCI).

---

## Project Structure

```text
app/                FastAPI application code
k8s/                Kubernetes YAML files
demo-images/        Sample images for testing
fire-models/        Pretrained wildfire model weights
tests/              Minimal API tests
Dockerfile          Container build file
requirements.txt    Python dependencies
locustfile.py       Locust benchmark script
README.md           Project instructions
```

---

## Requirements

### Local development

- Python 3.11
- pip
- virtual environment support

### Container / deployment

- Docker
- Docker Hub account
- Kubernetes cluster
- kubectl
- Oracle Cloud Infrastructure VMs

---

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI app locally:

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

---

## Local Testing

Run the minimal tests:

```bash
python3 -m pytest tests/test_api.py
```

---

## Docker

### Build the image locally

```bash
docker build -t cloudeco-wildfire .
```

### Run locally with Docker

```bash
docker run -p 8000:8000 cloudeco-wildfire
```

Then test:

```bash
curl http://127.0.0.1:8000/health
```

---

## Docker Hub Image

The deployed image used for Kubernetes is:

```text
hirunwe/cloudeco-wildfire:latest
```

---

## Kubernetes Deployment

The Kubernetes manifests are stored in the `k8s/` folder.

### Apply manifests

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Check deployment

```bash
kubectl get nodes
kubectl get deployments -n cloudeco
kubectl get pods -n cloudeco
kubectl get services -n cloudeco
```

### Scale replicas

```bash
kubectl scale deployment cloudeco-api --replicas=2 -n cloudeco
kubectl scale deployment cloudeco-api --replicas=4 -n cloudeco
kubectl scale deployment cloudeco-api --replicas=8 -n cloudeco
```

> If your deployment file is still named `deployement.yaml`, either rename it to `deployment.yaml` or replace the filename in the commands above.

---

## Public Deployment URL

The deployed application is accessible at:

```text
http://ipaddr:31142
```

### Public endpoints

- Health: `http://ipaddr:31142/health`
- Predict: `http://ipaddr:31142/api/predict`
- Annotate: `http://ipaddr:31142/api/annotate`

---

## Example Predict Request

Example Python client:

```python
import base64
import json
from pathlib import Path
from urllib import request

image_b64 = base64.b64encode(
    Path("demo-images/image3.jpeg").read_bytes()
).decode("utf-8")

payload = json.dumps({
    "uuid": "123e4567-e89b-12d3-a456-426614174000",
    "image": image_b64
}).encode("utf-8")

req = request.Request(
    "http://ipaddr/api/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with request.urlopen(req) as response:
    print(response.status)
    print(response.read().decode("utf-8"))
```

---


## Notes

- The pretrained model weights are expected in:
  - `fire-models/fire_m.pt`
- The service uses FastAPI with health probes suitable for Kubernetes deployment.
- The deployment was tested with 1, 2, 4, and 8 pod configurations.
- Internal node firewall rules had to be adjusted on the OCI VMs to allow Kubernetes node communication and NodePort traffic.

---

## Submission Notes

This repository contains the implementation artefacts required for the assignment, including:

- Dockerfile
- FastAPI source code
- Kubernetes YAML files
- Locust script
- README with run instructions and deployed URL

---

## Author

**Hirun Weththewa**  
