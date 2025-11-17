# 🚀 راهنمای استقرار (Deployment Guide)

این راهنما نحوه استقرار سیستم تشخیص CAD در محیط‌های مختلف را توضیح می‌دهد.

## 📋 فهرست مطالب

1. [استقرار Local (Development)](#استقرار-local)
2. [استقرار سرور Linux](#استقرار-سرور-linux)
3. [استقرار Docker](#استقرار-docker)
4. [استقرار Cloud (AWS/Azure/GCP)](#استقرار-cloud)
5. [استقرار Web API](#استقرار-web-api)
6. [استقرار Mobile/Edge](#استقرار-mobileedge)
7. [Scaling و Load Balancing](#scaling-و-load-balancing)
8. [Monitoring و Logging](#monitoring-و-logging)

---

## استقرار Local

برای توسعه یا استفاده شخصی.

### Windows

```powershell
# 1. نصب Python 3.10+
# دانلود از: https://www.python.org/downloads/

# 2. کلون پروژه
git clone https://github.com/your-repo/cad3d.git
cd cad3d

# 3. محیط مجازی
python -m venv .venv
.venv\Scripts\activate

# 4. نصب dependencies
pip install -r requirements.txt
pip install -r requirements-neural.txt

# 5. نصب PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 6. تست
python -m cad3d.cli --help
```

### Linux/Mac

```bash
# 1. نصب Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv

# 2. کلون پروژه
git clone https://github.com/your-repo/cad3d.git
cd cad3d

# 3. محیط مجازی
python3.10 -m venv .venv
source .venv/bin/activate

# 4. نصب dependencies
pip install -r requirements.txt
pip install -r requirements-neural.txt

# 5. نصب PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 6. تست
python -m cad3d.cli --help
```

---

## استقرار سرور Linux

برای استقرار production روی سرور.

### 1. آماده‌سازی سرور

```bash
# Ubuntu 22.04 LTS
sudo apt update
sudo apt upgrade -y

# نصب dependencies سیستمی
sudo apt install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential \
    git \
    nginx \
    supervisor

# نصب CUDA (برای GPU)
# https://developer.nvidia.com/cuda-downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda
```

### 2. نصب پروژه

```bash
# ساخت user جداگانه
sudo useradd -m -s /bin/bash cad3d
sudo su - cad3d

# کلون پروژه
git clone https://github.com/your-repo/cad3d.git
cd cad3d

# محیط مجازی و نصب
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-neural.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. تنظیمات محیط

```bash
# ساخت .env file
cat > .env << EOF
# Application
DEBUG=False
LOG_LEVEL=INFO

# Paths
MODEL_PATH=/opt/cad3d/models/best_model.pth
UPLOAD_DIR=/opt/cad3d/uploads
OUTPUT_DIR=/opt/cad3d/outputs

# Performance
MAX_WORKERS=4
GPU_ENABLED=True
DEVICE=cuda

# ODA Converter (optional)
ODA_CONVERTER_PATH=/opt/ODA/ODAFileConverter

# MiDaS (optional)
MIDAS_ONNX_PATH=/opt/cad3d/models/midas_v2_small_256.onnx
EOF
```

### 4. تنظیم Supervisor

```bash
# ساخت config supervisor
sudo nano /etc/supervisor/conf.d/cad3d.conf
```

```ini
[program:cad3d-api]
command=/home/cad3d/cad3d/.venv/bin/python -m cad3d.api
directory=/home/cad3d/cad3d
user=cad3d
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/cad3d/api.log
environment=PYTHONUNBUFFERED=1

[program:cad3d-worker]
command=/home/cad3d/cad3d/.venv/bin/python -m cad3d.worker
directory=/home/cad3d/cad3d
user=cad3d
autostart=true
autorestart=true
numprocs=4
process_name=%(program_name)s_%(process_num)02d
redirect_stderr=true
stdout_logfile=/var/log/cad3d/worker.log
```

```bash
# فعال‌سازی
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start cad3d-api:*
sudo supervisorctl start cad3d-worker:*
```

### 5. تنظیم Nginx

```bash
sudo nano /etc/nginx/sites-available/cad3d
```

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout برای پردازش طولانی
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
    }
    
    location /static {
        alias /home/cad3d/cad3d/static;
    }
    
    location /outputs {
        alias /opt/cad3d/outputs;
        autoindex on;
    }
}
```

```bash
# فعال‌سازی
sudo ln -s /etc/nginx/sites-available/cad3d /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. SSL با Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## استقرار Docker

برای استقرار portable و قابل توسعه.

### 1. Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# نصب Python و dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# ساخت user
RUN useradd -m -u 1000 cad3d

# کپی پروژه
WORKDIR /app
COPY requirements.txt requirements-neural.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-neural.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

COPY . .
RUN chown -R cad3d:cad3d /app

# تنظیمات
USER cad3d
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda

EXPOSE 8000

# دستور اجرا
CMD ["python", "-m", "cad3d.api"]
```

### 2. docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
      - MAX_WORKERS=4
    volumes:
      - ./models:/app/models:ro
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  worker:
    build: .
    command: python -m cad3d.worker
    environment:
      - DEVICE=cuda
    volumes:
      - ./models:/app/models:ro
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./outputs:/usr/share/nginx/html/outputs:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
```

### 3. Build و Run

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# مشاهده logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## استقرار Cloud

### AWS EC2

**1. ساخت Instance**

```bash
# Launch EC2 instance
# - AMI: Ubuntu 22.04 LTS
# - Type: g4dn.xlarge (GPU) یا t3.2xlarge (CPU)
# - Storage: 50GB SSD
# - Security Group: HTTP (80), HTTPS (443), SSH (22)

# SSH به instance
ssh -i key.pem ubuntu@ec2-xxx.compute.amazonaws.com

# نصب (مانند سرور Linux)
...
```

**2. تنظیم Auto Scaling**

```bash
# AWS CLI
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name cad3d-asg \
  --launch-configuration-name cad3d-lc \
  --min-size 1 \
  --max-size 10 \
  --desired-capacity 2 \
  --vpc-zone-identifier "subnet-xxx,subnet-yyy"
```

**3. استفاده از S3 برای ذخیره‌سازی**

```python
import boto3

s3 = boto3.client('s3')

# آپلود نتیجه
s3.upload_file('output.dxf', 'cad3d-bucket', 'outputs/output.dxf')

# دانلود
s3.download_file('cad3d-bucket', 'models/best_model.pth', 'model.pth')
```

### Azure VM

```bash
# ساخت VM
az vm create \
  --resource-group cad3d-rg \
  --name cad3d-vm \
  --image Ubuntu2204 \
  --size Standard_NC6 \
  --admin-username azureuser \
  --generate-ssh-keys

# SSH
ssh azureuser@<public-ip>

# نصب (مانند سرور Linux)
...
```

### Google Cloud Platform

```bash
# ساخت VM
gcloud compute instances create cad3d-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# SSH
gcloud compute ssh cad3d-instance

# نصب (مانند سرور Linux)
...
```

---

## استقرار Web API

FastAPI backend برای استفاده در Web/Mobile apps.

### 1. ساخت API

```python
# cad3d/api.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import os

app = FastAPI(title="CAD 3D Converter API")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/convert/dxf-extrude")
async def convert_dxf_extrude(
    file: UploadFile = File(...),
    height: float = 3000,
    optimize: bool = True
):
    """تبدیل DXF دوبعدی به سه‌بعدی"""
    # ذخیره فایل
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}.dxf"
    output_path = OUTPUT_DIR / f"{file_id}_3d.dxf"
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # تبدیل
    from cad3d.dxf_extrude import extrude_dxf_closed_polylines
    extrude_dxf_closed_polylines(
        str(input_path),
        str(output_path),
        height=height,
        optimize=optimize
    )
    
    # حذف فایل ورودی
    input_path.unlink()
    
    return {
        "file_id": file_id,
        "download_url": f"/download/{file_id}_3d.dxf"
    }

@app.post("/convert/pdf-to-dxf")
async def convert_pdf_to_dxf(
    file: UploadFile = File(...),
    dpi: int = 300,
    confidence: float = 0.5,
    device: str = "auto"
):
    """تبدیل PDF به DXF با Neural Network"""
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}.pdf"
    output_path = OUTPUT_DIR / f"{file_id}.dxf"
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # تبدیل
    from cad3d.neural_cad_detector import NeuralCADDetector
    from cad3d.pdf_processor import PDFToImageConverter, CADPipeline
    
    detector = NeuralCADDetector(device=device)
    pdf_converter = PDFToImageConverter(dpi=dpi)
    pipeline = CADPipeline(detector, pdf_converter)
    
    pipeline.process_pdf_to_dxf(
        str(input_path),
        str(output_path),
        confidence_threshold=confidence
    )
    
    input_path.unlink()
    
    return {
        "file_id": file_id,
        "download_url": f"/download/{file_id}.dxf"
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    """دانلود فایل خروجی"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return {"error": "File not found"}
    
    return FileResponse(file_path, filename=filename)

@app.get("/status")
async def status():
    """وضعیت سرور"""
    import torch
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. اجرا با Uvicorn

```bash
# Development
uvicorn cad3d.api:app --reload

# Production
uvicorn cad3d.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. تست API

```bash
# تست status
curl http://localhost:8000/status

# آپلود و تبدیل DXF
curl -X POST http://localhost:8000/convert/dxf-extrude \
  -F "file=@plan.dxf" \
  -F "height=3000" \
  -o output_3d.dxf

# آپلود و تبدیل PDF
curl -X POST http://localhost:8000/convert/pdf-to-dxf \
  -F "file=@plan.pdf" \
  -F "dpi=300" \
  -o output.dxf
```

---

## استقرار Mobile/Edge

برای دستگاه‌های محدود (موبایل، Raspberry Pi، ...).

### 1. بهینه‌سازی مدل

```bash
# Quantization برای کاهش حجم
python -m cad3d.cli optimize-model \
  --model best_model.pth \
  --output-dir ./mobile_model \
  --formats quantized \
  --benchmark
```

### 2. استقرار Android (TensorFlow Lite)

```bash
# تبدیل PyTorch → ONNX → TFLite
python convert_to_tflite.py --input model.pth --output model.tflite
```

```java
// Android
Interpreter tflite = new Interpreter(loadModelFile());
float[][] input = preprocessImage(bitmap);
float[][] output = new float[1][NUM_CLASSES];
tflite.run(input, output);
```

### 3. استقرار iOS (Core ML)

```bash
# تبدیل PyTorch → ONNX → Core ML
pip install coremltools
python convert_to_coreml.py --input model.pth --output model.mlmodel
```

```swift
// iOS
let model = try! CADDetector(configuration: MLModelConfiguration())
let prediction = try! model.prediction(image: pixelBuffer)
```

### 4. Raspberry Pi

```bash
# نصب PyTorch Lite
pip install https://download.pytorch.org/whl/cpu/torch-1.13.0%2Bcpu-cp39-cp39-linux_aarch64.whl

# استفاده از مدل Quantized
python inference.py --model quantized_model.pth --device cpu
```

---

## Scaling و Load Balancing

### 1. Horizontal Scaling با Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cad3d-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cad3d-api
  template:
    metadata:
      labels:
        app: cad3d-api
    spec:
      containers:
      - name: api
        image: cad3d:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: cad3d-service
spec:
  type: LoadBalancer
  selector:
    app: cad3d-api
  ports:
  - port: 80
    targetPort: 8000
```

### 2. Message Queue (Celery + Redis)

```python
# cad3d/tasks.py
from celery import Celery

app = Celery('cad3d', broker='redis://localhost:6379/0')

@app.task
def convert_dxf_task(input_path, output_path, height):
    from cad3d.dxf_extrude import extrude_dxf_closed_polylines
    extrude_dxf_closed_polylines(input_path, output_path, height=height)
    return output_path

# استفاده
result = convert_dxf_task.delay('input.dxf', 'output.dxf', 3000)
```

### 3. Load Balancing با Nginx

```nginx
upstream cad3d_backend {
    least_conn;
    server 192.168.1.10:8000 weight=3;
    server 192.168.1.11:8000 weight=2;
    server 192.168.1.12:8000 weight=1;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://cad3d_backend;
        proxy_next_upstream error timeout http_502 http_503 http_504;
    }
}
```

---

## Monitoring و Logging

### 1. Logging

```python
# cad3d/logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/var/log/cad3d/app.log')
        ]
    )
```

### 2. Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
requests_total = Counter('cad3d_requests_total', 'Total requests')
request_duration = Histogram('cad3d_request_duration_seconds', 'Request duration')

@request_duration.time()
def process_request():
    requests_total.inc()
    # ... پردازش

# شروع metrics server
start_http_server(9090)
```

### 3. Health Checks

```python
@app.get("/health")
async def health_check():
    import torch
    import psutil
    
    return {
        "status": "healthy",
        "gpu": {
            "available": torch.cuda.is_available(),
            "memory_used": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        },
        "cpu": {
            "percent": psutil.cpu_percent(),
            "count": psutil.cpu_count()
        },
        "memory": {
            "percent": psutil.virtual_memory().percent,
            "available_gb": psutil.virtual_memory().available / 1e9
        }
    }
```

### 4. Error Tracking (Sentry)

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="https://xxx@sentry.io/xxx",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0
)
```

---

## 🔒 امنیت

### 1. Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/convert/dxf-extrude")
async def convert_dxf(
    file: UploadFile,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # بررسی token
    if credentials.credentials != "secret-token":
        raise HTTPException(401, "Invalid token")
    # ...
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/convert/pdf-to-dxf")
@limiter.limit("5/minute")
async def convert_pdf(request: Request, file: UploadFile):
    # ...
```

### 3. File Validation

```python
ALLOWED_EXTENSIONS = {'.dxf', '.pdf', '.jpg', '.png'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def validate_file(file: UploadFile):
    # بررسی پسوند
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file type")
    
    # بررسی حجم
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
```

---

**✨ استقرار موفق!**
