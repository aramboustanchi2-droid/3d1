# 🚀 راهنمای سریع: از دمو تا Production

**هدف**: تبدیل سیستم CAD3D از حالت دمو به production در کمترین زمان

---

## 📊 وضعیت فعلی

✅ **آماده است**:

- قابلیت‌های اصلی (تبدیل CAD)
- رابط کاربری زیبا
- مستندات خوب

❌ **آماده نیست**:

- امنیت (Authentication, Rate limiting)
- Deployment production
- Monitoring و logging حرفه‌ای
- Backup strategy

---

## 🎯 سناریو پیشنهادی: شروع سریع (5 روز)

### روز 1: امنیت پایه ⚡

#### گام 1: نصب پکیج‌های امنیتی

```powershell
pip install slowapi python-jose[cryptography] passlib[bcrypt] python-multipart
```

#### گام 2: ایجاد فایل امنیتی

ایجاد `cad3d/security.py`:

```python
from fastapi import HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE-THIS-IN-PRODUCTION-min-32-chars")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Simple user DB (در production از database استفاده کنید)
USERS_DB = {
    "demo_user": {
        "username": "demo_user",
        "hashed_password": pwd_context.hash("demo_password_123"),
        "email": "demo@example.com"
    }
}

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# استفاده در endpoints:
# @app.post("/convert")
# async def convert(username: str = Depends(verify_token), ...):
```

#### گام 3: افزودن Login endpoint

در `web_server_fixed.py`:

```python
from .security import create_access_token, pwd_context, USERS_DB, verify_token
from fastapi import Depends

@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = USERS_DB.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        raise HTTPException(401, "Invalid credentials")
    
    token = create_access_token({"sub": username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/convert")
async def convert(
    username: str = Depends(verify_token),  # اضافه کنید
    file: UploadFile = File(...),
    ...
):
    # بقیه کد
```

#### گام 4: Rate Limiting

در `web_server_fixed.py`:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/convert")
@limiter.limit("5/minute")  # 5 تبدیل در دقیقه
async def convert(
    request: Request,
    username: str = Depends(verify_token),
    ...
):
    # بقیه کد
```

#### گام 5: File Validation

```python
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".dxf", ".dwg", ".pdf", ".jpg", ".jpeg", ".png"}

@app.post("/convert")
async def convert(file: UploadFile = File(...), ...):
    # بررسی extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {file_ext} not allowed")
    
    # بررسی حجم
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large (max 50MB)")
    
    # ذخیره موقت
    temp_path = Path(tempfile.mktemp(suffix=file_ext))
    temp_path.write_bytes(contents)
```

---

### روز 2: Docker Setup 🐳

#### گام 1: ایجاد Dockerfile

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn slowapi python-jose passlib

COPY . .

RUN mkdir -p uploads outputs models logs && \
    useradd -m appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["gunicorn", "cad3d.web_server_fixed:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

#### گام 2: ایجاد docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./models:/app/models
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app
    restart: unless-stopped

volumes:
  redis_data:
```

#### گام 3: پیکربندی Nginx

ایجاد `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    server {
        listen 80;
        server_name localhost;

        client_max_body_size 50M;

        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 300s;
        }

        location /static {
            proxy_pass http://app/static;
        }
    }
}
```

#### گام 4: راه‌اندازی

```powershell
# ایجاد secret key
$env:SECRET_KEY = -join ((65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})

# Build و run
docker-compose build
docker-compose up -d

# مشاهده logs
docker-compose logs -f app
```

---

### روز 3: Monitoring و Logging 📊

#### گام 1: نصب پکیج‌ها

```powershell
pip install python-json-logger prometheus-client prometheus-fastapi-instrumentator
```

#### گام 2: Setup Logging

ایجاد `cad3d/logging_config.py`:

```python
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # JSON formatter برای production
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logging()
```

در `web_server_fixed.py`:

```python
from .logging_config import logger

@app.post("/convert")
async def convert(...):
    logger.info(f"Conversion started", extra={
        "user": username,
        "filename": file.filename,
        "size": len(contents)
    })
    
    try:
        # پردازش
        result = process_file(...)
        logger.info("Conversion completed", extra={"result": result})
        return result
    except Exception as e:
        logger.error(f"Conversion failed", extra={
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise
```

#### گام 3: Prometheus Metrics

```python
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Metrics
conversion_counter = Counter(
    'conversions_total',
    'Total number of conversions',
    ['format', 'status']
)

conversion_duration = Histogram(
    'conversion_duration_seconds',
    'Time spent converting files',
    ['format']
)

# در app startup
@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)

@app.post("/convert")
async def convert(...):
    with conversion_duration.labels(format=out_format).time():
        try:
            result = process_file(...)
            conversion_counter.labels(format=out_format, status='success').inc()
            return result
        except Exception as e:
            conversion_counter.labels(format=out_format, status='error').inc()
            raise
```

---

### روز 4: Backup و Testing 💾

#### گام 1: Backup Script

ایجاد `scripts/backup.ps1`:

```powershell
# Backup settings
$BackupDir = "C:\backups\cad3d"
$Date = Get-Date -Format "yyyyMMdd_HHmmss"

# Create backup directory
New-Item -ItemType Directory -Force -Path $BackupDir

# Backup uploads
Compress-Archive -Path "uploads\*" -DestinationPath "$BackupDir\uploads_$Date.zip"

# Backup models
Compress-Archive -Path "models\*" -DestinationPath "$BackupDir\models_$Date.zip"

# Cleanup old backups (keep 7 days)
Get-ChildItem $BackupDir -Filter "*.zip" | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    Remove-Item

Write-Host "Backup completed: $Date"
```

Schedule با Task Scheduler:

```powershell
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\path\to\backup.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -TaskName "CAD3D Backup" -Action $action -Trigger $trigger
```

#### گام 2: Basic Tests

ایجاد `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from cad3d.web_server_fixed import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_login():
    response = client.post("/api/login", data={
        "username": "demo_user",
        "password": "demo_password_123"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_convert_requires_auth():
    response = client.post("/convert")
    assert response.status_code == 401
```

اجرا:

```powershell
pip install pytest httpx
pytest tests/ -v
```

---

### روز 5: Documentation و Deploy 📝

#### گام 1: Environment Variables

ایجاد `.env.production`:

```bash
# Security
SECRET_KEY=your-random-32-char-secret-key-here
ALLOWED_ORIGINS=https://yourdomain.com

# File Upload
MAX_FILE_SIZE_MB=50

# CAD
DEFAULT_EXTRUDE_HEIGHT=3000
ODA_CONVERTER_PATH=/path/to/ODA/ODAFileConverter.exe
MIDAS_ONNX_PATH=/app/models/midas_v2_small_256.onnx

# Redis
REDIS_URL=redis://redis:6379/0

# Logging
LOG_LEVEL=INFO
```

#### گام 2: Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": check_redis_connection(),
            "oda_converter": check_oda_available(),
        }
    }

def check_redis_connection() -> bool:
    try:
        # Test Redis connection
        return True
    except:
        return False
```

#### گام 3: Deploy به Server

```powershell
# روی server خودتان:

# 1. Clone repository
git clone https://github.com/your-username/cad3d.git
cd cad3d

# 2. Setup environment
cp .env.example .env.production
# ویرایش .env.production

# 3. Start services
docker-compose up -d

# 4. بررسی وضعیت
docker-compose ps
curl http://localhost/health
```

#### گام 4: SSL با Let's Encrypt

```powershell
# نصب certbot
apt install certbot python3-certbot-nginx

# دریافت certificate
certbot --nginx -d yourdomain.com

# Auto-renewal
certbot renew --dry-run
```

---

## 🎯 چک‌لیست نهایی

### قبل از Deploy

- [ ] SECRET_KEY تولید و در .env قرار داده شد
- [ ] Authentication تست شد
- [ ] Rate limiting فعال است
- [ ] File validation کار می‌کند
- [ ] Docker image build می‌شود
- [ ] docker-compose up موفق است
- [ ] Health check پاسخ می‌دهد
- [ ] Logs قابل مشاهده است
- [ ] Backup script کار می‌کند

### بعد از Deploy

- [ ] Domain به IP سرور point شده
- [ ] SSL certificate فعال است
- [ ] HTTPS کار می‌کند
- [ ] Login موفق است
- [ ] تبدیل فایل کار می‌کند
- [ ] تم‌ها به درستی نمایش داده می‌شوند
- [ ] Monitoring فعال است
- [ ] Backup schedule فعال است
- [ ] Documentation به‌روز است

---

## 🚦 سطوح Deploy

### Level 1: تست محلی (همین حالا)

```powershell
# فعال‌سازی امنیت پایه
python -m uvicorn cad3d.web_server_fixed:app --reload
```

### Level 2: Docker محلی (روز 2)

```powershell
docker-compose up
```

### Level 3: VPS Production (روز 5)

```powershell
# روی server
docker-compose -f docker-compose.prod.yml up -d
```

### Level 4: Kubernetes (اختیاری)

```powershell
kubectl apply -f k8s/
```

---

## 💡 نکات مهم

### امنیت

1. **NEVER** commit SECRET_KEY به git
2. **ALWAYS** استفاده از HTTPS در production
3. **ALWAYS** validate user input
4. **NEVER** trust file extensions (check content)

### Performance

1. استفاده از Redis برای caching
2. Background tasks برای تبدیل‌های سنگین
3. CDN برای static files
4. Database indexing

### Monitoring

1. Daily backup check
2. Disk space monitoring
3. Error rate alerts
4. Response time tracking

---

## 📞 مراحل بعدی

**امروز**:

1. ✅ اضافه کردن authentication
2. ✅ تست با postman یا curl
3. ✅ بررسی logs

**این هفته**:

1. ✅ Docker setup
2. ✅ Deploy به staging
3. ✅ Load testing

**این ماه**:

1. ✅ Production deployment
2. ✅ Monitoring complete
3. ✅ User testing

---

## 🆘 نیاز به کمک؟

می‌توانم در هر مرحله کمک کنم:

- پیاده‌سازی کد امنیتی
- Debug مشکلات Docker
- Setup monitoring
- بهینه‌سازی performance

فقط بگویید از کجا شروع کنیم! 🚀
