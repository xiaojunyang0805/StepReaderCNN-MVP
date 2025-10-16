# StepReaderCNN - Deployment Guide

**Version**: 1.0
**Last Updated**: October 16, 2025

Complete guide for deploying StepReaderCNN in various environments.

---

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [API Deployment](#api-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Local Deployment

### Quick Local Setup (Development)

**Prerequisites**:
- Python 3.9+
- 8GB RAM minimum
- 2GB free disk space

**Steps**:

```bash
# 1. Clone repository
git clone <repository-url>
cd StepReaderCNN

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch GUI
streamlit run app.py
```

**Access**: http://localhost:8501

---

## Production Deployment

### Production-Ready Local Setup

**Additional Requirements**:
- Stable network connection
- SSL certificate (for HTTPS)
- Domain name (optional)

**Steps**:

```bash
# 1. Install with production dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings:
#   - Set ENVIRONMENT=production
#   - Configure logging paths
#   - Set secure API keys

# 3. Run with production settings
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

# 4. Enable HTTPS (recommended)
streamlit run app.py \
  --server.port 443 \
  --server.sslCertFile /path/to/cert.pem \
  --server.sslKeyFile /path/to/key.pem
```

### Process Management with systemd

Create `/etc/systemd/system/stepreader.service`:

```ini
[Unit]
Description=StepReaderCNN Streamlit Application
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/StepReaderCNN
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run app.py --server.port 8501 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start**:

```bash
sudo systemctl enable stepreader
sudo systemctl start stepreader
sudo systemctl status stepreader
```

---

## Docker Deployment

### Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  stepreader:
    build: .
    container_name: stepreader-app
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./TestData:/app/TestData
    environment:
      - ENVIRONMENT=production
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add PostgreSQL for data storage
  # postgres:
  #   image: postgres:15
  #   container_name: stepreader-db
  #   environment:
  #     POSTGRES_DB: stepreader
  #     POSTGRES_USER: stepreader
  #     POSTGRES_PASSWORD: secure_password
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   ports:
  #     - "5432:5432"

# volumes:
#   postgres_data:
```

### Build and Run with Docker

```bash
# Build image
docker build -t stepreader:latest .

# Run container
docker run -d \
  --name stepreader \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  stepreader:latest

# Or use docker-compose
docker-compose up -d

# View logs
docker logs -f stepreader

# Stop
docker-compose down
```

### Docker with GPU Support

Update Dockerfile:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.9 python3-pip

# ... rest of Dockerfile
```

Run with GPU:

```bash
docker run --gpus all -d \
  --name stepreader-gpu \
  -p 8501:8501 \
  stepreader:latest
```

---

## Cloud Deployment

### AWS Deployment (EC2)

**1. Launch EC2 Instance**:
- Instance type: t3.medium or larger (8GB+ RAM)
- OS: Ubuntu 22.04 LTS
- Storage: 20GB minimum
- Security group: Open port 8501 (or 443 for HTTPS)

**2. Connect and Setup**:

```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.9 python3-pip python3-venv

# Clone repository
git clone <repository-url>
cd StepReaderCNN

# Setup and run (see Production Deployment above)
```

**3. Configure Security**:

```bash
# Setup firewall
sudo ufw allow 8501/tcp
sudo ufw enable

# Optional: Setup nginx reverse proxy
sudo apt install nginx
```

Nginx config (`/etc/nginx/sites-available/stepreader`):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Google Cloud Platform (GCP)

**Using Cloud Run**:

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/stepreader

# Deploy to Cloud Run
gcloud run deploy stepreader \
  --image gcr.io/PROJECT_ID/stepreader \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Azure Deployment

**Using Azure Container Instances**:

```bash
# Create resource group
az group create --name stepreader-rg --location eastus

# Create container instance
az container create \
  --resource-group stepreader-rg \
  --name stepreader \
  --image your-registry/stepreader:latest \
  --dns-name-label stepreader \
  --ports 8501
```

### Heroku Deployment

Create `Procfile`:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Deploy:

```bash
# Login to Heroku
heroku login

# Create app
heroku create stepreader-app

# Push to Heroku
git push heroku main

# Scale
heroku ps:scale web=1
```

---

## API Deployment

### Deploy FastAPI Backend Separately

**1. Create API-only Dockerfile** (`Dockerfile.api`):

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "src.api.training_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**2. Run API Server**:

```bash
# Local
uvicorn src.api.training_api:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn src.api.training_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
```

**3. Docker API Deployment**:

```bash
docker build -f Dockerfile.api -t stepreader-api:latest .
docker run -d -p 8000:8000 stepreader-api:latest
```

**API Documentation**: http://your-server:8000/docs

---

## Deployment Checklist

### Pre-Deployment

- [ ] Test all functionality locally
- [ ] Run integration tests (`python tests/test_integration_simple.py`)
- [ ] Verify all dependencies installed
- [ ] Check trained models in `outputs/trained_models/`
- [ ] Verify TestData directory accessible
- [ ] Configure environment variables (.env)
- [ ] Setup logging directory

### Security

- [ ] Enable HTTPS/SSL
- [ ] Set strong passwords/API keys
- [ ] Configure firewall rules
- [ ] Setup authentication (if needed)
- [ ] Regular security updates
- [ ] Backup sensitive data
- [ ] Set proper file permissions

### Performance

- [ ] Optimize for production (disable debug mode)
- [ ] Enable caching where appropriate
- [ ] Configure resource limits
- [ ] Setup monitoring
- [ ] Configure logging
- [ ] Setup alerting

### Monitoring

- [ ] Setup health checks
- [ ] Configure logging aggregation
- [ ] Setup performance monitoring
- [ ] Configure error tracking
- [ ] Setup uptime monitoring

---

## Environment Variables

Create `.env` file:

```bash
# Environment
ENVIRONMENT=production  # development, staging, production

# Application
APP_NAME=StepReaderCNN
APP_VERSION=1.0.0
DEBUG=false

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8501

# Paths
DATA_DIR=./data
MODELS_DIR=./outputs/trained_models
LOGS_DIR=./outputs/logs

# Database (if using)
# DATABASE_URL=postgresql://user:password@localhost:5432/stepreader

# API (if using)
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=*

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## Performance Optimization

### For CPU Deployment

```python
# In app.py or config
import torch
torch.set_num_threads(4)  # Limit CPU threads
```

### For GPU Deployment

```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Caching

Enable Streamlit caching:

```python
@st.cache_data
def load_data():
    # Data loading logic
    pass

@st.cache_resource
def load_model():
    # Model loading logic
    pass
```

---

## Monitoring & Logging

### Setup Logging

```python
# Add to app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Check Endpoint

Add to Streamlit config (`.streamlit/config.toml`):

```toml
[server]
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### Monitoring Tools

**Recommended**:
- **Prometheus + Grafana** - Metrics and dashboards
- **ELK Stack** - Log aggregation
- **Sentry** - Error tracking
- **UptimeRobot** - Uptime monitoring

---

## Backup & Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/stepreader"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz outputs/trained_models/

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/

# Backup configs
tar -czf $BACKUP_DIR/configs_$DATE.tar.gz configs/

# Keep only last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

Add to crontab:

```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

---

## Troubleshooting

### Common Issues

**Port Already in Use**:
```bash
# Find process using port
lsof -i :8501  # Linux/Mac
netstat -ano | findstr :8501  # Windows

# Kill process
kill -9 <PID>
```

**Memory Issues**:
```bash
# Increase Docker memory limit
docker run -m 8g stepreader:latest

# Monitor memory usage
docker stats stepreader
```

**Model Not Found**:
```bash
# Verify models directory
ls -la outputs/trained_models/

# Copy models if missing
cp path/to/trained/models/* outputs/trained_models/
```

**Permission Denied**:
```bash
# Fix permissions
chmod -R 755 data/ outputs/
chown -R $USER:$USER data/ outputs/
```

---

## Scaling

### Horizontal Scaling (Multiple Instances)

Use load balancer (nginx, HAProxy, AWS ELB):

```nginx
upstream stepreader {
    least_conn;
    server 192.168.1.10:8501;
    server 192.168.1.11:8501;
    server 192.168.1.12:8501;
}

server {
    listen 80;
    location / {
        proxy_pass http://stepreader;
    }
}
```

### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stepreader
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stepreader
  template:
    metadata:
      labels:
        app: stepreader
    spec:
      containers:
      - name: stepreader
        image: stepreader:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: stepreader-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: stepreader
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl get services
```

---

## Support

For deployment issues:
1. Check logs: `outputs/logs/`
2. Review [Troubleshooting section](#troubleshooting)
3. Consult [Developer Notes](Dev_note.md)
4. Open GitHub issue

---

**Deployment Status**: Ready for production deployment with Docker, Cloud, or local setup.
