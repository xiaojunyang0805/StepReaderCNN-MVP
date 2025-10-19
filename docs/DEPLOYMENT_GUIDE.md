# StepReaderCNN - Deployment Guide

**Version**: 1.0
**Last Updated**: October 19, 2025

Complete guide for deploying StepReaderCNN in various environments.

---

## Table of Contents

1. [Streamlit Cloud Deployment](#streamlit-cloud-deployment) ⭐ **Recommended**
2. [Local Deployment](#local-deployment)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [API Deployment](#api-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Streamlit Cloud Deployment

### Live Demo

**Current Deployment**: https://stepreadercnn.streamlit.app

This application is successfully deployed on Streamlit Cloud with full functionality.

### Why Streamlit Cloud?

- ✅ **Free hosting** for open-source projects
- ✅ **Automatic deployments** on git push
- ✅ **Zero configuration** required
- ✅ **Built-in HTTPS** and CDN
- ✅ **Suitable for academic research** and demos
- ✅ **No server management** needed

### Deploy Your Own Instance

**Prerequisites**:
- GitHub account
- Public GitHub repository with your code
- Streamlit Community Cloud account (free)

**Steps**:

1. **Prepare Your Repository**:
   ```bash
   # Ensure requirements.txt is present and simplified
   # For Streamlit Cloud, use CPU-only PyTorch
   cat requirements.txt
   ```

   Your `requirements.txt` should contain:
   ```
   torch
   torchvision
   numpy
   pandas
   scipy
   h5py
   matplotlib
   seaborn
   plotly
   streamlit
   fastapi
   uvicorn
   scikit-learn
   tensorboard
   tqdm
   pyyaml
   python-dotenv
   ```

2. **Ensure TestData is Included**:

   **IMPORTANT**: By default, `*.csv` files are often in `.gitignore`. You need to explicitly allow TestData:

   In your `.gitignore`, add:
   ```
   *.csv
   !TestData/*.csv  # ← Add this line to include TestData
   ```

   Then commit TestData:
   ```bash
   git add -f TestData/*.csv
   git commit -m "Include TestData for cloud deployment"
   git push
   ```

3. **Sign up for Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Click "Sign in with GitHub"
   - Authorize Streamlit to access your repositories

4. **Create New App**:
   - Click "New app"
   - Select your repository: `xiaojunyang0805/StepReaderCNN-MVP`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

5. **Wait for Deployment** (2-5 minutes):
   - Streamlit Cloud will:
     - Clone your repository
     - Install dependencies from `requirements.txt`
     - Run `streamlit run app.py`
     - Assign you a URL: `https://[app-name].streamlit.app`

6. **Verify Deployment**:
   - Navigate to your app URL
   - Test "Data Explorer → Upload Data → Load from TestData"
   - Verify 42 CSV files are found
   - Test all major features

### Deployment Configuration (Optional)

Create `.streamlit/config.toml` for customization:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

### Common Issues and Solutions

#### Issue 1: "TestData folder not found"

**Cause**: TestData CSV files are blocked by `.gitignore`

**Solution**:
```bash
# Update .gitignore
echo "!TestData/*.csv" >> .gitignore

# Force add TestData
git add -f TestData/*.csv
git commit -m "Include TestData for deployment"
git push
```

#### Issue 2: PyTorch Installation Timeout

**Cause**: GPU-enabled PyTorch is too large for Streamlit Cloud

**Solution**: Use CPU-only PyTorch (already configured in requirements.txt)

#### Issue 3: App Runs Out of Memory

**Cause**: Loading too much data at once

**Solution**:
- Implement data pagination in `upload_handler.py`
- Use `@st.cache_data` for expensive operations
- Limit batch sizes during training

### Automatic Updates

Once deployed, your app will automatically update when you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "Update feature X"
git push

# Streamlit Cloud will automatically:
# 1. Detect the push
# 2. Pull latest code
# 3. Reinstall dependencies (if requirements.txt changed)
# 4. Restart the app
# 5. Show "Your app is updating..." to users
```

### Managing Your Deployment

**Streamlit Cloud Dashboard**:
- **Logs**: View real-time application logs
- **Settings**: Configure environment secrets, Python version
- **Analytics**: View visitor statistics
- **Reboot**: Manually restart your app
- **Delete**: Remove the deployment

**Environment Secrets** (for sensitive data):
1. Go to app settings in Streamlit Cloud
2. Add secrets in TOML format:
   ```toml
   [passwords]
   admin_password = "secure_password"

   [api_keys]
   openai_key = "sk-..."
   ```
3. Access in code:
   ```python
   import streamlit as st
   password = st.secrets["passwords"]["admin_password"]
   ```

### Resource Limits

Streamlit Community Cloud provides:
- **RAM**: ~1GB per app
- **CPU**: Shared CPU cores
- **Storage**: Repository size limits apply
- **Bandwidth**: Generous for academic projects
- **Apps**: Multiple apps per account (free tier)

**For larger deployments**, see [Production Deployment](#production-deployment) or [Cloud Deployment](#cloud-deployment).

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

**TestData Folder Not Found** (Streamlit Cloud):

**Error**: `TestData folder not found at: /mount/src/stepreadercnn-mvp/TestData`

**Root Cause**: TestData CSV files are blocked by `.gitignore` and not committed to repository.

**Solution**:
1. Check `.gitignore` file for `*.csv` entry
2. Add exception for TestData:
   ```bash
   # Edit .gitignore, add after *.csv line:
   !TestData/*.csv
   ```

3. Force add and commit TestData files:
   ```bash
   git add -f TestData/*.csv
   git commit -m "Include TestData for cloud deployment"
   git push
   ```

4. Streamlit Cloud will automatically redeploy with TestData included

**Verification**:
```bash
# Check if TestData files are tracked
git ls-files TestData/ | wc -l
# Should output: 42

# Check file sizes
git ls-files TestData/ -s | head -5
```

**Files Affected**:
- `.gitignore` (line 52-53)
- `src/gui/upload_handler.py` (lines 44-48) - Path resolution logic
- All 42 CSV files in `TestData/` folder

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

## Deployment Success Stories

### Streamlit Cloud

✅ **Successfully deployed**: https://stepreadercnn.streamlit.app

**Deployment Timeline**:
- Initial deployment: October 16, 2025
- Issue identified: TestData folder not found (blocked by .gitignore)
- Solution implemented: Updated .gitignore to allow TestData/*.csv
- Resolution: October 19, 2025
- **Status**: Fully operational with all 42 TestData files

**Key Optimizations**:
- CPU-only PyTorch to reduce deployment size
- Simplified requirements.txt for faster installs
- Explicit TestData inclusion in .gitignore
- Automatic deployments on git push

**Performance**:
- Load time: ~30-45 seconds (initial cold start)
- Subsequent loads: ~5-10 seconds
- 42 CSV files (113MB) loaded successfully
- All features working correctly

---

**Deployment Status**: ✅ Successfully deployed to Streamlit Cloud - Ready for production with Docker, Cloud platforms, or local setup.
