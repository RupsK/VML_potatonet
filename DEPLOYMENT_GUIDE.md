# 🚀 Company Server Deployment Guide

## Overview
This guide provides multiple deployment options for the Thermal Image AI Analyzer and Escalator Safety Monitor to make them accessible to your CEO and team 24/7.

## 🎯 Deployment Options

### Option 1: Docker Deployment (Recommended)
**Best for**: Easy deployment, consistent environment, scalability

### Option 2: Cloud Platform Deployment
**Best for**: No server maintenance, automatic scaling, global access

### Option 3: Traditional Server Deployment
**Best for**: Full control, existing infrastructure, cost-effective

---

## 🐳 Option 1: Docker Deployment

### Prerequisites
- Docker installed on server
- Docker Compose (optional, for multi-app setup)

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for model cache
RUN mkdir -p model_cache

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Create Docker Compose for Multi-App Setup
```yaml
version: '3.8'

services:
  thermal-analyzer:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./model_cache:/app/model_cache
      - ./test_image:/app/test_image
      - ./test_video:/app/test_video
    environment:
      - HF_TOKEN=${HF_TOKEN}
    restart: unless-stopped
    container_name: thermal-analyzer

  escalator-safety:
    build: .
    ports:
      - "8502:8501"
    volumes:
      - ./model_cache:/app/model_cache
      - ./test_video:/app/test_video
    environment:
      - HF_TOKEN=${HF_TOKEN}
    command: ["streamlit", "run", "streamlit_escalator_vlm.py", "--server.port=8501", "--server.address=0.0.0.0"]
    restart: unless-stopped
    container_name: escalator-safety

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - thermal-analyzer
      - escalator-safety
    restart: unless-stopped
    container_name: nginx-proxy
```

### Step 3: Create Nginx Configuration
```nginx
events {
    worker_connections 1024;
}

http {
    upstream thermal_app {
        server thermal-analyzer:8501;
    }
    
    upstream escalator_app {
        server escalator-safety:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Thermal Image Analyzer
        location / {
            proxy_pass http://thermal_app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }
        
        # Escalator Safety Monitor
        location /escalator/ {
            proxy_pass http://escalator_app/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }
    }
}
```

### Step 4: Deployment Commands
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Update and restart
docker-compose pull
docker-compose up -d --build
```

---

## ☁️ Option 2: Cloud Platform Deployment

### A. Streamlit Cloud (Easiest)
1. **Push code to GitHub**
2. **Connect to Streamlit Cloud**
3. **Deploy automatically**

### B. Heroku Deployment
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Create runtime.txt
echo "python-3.9.18" > runtime.txt

# Deploy
heroku create your-app-name
heroku config:set HF_TOKEN=your_token_here
git push heroku main
```

### C. Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/thermal-analyzer
gcloud run deploy thermal-analyzer --image gcr.io/PROJECT_ID/thermal-analyzer --platform managed --allow-unauthenticated
```

---

## 🖥️ Option 3: Traditional Server Deployment

### Step 1: Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.9 python3.9-pip python3.9-venv nginx -y

# Create application directory
sudo mkdir -p /opt/thermal-analyzer
sudo chown $USER:$USER /opt/thermal-analyzer
```

### Step 2: Application Installation
```bash
cd /opt/thermal-analyzer

# Clone or copy your application
git clone <your-repo-url> .

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_token_here"
```

### Step 3: Systemd Service Setup
```bash
# Create service file
sudo nano /etc/systemd/system/thermal-analyzer.service
```

```ini
[Unit]
Description=Thermal Image Analyzer
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/thermal-analyzer
Environment=PATH=/opt/thermal-analyzer/venv/bin
Environment=HF_TOKEN=your_token_here
ExecStart=/opt/thermal-analyzer/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Step 4: Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/thermal-analyzer
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Step 5: Enable and Start Services
```bash
# Enable nginx site
sudo ln -s /etc/nginx/sites-available/thermal-analyzer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Enable and start application
sudo systemctl enable thermal-analyzer
sudo systemctl start thermal-analyzer
sudo systemctl status thermal-analyzer
```

---

## 🔐 Security Considerations

### 1. Environment Variables
```bash
# Create .env file
HF_TOKEN=your_huggingface_token_here
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

### 2. SSL Certificate (Let's Encrypt)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. Firewall Configuration
```bash
# Configure UFW
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

## 📊 Monitoring and Maintenance

### 1. Log Monitoring
```bash
# View application logs
sudo journalctl -u thermal-analyzer -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 2. Health Checks
```bash
# Create health check script
nano /opt/thermal-analyzer/health_check.sh
```

```bash
#!/bin/bash
if curl -f http://localhost:8501/_stcore/health; then
    echo "Application is healthy"
    exit 0
else
    echo "Application is down"
    sudo systemctl restart thermal-analyzer
    exit 1
fi
```

### 3. Automated Backups
```bash
# Create backup script
nano /opt/thermal-analyzer/backup.sh
```

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf /backups/thermal-analyzer_$DATE.tar.gz /opt/thermal-analyzer
find /backups -name "*.tar.gz" -mtime +7 -delete
```

---

## 🚀 Quick Start Commands

### For Docker Deployment:
```bash
# Clone and deploy
git clone <your-repo-url> thermal-analyzer
cd thermal-analyzer
docker-compose up -d

# Access applications
# Thermal Analyzer: http://your-server:8501
# Escalator Safety: http://your-server:8502
```

### For Traditional Server:
```bash
# One-command deployment script
curl -sSL https://raw.githubusercontent.com/your-repo/deploy.sh | bash
```

---

## 📞 Support and Troubleshooting

### Common Issues:
1. **Port conflicts**: Change ports in docker-compose.yml or nginx config
2. **Memory issues**: Increase Docker memory limits or server RAM
3. **Model loading**: Ensure model_cache directory has proper permissions
4. **SSL issues**: Check certificate paths and nginx configuration

### Contact Information:
- **Technical Support**: [your-email@company.com]
- **Emergency Contact**: [emergency-phone]
- **Documentation**: [internal-wiki-link]

---

## 🎯 CEO Access Instructions

### For Your CEO:
1. **Primary URL**: https://your-domain.com
2. **Escalator Safety**: https://your-domain.com/escalator/
3. **Mobile Access**: All applications are mobile-responsive
4. **Bookmark**: Add to browser bookmarks for quick access
5. **Notifications**: Set up email alerts for critical safety incidents

### Access Credentials:
- **No login required** for basic access
- **Admin panel**: [if needed, provide separate credentials]
- **API access**: [if needed for integrations]

---

## 📈 Performance Optimization

### 1. Model Caching
- Models are automatically cached in `model_cache/` directory
- First run may be slower, subsequent runs will be faster

### 2. Resource Allocation
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Production**: 16GB RAM, 8 CPU cores

### 3. Scaling Options
- **Horizontal**: Deploy multiple instances behind load balancer
- **Vertical**: Increase server resources
- **Cloud**: Use auto-scaling groups

---

**Ready to deploy? Choose your preferred option and follow the step-by-step instructions above!** 