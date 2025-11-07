# Deployment Guide

This guide covers deploying the Titanic Survival Predictor to various platforms.

---

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Render Deployment](#render-deployment)
- [Azure Deployment](#azure-deployment)
- [Environment Variables](#environment-variables)
- [CI/CD Setup](#cicd-setup)

---

## Local Development

### Quick Start

```bash
# Install dependencies
uv sync

# Run the Flask app
python app.py

# Access at http://localhost:5000
```

### Development Mode

Enable auto-reload for development:

```python
# In app.py
app.run(host='0.0.0.0', port=5000, debug=True)
```

!!! warning "Debug Mode"
    Never enable debug mode in production! It exposes sensitive information and security vulnerabilities.

---

## Docker Deployment

### Build the Image

```bash
docker build -t titanic-survival-predictor:latest .
```

### Run Locally

```bash
# Basic run
docker run -p 5000:5000 titanic-survival-predictor:latest

# Run with persistent models
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  titanic-survival-predictor:latest

# Run in background
docker run -d -p 5000:5000 --name titanic-predictor titanic-survival-predictor:latest
```

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - PORT=5000
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

---

## Render Deployment

[Render](https://render.com) offers free hosting with automatic deployments.

### Prerequisites

1. Render account (free)
2. Docker Hub account
3. GitHub repository connected

### Step 1: Configure Docker Hub

Add these GitHub secrets:
- `DOCKERHUB_USERNAME` - Your Docker Hub username
- `DOCKERHUB_TOKEN` - Access token from Docker Hub

### Step 2: Create Render Service

1. Go to Render Dashboard → "New +" → "Web Service"
2. Select "Deploy an existing image from a registry"
3. Configure:
   - **Image URL:** `docker.io/yourusername/titanic-survival-predictor:latest`
   - **Name:** `titanic-survival-predictor`
   - **Region:** Choose closest to your users
   - **Plan:** Free (or Starter for better performance)

4. Add environment variables:
   ```
   PORT=5000
   PYTHON_VERSION=3.11
   ```

5. Set Health Check Path: `/health`

6. Click "Create Web Service"

### Step 3: Automatic Deployments

The provided `.github/workflows/deploy-render.yml` automatically:
1. Builds Docker image on push to `main`
2. Pushes to Docker Hub
3. Render auto-deploys new images

### Accessing Your App

Your API will be at: `https://titanic-survival-predictor.onrender.com`

!!! info "Free Tier Limitations"
    Free tier apps sleep after 15 minutes of inactivity. First request may take 30-60 seconds to wake up.

---

## Azure Deployment

### Prerequisites

- Azure account
- Azure CLI installed
- Docker Hub account (optional)

### Option 1: Azure Container Instances (Recommended)

Simplest option for small-scale deployment:

```bash
# Login to Azure
az login

# Create resource group
az group create --name titanic-rg --location eastus

# Create container instance
az container create \
  --resource-group titanic-rg \
  --name titanic-predictor \
  --image yourusername/titanic-survival-predictor:latest \
  --dns-name-label titanic-predictor-unique \
  --ports 5000 \
  --cpu 1 \
  --memory 2

# Get the URL
az container show \
  --resource-group titanic-rg \
  --name titanic-predictor \
  --query ipAddress.fqdn
```

### Option 2: Azure App Service

Better for production with scaling capabilities:

```bash
# Create App Service plan
az appservice plan create \
  --name titanic-plan \
  --resource-group titanic-rg \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group titanic-rg \
  --plan titanic-plan \
  --name titanic-survival-app \
  --deployment-container-image-name yourusername/titanic-ml:latest

# Configure port
az webapp config appsettings set \
  --resource-group titanic-rg \
  --name titanic-survival-app \
  --settings WEBSITES_PORT=5000
```

### Monitoring

```bash
# Stream logs
az webapp log tail \
  --resource-group titanic-rg \
  --name titanic-survival-app

# Restart app
az webapp restart \
  --resource-group titanic-rg \
  --name titanic-survival-app
```

---

## Environment Variables

### Required Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Application port | `5000` |
| `FLASK_ENV` | Environment (development/production) | `production` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_DEBUG` | Enable debug mode | `False` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MODEL_PATH` | Path to model file | `models/model.pkl` |
| `PREPROCESSOR_PATH` | Path to preprocessor | `models/preprocessor.pkl` |

### Setting Environment Variables

**Docker:**
```bash
docker run -e FLASK_ENV=production -e PORT=5000 titanic-ml
```

**Render:**
Add in the Render dashboard under "Environment" tab

**Azure:**
```bash
az webapp config appsettings set \
  --settings FLASK_ENV=production PORT=5000
```

---

## CI/CD Setup

The project includes GitHub Actions workflows for automated deployment.

### Workflow Files

- `.github/workflows/ci.yml` - Tests and code quality
- `.github/workflows/deploy-render.yml` - Render deployment
- `.github/workflows/deploy-azure.yml` - Azure deployment

### Required GitHub Secrets

#### For Docker Hub & Render:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

#### For Azure:
- `AZURE_CREDENTIALS` - Service principal JSON
- `AZURE_WEBAPP_NAME` - Your app name
- `AZURE_REGISTRY_LOGIN_SERVER` - ACR server (optional)
- `AZURE_REGISTRY_USERNAME` - ACR username (optional)
- `AZURE_REGISTRY_PASSWORD` - ACR password (optional)

### Creating Azure Service Principal

```bash
az ad sp create-for-rbac \
  --name "github-actions-titanic" \
  --sdk-auth \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/titanic-rg
```

Copy the JSON output to `AZURE_CREDENTIALS` secret.

### Manual Deployment Trigger

Go to GitHub → Actions → Select workflow → Run workflow

---

## Performance Tips

### For Free Tiers

**Render Free:**
- Use health check service to keep awake: [cron-job.org](https://cron-job.org)
- Expect 30-60s cold start

**Azure Free (F1):**
- Limited to 60 CPU minutes/day
- 1 GB RAM limit
- Consider B1 tier ($13/month) for production

### Optimizing Docker Image

Current optimizations:
- Multi-stage build
- Minimal base image (python:3.11-slim)
- No unnecessary dependencies
- `.dockerignore` excludes dev files

To further optimize:
```dockerfile
# Use alpine for smaller size
FROM python:3.11-alpine

# Or use distroless for security
FROM gcr.io/distroless/python3-debian11
```

### Model Loading

Models are loaded on startup (~2-3 seconds). For faster response:
- Use model caching
- Consider model as a service (separate microservice)
- Pre-warm container on deployment

---

## Monitoring & Observability

### Health Check Endpoint

```bash
curl http://your-app-url/health
```

Response:
```json
{
  "status": "healthy",
  "service": "titanic-ml-api"
}
```

### Logging

Logs are sent to:
- **Local:** Console output
- **Render:** Logs tab in dashboard
- **Azure:** Application Insights (if configured)

### Metrics to Monitor

- Request rate
- Response time
- Error rate (4xx, 5xx)
- Prediction accuracy (if collecting feedback)
- Model version in use

---

## Troubleshooting

### Application Won't Start

Check logs for errors:
```bash
# Docker
docker logs titanic-ml

# Render
# View in dashboard

# Azure
az webapp log tail --name your-app-name
```

### Predictions Failing

1. Verify model files exist
2. Check input data format
3. Review error logs
4. Test locally first

### Slow Response Times

1. Check cold start (Render free tier)
2. Verify sufficient memory (2GB recommended)
3. Optimize model loading
4. Consider caching predictions

### Port Binding Issues

Ensure PORT environment variable matches:
- Dockerfile `EXPOSE` command
- Flask app configuration
- Platform settings

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| **Render** | ✅ 750 hrs/month | $7/mo | Quick demos |
| **Azure ACI** | ❌ No free | Pay-per-use | Simple containers |
| **Azure App Service** | ✅ Limited (F1) | $13/mo (B1) | Production apps |
| **Railway** | ✅ $5 credit | Pay-as-you-go | Simple projects |
| **Fly.io** | ✅ Free allowance | Pay-as-you-go | Global edge |

---

## Security Best Practices

1. **Never commit secrets** - Use environment variables
2. **Use HTTPS** - All platforms provide free SSL
3. **Disable debug mode** - Set `FLASK_DEBUG=False`
4. **Validate inputs** - Implement input validation
5. **Rate limiting** - Prevent abuse
6. **Update dependencies** - Regular security updates

---

## Next Steps

After deployment:

<div class="grid cards" markdown>

- :material-monitor-dashboard:{ .lg .middle } **Set Up Monitoring**

    ---

    Monitor your app with Render's built-in metrics dashboard and configure alerts for downtime or errors.

- :material-domain:{ .lg .middle } **Custom Domain**

    ---

    Add your own domain name through Render's settings or use their subdomain.

- :material-scale-balance:{ .lg .middle } **Scale Your App**

    ---

    Upgrade your Render plan for better performance and handle more concurrent requests.

- :material-security:{ .lg .middle } **Enhance Security**

    ---

    Add HTTPS (enabled by default), environment-based secrets, and rate limiting middleware.

</div>

---

## Support

Need help with deployment?

- Review [Troubleshooting](#troubleshooting) section above
- Open an issue on [GitHub](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster/issues)
- Email: k.s.bonev@gmail.com
