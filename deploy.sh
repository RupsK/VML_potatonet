#!/bin/bash

# Thermal Image AI Analyzer Deployment Script
# This script deploys the application to a company server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="thermal-analyzer"
DOMAIN="${1:-localhost}"
HF_TOKEN="${2:-}"

echo -e "${BLUE}🔥 Thermal Image AI Analyzer Deployment Script 🔥${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check prerequisites
print_status "Checking prerequisites..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Installation guide: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    echo "Installation guide: https://docs.docker.com/compose/install/"
    exit 1
fi

print_status "Docker and Docker Compose are installed"

# Check if we're in the right directory
if [[ ! -f "streamlit_app.py" ]] || [[ ! -f "docker-compose.yml" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_status "Project files found"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p model_cache test_image test_video ssl

# Set up environment variables
if [[ -z "$HF_TOKEN" ]]; then
    print_warning "No Hugging Face token provided. Some features may not work."
    echo "You can set the token later by editing the .env file"
else
    print_status "Hugging Face token provided"
fi

# Create .env file
cat > .env << EOF
HF_TOKEN=$HF_TOKEN
DOMAIN=$DOMAIN
EOF

print_status "Environment file created"

# Update nginx configuration with domain
if [[ "$DOMAIN" != "localhost" ]]; then
    print_status "Updating nginx configuration for domain: $DOMAIN"
    sed -i "s/server_name _;/server_name $DOMAIN;/" nginx.conf
fi

# Build and start services
print_status "Building and starting services..."
docker-compose down 2>/dev/null || true
docker-compose build --no-cache
docker-compose up -d

# Wait for services to start
print_status "Waiting for services to start..."
sleep 10

# Check if services are running
print_status "Checking service status..."

if docker-compose ps | grep -q "Up"; then
    print_status "All services are running successfully!"
else
    print_error "Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Display access information
echo -e "\n${BLUE}🎉 Deployment Complete! 🎉${NC}"
echo -e "${BLUE}========================${NC}"
echo -e "${GREEN}Access URLs:${NC}"
echo -e "  🔥 Thermal Image Analyzer: ${YELLOW}http://$DOMAIN${NC}"
echo -e "  🤖 Escalator Safety Monitor: ${YELLOW}http://$DOMAIN/escalator/${NC}"
echo -e "  📊 Direct Port Access:"
echo -e "    - Thermal Analyzer: ${YELLOW}http://$DOMAIN:8501${NC}"
echo -e "    - Escalator Safety: ${YELLOW}http://$DOMAIN:8502${NC}"
echo -e "  🏥 Health Check: ${YELLOW}http://$DOMAIN/health${NC}"

echo -e "\n${GREEN}Useful Commands:${NC}"
echo -e "  📋 View logs: ${YELLOW}docker-compose logs -f${NC}"
echo -e "  🛑 Stop services: ${YELLOW}docker-compose down${NC}"
echo -e "  🔄 Restart services: ${YELLOW}docker-compose restart${NC}"
echo -e "  📊 Service status: ${YELLOW}docker-compose ps${NC}"

echo -e "\n${GREEN}For CEO Access:${NC}"
echo -e "  📱 Mobile-friendly interface available"
echo -e "  🔐 No login required for basic access"
echo -e "  📧 Set up email notifications for critical alerts"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Test the applications using the URLs above"
echo -e "  2. Configure SSL certificates for HTTPS (recommended for production)"
echo -e "  3. Set up monitoring and alerts"
echo -e "  4. Configure backup procedures"

print_status "Deployment completed successfully!" 