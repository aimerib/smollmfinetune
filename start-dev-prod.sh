#!/bin/bash

# Character Creation Devkit - Production-like Development Environment
# Startup script for Mac with 32GB RAM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop for Mac."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    # Check available memory
    total_memory=$(sysctl -n hw.memsize)
    total_memory_gb=$((total_memory / 1024 / 1024 / 1024))
    
    if [ $total_memory_gb -lt 16 ]; then
        print_warning "System has ${total_memory_gb}GB RAM. 32GB recommended for optimal performance."
    else
        print_success "System has ${total_memory_gb}GB RAM - sufficient for dev-prod environment."
    fi
    
    print_success "Prerequisites check passed!"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p infra/ssl
    mkdir -p infra/monitoring
    mkdir -p logs
    mkdir -p data/postgres
    mkdir -p data/elasticsearch
    mkdir -p data/prometheus
    mkdir -p data/grafana
    
    print_success "Directories created!"
}

# Generate self-signed SSL certificates for development
generate_ssl_certs() {
    print_status "Generating SSL certificates for development..."
    
    if [ ! -f infra/ssl/localhost.crt ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout infra/ssl/localhost.key \
            -out infra/ssl/localhost.crt \
            -subj "/C=US/ST=Development/L=Local/O=CharacterDevkit/CN=localhost"
        
        print_success "SSL certificates generated!"
    else
        print_status "SSL certificates already exist."
    fi
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build backend image
    if [ -f backend/Dockerfile.dev ]; then
        docker-compose -f docker-compose.dev-prod.yml build platform-api-1
        print_success "Backend image built!"
    fi
    
    # Build frontend image
    if [ -f client/Dockerfile.dev ]; then
        docker-compose -f docker-compose.dev-prod.yml build platform-client-1
        print_success "Frontend image built!"
    fi
    
    # Build inference engine if it exists
    if [ -f app/inference_engine/Dockerfile.dev ]; then
        docker-compose -f docker-compose.dev-prod.yml build inference-engine
        print_success "Inference engine image built!"
    fi
}

# Start services in stages
start_services() {
    print_status "Starting production-like development environment..."
    
    # Stage 1: Infrastructure services
    print_status "Stage 1: Starting infrastructure services..."
    docker-compose -f docker-compose.dev-prod.yml up -d \
        db-primary \
        redis-cluster \
        elasticsearch
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure services to be ready..."
    sleep 30
    
    # Stage 2: Application services
    print_status "Stage 2: Starting application services..."
    docker-compose -f docker-compose.dev-prod.yml up -d \
        platform-api-1 \
        platform-api-2 \
        inference-engine \
        celery-worker
    
    # Wait for backend services
    print_status "Waiting for backend services to be ready..."
    sleep 20
    
    # Stage 3: Frontend and load balancer
    print_status "Stage 3: Starting frontend and load balancer..."
    docker-compose -f docker-compose.dev-prod.yml up -d \
        platform-client-1 \
        platform-client-2 \
        nginx-lb
    
    # Stage 4: Monitoring services
    print_status "Stage 4: Starting monitoring services..."
    docker-compose -f docker-compose.dev-prod.yml up -d \
        prometheus \
        grafana \
        kibana \
        flower
    
    # Wait for all services
    print_status "Waiting for all services to be ready..."
    sleep 30
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    services=(
        "http://localhost/health:Load Balancer"
        "http://localhost:8001/health:Platform API 1"
        "http://localhost:8002/health:Platform API 2"
        "http://localhost:8100/health:Inference Engine"
        "http://localhost:9090/-/healthy:Prometheus"
        "http://localhost:5601/api/status:Kibana"
        "http://localhost:5555:Flower"
    )
    
    healthy_count=0
    total_count=${#services[@]}
    
    for service in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service"
        
        if curl -f -s "$url" > /dev/null 2>&1; then
            print_success "$name is healthy"
            ((healthy_count++))
        else
            print_warning "$name is not responding"
        fi
    done
    
    print_status "Health check complete: $healthy_count/$total_count services healthy"
}

# Display access information
show_access_info() {
    echo ""
    echo "=========================================="
    echo "🎮 Character Creation Devkit - Dev-Prod Environment"
    echo "=========================================="
    echo ""
    echo "🌐 Main Application:"
    echo "   http://localhost        - Load balanced application"
    echo "   http://localhost/monitoring - Production monitoring dashboard"
    echo ""
    echo "🔧 Direct Service Access:"
    echo "   http://localhost:3001   - React Client 1"
    echo "   http://localhost:3002   - React Client 2"  
    echo "   http://localhost:8001   - Platform API 1"
    echo "   http://localhost:8002   - Platform API 2"
    echo "   http://localhost:8100   - Inference Engine"
    echo ""
    echo "📊 Monitoring & Debugging:"
    echo "   http://localhost:9090   - Prometheus Metrics"
    echo "   http://localhost:3000   - Grafana Dashboards (admin/admin123)"
    echo "   http://localhost:5601   - Kibana Logs"
    echo "   http://localhost:5555   - Celery Flower"
    echo "   http://localhost:8080   - Debug proxy (access all services)"
    echo ""
    echo "🔌 Database Access:"
    echo "   localhost:5432          - PostgreSQL Primary"
    echo "   localhost:5433          - PostgreSQL Replica"
    echo "   localhost:6379          - Redis"
    echo "   localhost:9200          - Elasticsearch"
    echo ""
    echo "🐛 Debug Ports:"
    echo "   localhost:5678          - Platform API 1 Debug"
    echo "   localhost:5679          - Platform API 2 Debug"
    echo "   localhost:5680          - Inference Engine Debug"
    echo ""
    echo "🛠️  Management Commands:"
    echo "   ./start-dev-prod.sh stop     - Stop all services"
    echo "   ./start-dev-prod.sh restart  - Restart all services"
    echo "   ./start-dev-prod.sh logs     - Show service logs"
    echo "   ./start-dev-prod.sh status   - Check service status"
    echo ""
    echo "🎯 Production Features Active:"
    echo "   ✅ Load balancing across multiple instances"
    echo "   ✅ Database primary/replica setup"
    echo "   ✅ Real-time monitoring and metrics"
    echo "   ✅ Centralized logging with Elasticsearch"
    echo "   ✅ Auto-scaling monitoring (Prometheus alerts)"
    echo "   ✅ Production-like service mesh"
    echo ""
    print_success "Development environment is ready! 🚀"
}

# Stop services
stop_services() {
    print_status "Stopping dev-prod environment..."
    docker-compose -f docker-compose.dev-prod.yml down
    print_success "All services stopped!"
}

# Restart services
restart_services() {
    print_status "Restarting dev-prod environment..."
    stop_services
    sleep 5
    start_services
    check_health
    show_access_info
}

# Show logs
show_logs() {
    service=${2:-""}
    if [ -n "$service" ]; then
        print_status "Showing logs for $service..."
        docker-compose -f docker-compose.dev-prod.yml logs -f "$service"
    else
        print_status "Showing logs for all services..."
        docker-compose -f docker-compose.dev-prod.yml logs -f
    fi
}

# Show status
show_status() {
    print_status "Service Status:"
    docker-compose -f docker-compose.dev-prod.yml ps
    echo ""
    check_health
}

# Clean up everything
cleanup() {
    print_status "Cleaning up dev-prod environment..."
    docker-compose -f docker-compose.dev-prod.yml down -v
    docker system prune -f
    print_success "Environment cleaned up!"
}

# Main script logic
case "${1:-start}" in
    "start")
        echo "🎮 Starting Character Creation Devkit - Production-like Development Environment"
        echo ""
        check_prerequisites
        create_directories
        generate_ssl_certs
        build_images
        start_services
        check_health
        show_access_info
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "logs")
        show_logs "$@"
        ;;
    "status")
        show_status
        ;;
    "health")
        check_health
        ;;
    "cleanup")
        cleanup
        ;;
    "help")
        echo "Usage: $0 {start|stop|restart|logs [service]|status|health|cleanup|help}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the dev-prod environment (default)"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart all services"
        echo "  logs      - Show logs (optionally for specific service)"
        echo "  status    - Show service status"
        echo "  health    - Check service health"
        echo "  cleanup   - Stop and clean up everything"
        echo "  help      - Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac 