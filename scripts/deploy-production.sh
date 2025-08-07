#!/bin/bash
#
# Complete Project Deployment Script for Cyber-LLM
# Production-ready deployment across multiple platforms
#
# Author: Muzan Sano <sanosensei36@gmail.com>

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="cyber-llm"
VERSION="1.0.0"
DOCKER_REGISTRY="cyber-llm"
NAMESPACE="cyber-llm-production"

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check for required tools
    local required_tools=("docker" "kubectl" "helm" "python3" "pip3")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is not installed or not in PATH"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    log "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Core services
    local services=(
        "recon-agent"
        "c2-agent" 
        "post-exploit-agent"
        "explainability-agent"
        "safety-agent"
        "orchestrator"
        "api-gateway"
        "web-interface"
    )
    
    for service in "${services[@]}"; do
        log "Building $service image..."
        docker build \
            -f "src/deployment/docker/Dockerfile.$service" \
            -t "$DOCKER_REGISTRY/$service:$VERSION" \
            -t "$DOCKER_REGISTRY/$service:latest" \
            .
        
        # Push to registry if specified
        if [[ "${PUSH_IMAGES:-false}" == "true" ]]; then
            docker push "$DOCKER_REGISTRY/$service:$VERSION"
            docker push "$DOCKER_REGISTRY/$service:latest"
        fi
    done
    
    log "Docker images built successfully"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure..."
    
    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy secrets
    kubectl apply -f src/deployment/k8s/secrets/ -n "$NAMESPACE"
    
    # Deploy configmaps
    kubectl apply -f src/deployment/k8s/configmaps/ -n "$NAMESPACE"
    
    # Deploy storage
    kubectl apply -f src/deployment/k8s/storage.yaml -n "$NAMESPACE"
    
    # Deploy RBAC
    kubectl apply -f src/deployment/k8s/rbac.yaml -n "$NAMESPACE"
    
    # Deploy network policies
    kubectl apply -f src/deployment/k8s/network-policies.yaml -n "$NAMESPACE"
    
    log "Infrastructure deployed"
}

# Deploy core services
deploy_core_services() {
    log "Deploying core services..."
    
    # Deploy databases
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    # PostgreSQL for main database
    helm upgrade --install postgresql bitnami/postgresql \
        --namespace "$NAMESPACE" \
        --set auth.postgresPassword="$(kubectl get secret postgres-secret -n $NAMESPACE -o jsonpath='{.data.password}' | base64 -d)" \
        --set primary.persistence.enabled=true \
        --set primary.persistence.size=50Gi
    
    # Redis for caching
    helm upgrade --install redis bitnami/redis \
        --namespace "$NAMESPACE" \
        --set auth.password="$(kubectl get secret redis-secret -n $NAMESPACE -o jsonpath='{.data.password}' | base64 -d)" \
        --set master.persistence.enabled=true \
        --set master.persistence.size=20Gi
    
    # Elasticsearch for logging
    helm repo add elastic https://helm.elastic.co
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace "$NAMESPACE" \
        --set persistence.enabled=true \
        --set volumeClaimTemplate.resources.requests.storage=100Gi
    
    log "Core services deployed"
}

# Deploy AI agents
deploy_ai_agents() {
    log "Deploying AI agents..."
    
    # Apply AI agent deployments
    kubectl apply -f src/deployment/k8s/agents/ -n "$NAMESPACE"
    
    # Wait for deployments to be ready
    local agents=("recon-agent" "c2-agent" "post-exploit-agent" "explainability-agent" "safety-agent")
    for agent in "${agents[@]}"; do
        log "Waiting for $agent deployment..."
        kubectl rollout status deployment/"$agent" -n "$NAMESPACE" --timeout=300s
    done
    
    log "AI agents deployed successfully"
}

# Deploy orchestration layer
deploy_orchestration() {
    log "Deploying orchestration layer..."
    
    # Deploy main orchestrator
    kubectl apply -f src/deployment/k8s/orchestrator.yaml -n "$NAMESPACE"
    
    # Deploy workflow engine
    kubectl apply -f src/deployment/k8s/workflow-engine.yaml -n "$NAMESPACE"
    
    # Wait for orchestration to be ready
    kubectl rollout status deployment/orchestrator -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/workflow-engine -n "$NAMESPACE" --timeout=300s
    
    log "Orchestration layer deployed"
}

# Deploy API and web interface
deploy_api_interface() {
    log "Deploying API and web interface..."
    
    # Deploy API gateway
    kubectl apply -f src/deployment/k8s/api-gateway.yaml -n "$NAMESPACE"
    
    # Deploy web interface
    kubectl apply -f src/deployment/k8s/web-interface.yaml -n "$NAMESPACE"
    
    # Deploy ingress
    kubectl apply -f src/deployment/k8s/ingress.yaml -n "$NAMESPACE"
    
    # Wait for deployments
    kubectl rollout status deployment/api-gateway -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/web-interface -n "$NAMESPACE" --timeout=300s
    
    log "API and web interface deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Add Prometheus community repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus stack
    helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=10Gi \
        --values src/deployment/k8s/monitoring/prometheus-values.yaml
    
    # Deploy custom dashboards
    kubectl apply -f src/deployment/k8s/monitoring/ -n monitoring
    
    log "Monitoring stack deployed"
}

# Run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check all deployments are ready
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')
    
    for deployment in $deployments; do
        local ready_replicas
        ready_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        local desired_replicas
        desired_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        
        if [[ "$ready_replicas" != "$desired_replicas" ]]; then
            warn "Deployment $deployment is not fully ready ($ready_replicas/$desired_replicas)"
        else
            log "✓ Deployment $deployment is ready ($ready_replicas/$desired_replicas)"
        fi
    done
    
    # Check services are accessible
    local services
    services=$(kubectl get services -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')
    
    for service in $services; do
        local cluster_ip
        cluster_ip=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if [[ "$cluster_ip" != "None" && "$cluster_ip" != "" ]]; then
            log "✓ Service $service is accessible at $cluster_ip"
        fi
    done
    
    log "Health checks completed"
}

# Get deployment information
get_deployment_info() {
    log "Getting deployment information..."
    
    echo
    echo "==================================="
    echo "CYBER-LLM DEPLOYMENT COMPLETE!"
    echo "==================================="
    echo
    
    # Get ingress information
    local ingress_ip
    ingress_ip=$(kubectl get ingress cyber-llm-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending...")
    
    echo "📋 Deployment Summary:"
    echo "  • Namespace: $NAMESPACE"
    echo "  • Version: $VERSION"
    echo "  • Ingress IP: $ingress_ip"
    echo
    
    echo "🌐 Access URLs:"
    if [[ "$ingress_ip" != "Pending..." ]]; then
        echo "  • Web Interface: https://$ingress_ip/"
        echo "  • API Gateway: https://$ingress_ip/api/"
        echo "  • Health Check: https://$ingress_ip/health"
    else
        echo "  • Waiting for Load Balancer IP assignment..."
        echo "  • Check with: kubectl get ingress cyber-llm-ingress -n $NAMESPACE"
    fi
    echo
    
    echo "📊 Monitoring:"
    local grafana_password
    grafana_password=$(kubectl get secret kube-prometheus-stack-grafana -n monitoring -o jsonpath='{.data.admin-password}' | base64 -d 2>/dev/null || echo "Not available")
    echo "  • Grafana: http://grafana.monitoring.svc.cluster.local:3000"
    echo "  • Username: admin"
    echo "  • Password: $grafana_password"
    echo
    
    echo "🔧 Management Commands:"
    echo "  • View pods: kubectl get pods -n $NAMESPACE"
    echo "  • View logs: kubectl logs -f deployment/orchestrator -n $NAMESPACE"
    echo "  • Scale deployment: kubectl scale deployment orchestrator --replicas=5 -n $NAMESPACE"
    echo "  • Port forward API: kubectl port-forward svc/api-gateway 8080:80 -n $NAMESPACE"
    echo
    
    echo "🏆 Deployment Status: SUCCESSFUL"
    echo "   Ready for enterprise cybersecurity operations!"
    echo
}

# Cleanup function
cleanup_on_exit() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        error "Deployment failed with exit code $exit_code"
        echo
        echo "🔍 Troubleshooting tips:"
        echo "  • Check pod logs: kubectl logs -n $NAMESPACE -l app=<service-name>"
        echo "  • Check events: kubectl get events -n $NAMESPACE --sort-by='.firstTimestamp'"
        echo "  • Check resource usage: kubectl top pods -n $NAMESPACE"
        echo
        echo "🧹 To cleanup partial deployment:"
        echo "  • Delete namespace: kubectl delete namespace $NAMESPACE"
        echo "  • Remove monitoring: helm uninstall kube-prometheus-stack -n monitoring"
    fi
}

# Main deployment function
main() {
    trap cleanup_on_exit EXIT
    
    log "Starting Cyber-LLM deployment..."
    log "Target namespace: $NAMESPACE"
    log "Version: $VERSION"
    
    check_prerequisites
    build_images
    deploy_infrastructure
    deploy_core_services
    deploy_ai_agents
    deploy_orchestration
    deploy_api_interface
    deploy_monitoring
    run_health_checks
    get_deployment_info
    
    log "Cyber-LLM deployment completed successfully! 🚀"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        log "Cleaning up Cyber-LLM deployment..."
        kubectl delete namespace "$NAMESPACE" --ignore-not-found
        helm uninstall kube-prometheus-stack -n monitoring --ignore-not-found
        log "Cleanup completed"
        ;;
    "status")
        log "Checking Cyber-LLM deployment status..."
        kubectl get all -n "$NAMESPACE"
        ;;
    "logs")
        log "Showing orchestrator logs..."
        kubectl logs -f deployment/orchestrator -n "$NAMESPACE"
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|status|logs}"
        echo
        echo "Commands:"
        echo "  deploy  - Deploy complete Cyber-LLM platform"
        echo "  cleanup - Remove all Cyber-LLM resources"
        echo "  status  - Show deployment status"
        echo "  logs    - Show orchestrator logs"
        exit 1
        ;;
esac
