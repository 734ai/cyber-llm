#!/bin/bash
# Cyber-LLM Kubernetes Deployment Script
# Author: Muzan Sano <sanosensei36@gmail.com>

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-"cyber-llm"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
KUBECTL_CONTEXT=${KUBECTL_CONTEXT:-""}
DRY_RUN=${DRY_RUN:-"false"}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if context is set
    if [[ -n "$KUBECTL_CONTEXT" ]]; then
        kubectl config use-context "$KUBECTL_CONTEXT"
    fi
    
    # Verify cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

deploy_namespace() {
    log_info "Deploying namespace and RBAC..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f src/deployment/k8s/namespace.yaml
        kubectl apply --dry-run=client -f src/deployment/k8s/rbac.yaml
    else
        kubectl apply -f src/deployment/k8s/namespace.yaml
        kubectl apply -f src/deployment/k8s/rbac.yaml
    fi
    
    log_success "Namespace and RBAC deployed"
}

deploy_storage() {
    log_info "Deploying storage resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f src/deployment/k8s/storage.yaml
    else
        kubectl apply -f src/deployment/k8s/storage.yaml
        
        # Wait for PVCs to be bound
        kubectl wait --for=condition=Bound pvc/cyber-llm-models-pvc -n "$NAMESPACE" --timeout=300s
        kubectl wait --for=condition=Bound pvc/cyber-llm-data-pvc -n "$NAMESPACE" --timeout=300s
    fi
    
    log_success "Storage resources deployed"
}

deploy_config() {
    log_info "Deploying configuration..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f src/deployment/k8s/configmap.yaml
    else
        kubectl apply -f src/deployment/k8s/configmap.yaml
    fi
    
    log_success "Configuration deployed"
}

deploy_application() {
    log_info "Deploying Cyber-LLM application..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f src/deployment/k8s/deployment.yaml
        kubectl apply --dry-run=client -f src/deployment/k8s/service.yaml
    else
        kubectl apply -f src/deployment/k8s/deployment.yaml
        kubectl apply -f src/deployment/k8s/service.yaml
        
        # Wait for deployment to be ready
        kubectl wait --for=condition=available --timeout=600s deployment/cyber-llm-api -n "$NAMESPACE"
    fi
    
    log_success "Application deployed"
}

deploy_autoscaling() {
    log_info "Deploying autoscaling configuration..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f src/deployment/k8s/autoscaling.yaml
    else
        kubectl apply -f src/deployment/k8s/autoscaling.yaml
    fi
    
    log_success "Autoscaling configured"
}

deploy_ingress() {
    log_info "Deploying ingress..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply --dry-run=client -f src/deployment/k8s/ingress.yaml
    else
        kubectl apply -f src/deployment/k8s/ingress.yaml
    fi
    
    log_success "Ingress deployed"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pods
    log_info "Checking pods status..."
    kubectl get pods -n "$NAMESPACE"
    
    # Check services
    log_info "Checking services..."
    kubectl get services -n "$NAMESPACE"
    
    # Check ingress
    log_info "Checking ingress..."
    kubectl get ingress -n "$NAMESPACE"
    
    # Health check
    log_info "Performing health check..."
    if kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=cyber-llm | grep -q "Running"; then
        log_success "All pods are running"
    else
        log_warning "Some pods may not be running properly"
    fi
}

cleanup() {
    log_info "Cleaning up resources..."
    
    kubectl delete -f src/deployment/k8s/ingress.yaml --ignore-not-found=true
    kubectl delete -f src/deployment/k8s/autoscaling.yaml --ignore-not-found=true
    kubectl delete -f src/deployment/k8s/service.yaml --ignore-not-found=true
    kubectl delete -f src/deployment/k8s/deployment.yaml --ignore-not-found=true
    kubectl delete -f src/deployment/k8s/configmap.yaml --ignore-not-found=true
    kubectl delete -f src/deployment/k8s/storage.yaml --ignore-not-found=true
    kubectl delete -f src/deployment/k8s/rbac.yaml --ignore-not-found=true
    kubectl delete -f src/deployment/k8s/namespace.yaml --ignore-not-found=true
    
    log_success "Cleanup completed"
}

show_help() {
    cat << EOF
Cyber-LLM Kubernetes Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy      Deploy the complete Cyber-LLM application
    cleanup     Remove all deployed resources
    verify      Verify the current deployment status
    help        Show this help message

Options:
    --namespace NAMESPACE       Kubernetes namespace (default: cyber-llm)
    --environment ENVIRONMENT   Environment name (default: production)
    --context CONTEXT          Kubectl context to use
    --dry-run                   Perform a dry run without applying changes

Examples:
    $0 deploy --namespace cyber-llm-prod --environment production
    $0 deploy --dry-run
    $0 cleanup --namespace cyber-llm-dev
    $0 verify

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --context)
            KUBECTL_CONTEXT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        deploy)
            COMMAND="deploy"
            shift
            ;;
        cleanup)
            COMMAND="cleanup"
            shift
            ;;
        verify)
            COMMAND="verify"
            shift
            ;;
        help|--help|-h)
            COMMAND="help"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    case "${COMMAND:-deploy}" in
        deploy)
            log_info "Starting Cyber-LLM deployment to namespace: $NAMESPACE"
            check_prerequisites
            deploy_namespace
            deploy_storage
            deploy_config
            deploy_application
            deploy_autoscaling
            deploy_ingress
            verify_deployment
            log_success "Deployment completed successfully!"
            ;;
        cleanup)
            log_info "Starting cleanup process..."
            check_prerequisites
            cleanup
            ;;
        verify)
            log_info "Verifying deployment..."
            check_prerequisites
            verify_deployment
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Unknown command: ${COMMAND:-}"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
