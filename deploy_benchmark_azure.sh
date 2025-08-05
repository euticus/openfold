#!/bin/bash

# FoldForever Azure GPU Benchmark Deployment Script
# This script deploys the comprehensive benchmark to Azure Kubernetes Service

set -e

echo "🚀 FoldForever Azure GPU Benchmark Deployment"
echo "=============================================="

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-protein-folding}"
CLUSTER_NAME="${AZURE_CLUSTER_NAME:-protein-folding-aks}"
NAMESPACE="${KUBERNETES_NAMESPACE:-default}"
BENCHMARK_MODE="${BENCHMARK_MODE:-full}"
NUM_SEQUENCES="${NUM_SEQUENCES:-30}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI not found. Please install Azure CLI."
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check if logged into Azure
    if ! az account show &> /dev/null; then
        log_error "Not logged into Azure. Please run 'az login'."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Connect to AKS cluster
connect_to_cluster() {
    log_info "Connecting to AKS cluster..."
    
    az aks get-credentials \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" \
        --overwrite-existing
    
    # Verify connection
    if kubectl cluster-info &> /dev/null; then
        log_success "Connected to AKS cluster: $CLUSTER_NAME"
    else
        log_error "Failed to connect to AKS cluster"
        exit 1
    fi
}

# Check cluster resources
check_cluster_resources() {
    log_info "Checking cluster resources..."
    
    # Check nodes
    echo "Available nodes:"
    kubectl get nodes -o wide
    
    # Check for GPU nodes
    GPU_NODES=$(kubectl get nodes -l accelerator=nvidia-tesla-t4 --no-headers 2>/dev/null | wc -l || echo "0")
    if [ "$GPU_NODES" -eq 0 ]; then
        log_warning "No GPU nodes found. Benchmark will run on CPU."
        log_info "To add GPU nodes, run:"
        log_info "az aks nodepool add --resource-group $RESOURCE_GROUP --cluster-name $CLUSTER_NAME --name gpu --node-count 1 --node-vm-size Standard_NC4as_T4_v3"
    else
        log_success "Found $GPU_NODES GPU nodes"
    fi
}

# Create benchmark ConfigMap with actual script content
create_benchmark_configmap() {
    log_info "Creating benchmark script ConfigMap..."
    
    # Check if benchmark script exists
    if [ ! -f "benchmark_casp14_foldforever_vs_baselines.py" ]; then
        log_error "Benchmark script not found: benchmark_casp14_foldforever_vs_baselines.py"
        exit 1
    fi
    
    # Create ConfigMap with the actual benchmark script
    kubectl create configmap benchmark-script \
        --from-file=benchmark_casp14_foldforever_vs_baselines.py \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Benchmark script ConfigMap created"
}

# Deploy benchmark
deploy_benchmark() {
    log_info "Deploying benchmark to Kubernetes..."
    
    # Apply the deployment
    kubectl apply -f k8s/benchmark-deployment.yaml -n "$NAMESPACE"
    
    log_success "Benchmark deployment created"
}

# Monitor benchmark progress
monitor_benchmark() {
    log_info "Monitoring benchmark progress..."
    
    echo "Waiting for benchmark job to start..."
    kubectl wait --for=condition=ready pod -l app=foldforever-benchmark -n "$NAMESPACE" --timeout=300s || {
        log_warning "Benchmark pod not ready within 5 minutes, checking status..."
        kubectl get pods -l app=foldforever-benchmark -n "$NAMESPACE"
        kubectl describe pods -l app=foldforever-benchmark -n "$NAMESPACE"
    }
    
    # Follow logs
    echo "Following benchmark logs..."
    kubectl logs -f job/foldforever-benchmark -n "$NAMESPACE" || {
        log_warning "Failed to follow logs, showing current logs..."
        kubectl logs job/foldforever-benchmark -n "$NAMESPACE" --tail=50
    }
}

# Collect results
collect_results() {
    log_info "Collecting benchmark results..."
    
    # Wait for job completion
    kubectl wait --for=condition=complete job/foldforever-benchmark -n "$NAMESPACE" --timeout=3600s || {
        log_warning "Benchmark job did not complete within 1 hour"
        kubectl get jobs -n "$NAMESPACE"
        kubectl describe job/foldforever-benchmark -n "$NAMESPACE"
    }
    
    # Get the pod name
    POD_NAME=$(kubectl get pods -l app=foldforever-benchmark -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -n "$POD_NAME" ]; then
        log_info "Copying results from pod: $POD_NAME"
        
        # Create local results directory
        mkdir -p azure_benchmark_results
        
        # Copy results
        kubectl cp "$NAMESPACE/$POD_NAME:/results" ./azure_benchmark_results/ || {
            log_warning "Failed to copy results, trying alternative method..."
            kubectl exec "$POD_NAME" -n "$NAMESPACE" -- tar czf - -C /results . | tar xzf - -C ./azure_benchmark_results/
        }
        
        log_success "Results copied to ./azure_benchmark_results/"
        
        # Display summary
        if [ -f "./azure_benchmark_results/benchmark_report.md" ]; then
            echo ""
            echo "📊 BENCHMARK SUMMARY:"
            echo "===================="
            head -30 ./azure_benchmark_results/benchmark_report.md
        fi
        
    else
        log_error "Could not find benchmark pod"
    fi
}

# Cleanup resources
cleanup() {
    log_info "Cleaning up resources..."
    
    kubectl delete job foldforever-benchmark -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete job results-collector -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete configmap benchmark-script -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete configmap benchmark-config -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete service benchmark-results -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    echo "Configuration:"
    echo "  Resource Group: $RESOURCE_GROUP"
    echo "  Cluster Name: $CLUSTER_NAME"
    echo "  Namespace: $NAMESPACE"
    echo "  Benchmark Mode: $BENCHMARK_MODE"
    echo "  Number of Sequences: $NUM_SEQUENCES"
    echo ""
    
    check_prerequisites
    connect_to_cluster
    check_cluster_resources
    
    # Ask for confirmation
    read -p "Do you want to proceed with the benchmark deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled"
        exit 0
    fi
    
    create_benchmark_configmap
    deploy_benchmark
    monitor_benchmark
    collect_results
    
    # Ask about cleanup
    echo ""
    read -p "Do you want to clean up the deployment resources? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    else
        log_info "Resources left running. Clean up manually with: kubectl delete -f k8s/benchmark-deployment.yaml"
    fi
    
    log_success "Benchmark deployment completed!"
    echo ""
    echo "📁 Results are available in: ./azure_benchmark_results/"
    echo "📊 View the full report: ./azure_benchmark_results/benchmark_report.md"
    echo "📈 View plots: ./azure_benchmark_results/plots/"
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"
