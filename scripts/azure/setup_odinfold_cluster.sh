#!/bin/bash
set -e

# OdinFold Azure GPU Cluster Setup Script
# This script sets up an Azure AKS cluster optimized for OdinFold workloads

echo "⚡ Setting up OdinFold on Azure GPU Cluster"
echo "=========================================="

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-odinfold-rg}"
CLUSTER_NAME="${AZURE_CLUSTER_NAME:-odinfold-cluster}"
LOCATION="${AZURE_LOCATION:-eastus}"  # eastus has best GPU availability
SYSTEM_NODE_COUNT="${SYSTEM_NODE_COUNT:-1}"
GPU_NODE_COUNT="${GPU_NODE_COUNT:-0}"  # Start with 0 GPU nodes to save cost
SYSTEM_VM_SIZE="${SYSTEM_VM_SIZE:-Standard_D2s_v3}"  # Cheap system nodes
GPU_VM_SIZE="${GPU_VM_SIZE:-Standard_NC24ads_A100_v4}"  # A100 GPU (80GB VRAM) ⚡ BEAST MODE
# Azure A100 GPU options:
# Standard_NC24ads_A100_v4  - 1x A100 (80GB VRAM) ~$3.67/hr ⭐ RECOMMENDED
# Standard_ND96amsr_A100_v4 - 8x A100 (40GB each) ~$27.20/hr (320GB total)
# Standard_ND96asr_v4       - 8x A100 (80GB each) ~$32.77/hr (640GB total!) 🔥
# Standard_NC8as_T4_v3      - 1x T4 (16GB VRAM) ~$0.526/hr (budget option)
MAX_PODS="${MAX_PODS:-30}"

echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Cluster Name: $CLUSTER_NAME"
echo "  Location: $LOCATION"
echo "  System Nodes: $SYSTEM_NODE_COUNT x $SYSTEM_VM_SIZE"
echo "  GPU Nodes: $GPU_NODE_COUNT x $GPU_VM_SIZE"
echo ""

# Check if logged in to Azure
echo "🔍 Checking Azure login status..."
if ! az account show >/dev/null 2>&1; then
    echo "❌ Not logged in to Azure. Please run: az login"
    exit 1
fi

SUBSCRIPTION=$(az account show --query name -o tsv)
echo "✅ Logged in to Azure subscription: $SUBSCRIPTION"
echo ""

# Create resource group if it doesn't exist
echo "📦 Creating resource group..."
if az group show --name $RESOURCE_GROUP >/dev/null 2>&1; then
    echo "✅ Resource group already exists: $RESOURCE_GROUP"
else
    az group create --name $RESOURCE_GROUP --location $LOCATION
    echo "✅ Created resource group: $RESOURCE_GROUP"
fi
echo ""

# Create AKS cluster with GPU support
echo "🚀 Creating AKS cluster with GPU support..."
if az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME >/dev/null 2>&1; then
    echo "✅ Cluster already exists: $CLUSTER_NAME"
    
    # Check if it's running
    POWER_STATE=$(az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --query "powerState.code" -o tsv)
    echo "   Power State: $POWER_STATE"
    
    if [ "$POWER_STATE" != "Running" ]; then
        echo "🔄 Starting cluster..."
        az aks start --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME
    fi
else
    echo "Creating new AKS cluster with system node pool..."
    az aks create \
        --resource-group $RESOURCE_GROUP \
        --name $CLUSTER_NAME \
        --location $LOCATION \
        --node-count $SYSTEM_NODE_COUNT \
        --node-vm-size $SYSTEM_VM_SIZE \
        --max-pods $MAX_PODS \
        --enable-addons monitoring \
        --generate-ssh-keys \
        --enable-cluster-autoscaler \
        --min-count 1 \
        --max-count 3 \
        --node-osdisk-size 30 \
        --kubernetes-version 1.27.3 \
        --nodepool-name systempool \
        --nodepool-labels pool=system

    echo "✅ Created AKS cluster: $CLUSTER_NAME"

    # Add GPU node pool
    echo "🎯 Adding GPU node pool..."
    az aks nodepool add \
        --resource-group $RESOURCE_GROUP \
        --cluster-name $CLUSTER_NAME \
        --name gpupool \
        --node-count $GPU_NODE_COUNT \
        --node-vm-size $GPU_VM_SIZE \
        --max-pods $MAX_PODS \
        --enable-cluster-autoscaler \
        --min-count 0 \
        --max-count 3 \
        --node-osdisk-size 100 \
        --node-taints nvidia.com/gpu=true:NoSchedule \
        --labels pool=gpu,accelerator=nvidia-tesla-a100

    echo "✅ Added GPU node pool"
fi
echo ""

# Get cluster credentials
echo "🔑 Getting cluster credentials..."
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing
echo "✅ Cluster credentials configured"
echo ""

# Install NVIDIA GPU operator
echo "🎯 Installing NVIDIA GPU operator..."
kubectl create namespace gpu-operator --dry-run=client -o yaml | kubectl apply -f -

# Add NVIDIA Helm repository
helm repo add nvidia https://nvidia.github.io/gpu-operator || true
helm repo update

# Install GPU operator
if helm list -n gpu-operator | grep -q gpu-operator; then
    echo "✅ GPU operator already installed"
else
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --set driver.enabled=true \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=true \
        --set nodeStatusExporter.enabled=true \
        --set gfd.enabled=true \
        --set migManager.enabled=false \
        --wait --timeout=600s
    
    echo "✅ GPU operator installed"
fi
echo ""

# Create OdinFold namespace
echo "📦 Creating OdinFold namespace..."
kubectl create namespace odinfold --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace odinfold app=odinfold --overwrite
echo "✅ OdinFold namespace created"
echo ""

# Create OdinFold service account
echo "👤 Creating OdinFold service account..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: odinfold-sa
  namespace: odinfold
  labels:
    app: odinfold
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: odinfold-role
  labels:
    app: odinfold
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: odinfold-binding
  labels:
    app: odinfold
subjects:
- kind: ServiceAccount
  name: odinfold-sa
  namespace: odinfold
roleRef:
  kind: ClusterRole
  name: odinfold-role
  apiGroup: rbac.authorization.k8s.io
EOF
echo "✅ OdinFold service account created"
echo ""

# Wait for GPU nodes to be ready
echo "⏳ Waiting for GPU nodes to be ready..."
kubectl wait --for=condition=ready nodes --all --timeout=300s
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
kubectl get nodes -o wide
echo ""

echo "GPU Node Details:"
kubectl describe nodes | grep -A 5 -B 5 "nvidia.com/gpu" || echo "No GPU resources found yet"
echo ""

# Deploy GPU test pod
echo "🧪 Deploying GPU test pod..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
  namespace: odinfold
  labels:
    app: odinfold
    component: gpu-test
spec:
  restartPolicy: Never
  containers:
  - name: gpu-test
    image: nvidia/cuda:11.8-runtime-ubuntu20.04
    command: ["/bin/bash"]
    args:
    - -c
    - |
      echo "🎯 GPU Test Results:"
      nvidia-smi
      echo "✅ GPU test completed"
      sleep 60
    resources:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
EOF

# Wait for test pod and show results
echo "⏳ Waiting for GPU test pod..."
kubectl wait --for=condition=ready pod/gpu-test -n odinfold --timeout=300s || true

echo "📊 GPU Test Results:"
kubectl logs gpu-test -n odinfold || echo "GPU test pod not ready yet"
echo ""

# Cleanup test pod
kubectl delete pod gpu-test -n odinfold --ignore-not-found=true

# Create OdinFold ConfigMap
echo "⚙️ Creating OdinFold configuration..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: odinfold-config
  namespace: odinfold
  labels:
    app: odinfold
data:
  config.yaml: |
    odinfold:
      engine: "FoldForever"
      version: "1.0.0"
      gpu:
        enabled: true
        memory_limit: "16Gi"
      benchmark:
        tm_threshold: 0.66
        runtime_threshold: 5.5
        memory_threshold: 8.0
      azure:
        resource_group: "$RESOURCE_GROUP"
        cluster_name: "$CLUSTER_NAME"
        location: "$LOCATION"
EOF
echo "✅ OdinFold configuration created"
echo ""

# Show cluster status
echo "📊 Cluster Status:"
echo "=================="
kubectl get nodes -o wide
echo ""
kubectl get pods -n odinfold
echo ""
kubectl get services -n odinfold
echo ""

echo "🎉 OdinFold Azure GPU Cluster Setup Complete!"
echo ""
echo "Next Steps:"
echo "1. Run benchmark: kubectl apply -f .github/workflows/azure-gpu-deploy.yml"
echo "2. Scale cluster: az aks scale --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --node-count <count>"
echo "3. Monitor costs: az consumption usage list"
echo "4. Deploy FoldForever: kubectl apply -f k8s/foldforever/"
echo ""
echo "Useful Commands:"
echo "- Check GPU nodes: kubectl get nodes -l accelerator=nvidia-tesla-v100"
echo "- View logs: kubectl logs -f <pod-name> -n odinfold"
echo "- Scale down: az aks scale --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --node-count 0"
echo "- Delete cluster: az aks delete --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME"
