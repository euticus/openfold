#!/bin/bash
set -e

# OdinFold Azure A100 Cluster Setup Script
# Optimized for A100 GPUs with maximum performance

echo "🔥 Setting up OdinFold on Azure A100 Cluster"
echo "============================================="

# A100-optimized configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-odinfold-a100-rg}"
CLUSTER_NAME="${AZURE_CLUSTER_NAME:-odinfold-a100-cluster}"
LOCATION="${AZURE_LOCATION:-eastus}"  # Best A100 availability
SYSTEM_NODE_COUNT="${SYSTEM_NODE_COUNT:-1}"
GPU_NODE_COUNT="${GPU_NODE_COUNT:-0}"  # Start with 0 A100 nodes (expensive!)
SYSTEM_VM_SIZE="${SYSTEM_VM_SIZE:-Standard_D4s_v3}"  # Slightly bigger for A100 workloads
GPU_VM_SIZE="${GPU_VM_SIZE:-Standard_NC24ads_A100_v4}"  # 1x A100 80GB
MAX_PODS="${MAX_PODS:-30}"

echo "🚀 A100 Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Cluster Name: $CLUSTER_NAME"
echo "  Location: $LOCATION"
echo "  System Nodes: $SYSTEM_NODE_COUNT x $SYSTEM_VM_SIZE"
echo "  A100 Nodes: $GPU_NODE_COUNT x $GPU_VM_SIZE"
echo ""
echo "💰 Cost Estimates (per hour):"
echo "  System Node: ~\$0.192/hr"
echo "  A100 Node: ~\$3.67/hr (80GB VRAM!)"
echo "  Total when idle: ~\$0.192/hr"
echo "  Total with 1 A100: ~\$3.86/hr"
echo ""

# Check Azure login
echo "🔍 Checking Azure login status..."
if ! az account show >/dev/null 2>&1; then
    echo "❌ Not logged in to Azure. Please run: az login"
    exit 1
fi

SUBSCRIPTION=$(az account show --query name -o tsv)
echo "✅ Logged in to Azure subscription: $SUBSCRIPTION"

# Check A100 quota
echo "🎯 Checking A100 quota in $LOCATION..."
QUOTA_INFO=$(az vm list-usage --location $LOCATION --query "[?contains(name.value, 'standardNCADSA100v4Family')]" -o table)
echo "$QUOTA_INFO"
echo ""

# Create resource group
echo "📦 Creating resource group..."
if az group show --name $RESOURCE_GROUP >/dev/null 2>&1; then
    echo "✅ Resource group already exists: $RESOURCE_GROUP"
else
    az group create --name $RESOURCE_GROUP --location $LOCATION
    echo "✅ Created resource group: $RESOURCE_GROUP"
fi
echo ""

# Create AKS cluster optimized for A100
echo "🚀 Creating AKS cluster optimized for A100..."
if az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME >/dev/null 2>&1; then
    echo "✅ Cluster already exists: $CLUSTER_NAME"
    
    POWER_STATE=$(az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --query "powerState.code" -o tsv)
    echo "   Power State: $POWER_STATE"
    
    if [ "$POWER_STATE" != "Running" ]; then
        echo "🔄 Starting cluster..."
        az aks start --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME
    fi
else
    echo "Creating new AKS cluster with A100 support..."
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
        --max-count 5 \
        --node-osdisk-size 128 \
        --kubernetes-version 1.30.12 \
        --nodepool-name systempool \
        --nodepool-labels pool=system \
        --network-plugin azure \
        --network-policy azure
    
    echo "✅ Created AKS cluster: $CLUSTER_NAME"
    
    # Add A100 GPU node pool
    echo "🔥 Adding A100 GPU node pool..."
    az aks nodepool add \
        --resource-group $RESOURCE_GROUP \
        --cluster-name $CLUSTER_NAME \
        --name a100pool \
        --node-count $GPU_NODE_COUNT \
        --node-vm-size $GPU_VM_SIZE \
        --max-pods $MAX_PODS \
        --enable-cluster-autoscaler \
        --min-count 0 \
        --max-count 3 \
        --node-osdisk-size 512 \
        --node-taints nvidia.com/gpu=true:NoSchedule \
        --labels pool=gpu,accelerator=nvidia_tesla_a100,gpu_memory=80gb \
        --zones 1 2 3
    
    echo "✅ Added A100 GPU node pool"
fi
echo ""

# Get cluster credentials
echo "🔑 Getting cluster credentials..."
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing
echo "✅ Cluster credentials configured"
echo ""

# Install NVIDIA GPU operator optimized for A100
echo "🎯 Installing NVIDIA GPU operator for A100..."
kubectl create namespace gpu-operator --dry-run=client -o yaml | kubectl apply -f -

# Add NVIDIA Helm repository
helm repo add nvidia https://nvidia.github.io/gpu-operator || true
helm repo update

# Install GPU operator with A100 optimizations
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
        --set migManager.enabled=true \
        --set operator.defaultRuntime=containerd \
        --wait --timeout=900s
    
    echo "✅ GPU operator installed with A100 support"
fi
echo ""

# Create OdinFold namespace with A100 optimizations
echo "📦 Creating OdinFold namespace..."
kubectl create namespace odinfold --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace odinfold app=odinfold gpu=a100 --overwrite
echo "✅ OdinFold namespace created"
echo ""

# Wait for A100 nodes to be ready
echo "⏳ Waiting for nodes to be ready..."
kubectl wait --for=condition=ready nodes --all --timeout=600s
echo ""

# Check A100 availability
echo "🔍 Checking A100 GPU availability..."
kubectl get nodes -o wide
echo ""

echo "A100 GPU Details:"
kubectl describe nodes | grep -A 10 -B 5 "nvidia.com/gpu" || echo "No GPU resources found yet"
echo ""

# Deploy A100 test pod
echo "🧪 Deploying A100 test pod..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: a100-test
  namespace: odinfold
  labels:
    app: odinfold
    component: a100-test
spec:
  restartPolicy: Never
  containers:
  - name: a100-test
    image: nvidia/cuda:11.8-runtime-ubuntu20.04
    command: ["/bin/bash"]
    args:
    - -c
    - |
      echo "🔥 A100 GPU Test Results:"
      nvidia-smi
      echo ""
      echo "GPU Memory Info:"
      nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
      echo ""
      echo "CUDA Compute Capability:"
      nvidia-smi --query-gpu=compute_cap --format=csv
      echo ""
      echo "✅ A100 test completed"
      sleep 120
    resources:
      requests:
        nvidia.com/gpu: 1
        memory: "16Gi"
        cpu: "4"
      limits:
        nvidia.com/gpu: 1
        memory: "32Gi"
        cpu: "8"
  nodeSelector:
    accelerator: nvidia_tesla_a100
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
EOF

# Wait for test pod and show results
echo "⏳ Waiting for A100 test pod..."
kubectl wait --for=condition=ready pod/a100-test -n odinfold --timeout=600s || true

echo "📊 A100 Test Results:"
kubectl logs a100-test -n odinfold || echo "A100 test pod not ready yet"
echo ""

# Cleanup test pod
kubectl delete pod a100-test -n odinfold --ignore-not-found=true

# Create A100-optimized OdinFold ConfigMap
echo "⚙️ Creating A100-optimized OdinFold configuration..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: odinfold-a100-config
  namespace: odinfold
  labels:
    app: odinfold
    gpu: a100
data:
  config.yaml: |
    odinfold:
      engine: "FoldForever"
      version: "1.0.0"
      gpu:
        enabled: true
        type: "A100"
        memory_gb: 80
        compute_capability: "8.0"
        tensor_cores: true
        mixed_precision: true
      model:
        batch_size: 8  # Larger batches for A100
        sequence_length: 2048  # Longer sequences
        quantization: "fp16"  # A100 optimized
      benchmark:
        tm_threshold: 0.70  # Higher target for A100
        runtime_threshold: 3.0  # Faster with A100
        memory_threshold: 16.0  # More memory available
      azure:
        resource_group: "$RESOURCE_GROUP"
        cluster_name: "$CLUSTER_NAME"
        location: "$LOCATION"
        vm_size: "$GPU_VM_SIZE"
        cost_per_hour: 3.67
EOF
echo "✅ A100-optimized configuration created"
echo ""

# Show cluster status
echo "📊 A100 Cluster Status:"
echo "======================="
kubectl get nodes -o wide
echo ""
kubectl get pods -n odinfold
echo ""
kubectl get services -n odinfold
echo ""

echo "🔥 OdinFold A100 Cluster Setup Complete!"
echo ""
echo "💰 Cost Management:"
echo "- Current cost: ~$0.192/hr (system nodes only)"
echo "- With 1 A100: ~$3.86/hr"
echo "- A100 nodes auto-scale from 0-3 based on demand"
echo ""
echo "🚀 Next Steps:"
echo "1. Deploy OdinFold: kubectl apply -f k8s/odinfold-deployment.yaml"
echo "2. Scale A100 nodes: az aks nodepool scale --name a100pool --node-count 1"
echo "3. Run benchmarks: kubectl apply -f .github/workflows/azure-gpu-deploy.yml"
echo "4. Monitor costs: az consumption usage list"
echo ""
echo "🎯 A100 Commands:"
echo "- Check A100 nodes: kubectl get nodes -l accelerator=nvidia-tesla-a100"
echo "- Scale A100 pool: az aks nodepool scale --resource-group $RESOURCE_GROUP --cluster-name $CLUSTER_NAME --name a100pool --node-count <count>"
echo "- View A100 usage: kubectl top nodes"
echo "- A100 pod logs: kubectl logs -f <pod-name> -n odinfold"
echo ""
echo "⚡ Your A100 cluster is ready to power FoldForever!"
