#!/bin/bash

# Azure A100 GPU Quota Request Script
# This script helps you request A100 GPU quota for protein folding research

set -e

echo "🚀 Azure A100 GPU Quota Request"
echo "================================"

# Configuration
SUBSCRIPTION_ID="39bd5d25-e94b-4a97-a192-f0781446d526"
REGIONS=("eastus" "westus2")
QUOTA_REQUESTS=(
    "Standard NCADS A100 v4 Family vCPUs:80"
    "Standard NCadsH100v5 Family vCPUs:40"
    "Total Regional vCPUs:150"
)

# Colors for output
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

# Check if logged in
if ! az account show &> /dev/null; then
    echo "❌ Not logged into Azure. Please run 'az login' first."
    exit 1
fi

# Display current subscription
CURRENT_SUB=$(az account show --query name -o tsv)
log_info "Current subscription: $CURRENT_SUB"

echo ""
echo "📋 BUSINESS JUSTIFICATION FOR A100 REQUEST:"
echo "==========================================="
cat << EOF

RESEARCH PROJECT: FoldForever Protein Folding Benchmark
INSTITUTION: Research/Academic Use
DURATION: 3-6 months intensive research

TECHNICAL REQUIREMENTS:
- Large protein sequences (up to 900 amino acids)
- GPU memory requirements: 40-80GB for large sequences
- Comparative analysis against AlphaFold2, ESMFold, OpenFold
- Real CASP14 and CAMEO dataset evaluation
- Multiple model training and inference experiments

EXPECTED USAGE:
- 1-2 A100 nodes running 8-12 hours daily
- Estimated monthly cost: $1,000-2,000
- Non-production research workload
- Results will be published in academic papers

JUSTIFICATION:
- A100 80GB required for large protein sequences (>500 AA)
- Current quota insufficient for research requirements
- Time-sensitive research with publication deadlines
- Will contribute to open-source protein folding research

EOF

echo ""
echo "🎯 QUOTA REQUESTS:"
echo "=================="
for request in "${QUOTA_REQUESTS[@]}"; do
    echo "  • $request"
done

echo ""
echo "🌍 REGIONS:"
echo "==========="
for region in "${REGIONS[@]}"; do
    echo "  • $region"
done

echo ""
read -p "Do you want to proceed with creating support tickets for these quota requests? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Quota request cancelled"
    exit 0
fi

# Function to create support ticket
create_support_ticket() {
    local region=$1
    local quota_type=$2
    local quota_limit=$3
    
    log_info "Creating support ticket for $quota_type in $region..."
    
    # Create support ticket
    TICKET_NAME="A100-Quota-Request-${region}-$(date +%Y%m%d-%H%M%S)"
    
    DESCRIPTION="Request quota increase for protein folding research using FoldForever benchmark.

QUOTA REQUEST:
- Region: $region
- Quota Type: $quota_type
- Requested Limit: $quota_limit

BUSINESS JUSTIFICATION:
Research project benchmarking protein folding algorithms (FoldForever) against established baselines (AlphaFold2, ESMFold, OpenFold) using real CASP14 and CAMEO datasets.

TECHNICAL REQUIREMENTS:
- Large protein sequences (up to 900 amino acids)
- GPU memory requirements: 40-80GB for large sequences
- Comparative analysis requiring consistent hardware across models
- Research timeline: 3-6 months of intensive benchmarking

EXPECTED USAGE:
- 1-2 GPU nodes running 8-12 hours daily
- Estimated monthly cost: $1,000-2,000
- Non-production research workload
- Results will be published in academic research

Please approve this quota increase to enable critical protein folding research."

    # Note: The actual support ticket creation via CLI requires complex parameters
    # For now, we'll provide the information needed for manual creation
    
    echo ""
    log_warning "Azure CLI support ticket creation requires additional setup."
    log_info "Please create the support ticket manually using the information above."
    log_info "Ticket Name: $TICKET_NAME"
    echo ""
}

# Alternative: Direct quota request (if available)
request_quota_increase() {
    local region=$1
    
    log_info "Checking current quota in $region..."
    
    # Check current quota
    az vm list-usage --location "$region" --query "[?contains(name.value, 'NCADS') || contains(name.value, 'Total Regional')]" --output table
    
    echo ""
    log_warning "Direct quota increase via CLI not available for all quota types."
    log_info "Please use Azure Portal for quota requests: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas"
}

# Main execution
echo ""
log_info "Processing quota requests..."

for region in "${REGIONS[@]}"; do
    echo ""
    log_info "=== Processing region: $region ==="
    request_quota_increase "$region"
done

echo ""
log_success "Quota request information prepared!"
echo ""
echo "📋 NEXT STEPS:"
echo "=============="
echo "1. Go to Azure Portal: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas"
echo "2. Click 'Request increase'"
echo "3. Select 'Compute-VM (cores-vCPUs) subscription limit increases'"
echo "4. Choose regions: eastus, westus2"
echo "5. Request the quotas listed above"
echo "6. Use the business justification provided in this script"
echo ""
echo "⏱️  EXPECTED TIMELINE:"
echo "====================="
echo "• Standard requests: 1-3 business days"
echo "• Large requests (>100 cores): 3-5 business days"
echo "• Academic/Research requests: Often expedited"
echo ""
echo "📞 ESCALATION OPTIONS:"
echo "====================="
echo "• Contact Azure Support directly"
echo "• Mention 'Academic Research' for priority"
echo "• Reference protein folding / AI research"
echo ""

log_success "A100 quota request process completed!"
