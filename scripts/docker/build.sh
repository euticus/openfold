#!/bin/bash
set -e

# OdinFold Docker Build Script
# Builds all Docker images for different environments

echo "🐳 Building OdinFold Docker Images"
echo "=================================="

# Configuration
IMAGE_NAME="odinfold"
TAG="${1:-latest}"
REGISTRY="${DOCKER_REGISTRY:-}"

if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME"
else
    FULL_IMAGE_NAME="$IMAGE_NAME"
fi

echo "Image name: $FULL_IMAGE_NAME"
echo "Tag: $TAG"
echo ""

# Build CPU image
echo "🔧 Building CPU image..."
docker build \
    --target cpu \
    --tag "$FULL_IMAGE_NAME:cpu-$TAG" \
    --tag "$FULL_IMAGE_NAME:cpu-latest" \
    .

echo "✅ CPU image built: $FULL_IMAGE_NAME:cpu-$TAG"
echo ""

# Build GPU image
echo "🔧 Building GPU image..."
docker build \
    --target gpu \
    --tag "$FULL_IMAGE_NAME:gpu-$TAG" \
    --tag "$FULL_IMAGE_NAME:gpu-latest" \
    .

echo "✅ GPU image built: $FULL_IMAGE_NAME:gpu-$TAG"
echo ""

# Build benchmark image
echo "🔧 Building benchmark image..."
docker build \
    --target benchmark \
    --tag "$FULL_IMAGE_NAME:benchmark-$TAG" \
    --tag "$FULL_IMAGE_NAME:benchmark-latest" \
    .

echo "✅ Benchmark image built: $FULL_IMAGE_NAME:benchmark-$TAG"
echo ""

# Build production image
echo "🔧 Building production image..."
docker build \
    --target production \
    --tag "$FULL_IMAGE_NAME:production-$TAG" \
    --tag "$FULL_IMAGE_NAME:production-latest" \
    --tag "$FULL_IMAGE_NAME:$TAG" \
    --tag "$FULL_IMAGE_NAME:latest" \
    .

echo "✅ Production image built: $FULL_IMAGE_NAME:production-$TAG"
echo ""

# Show built images
echo "📊 Built images:"
docker images | grep "$IMAGE_NAME" | head -10

echo ""
echo "🚀 Build complete! Available images:"
echo "  CPU:        $FULL_IMAGE_NAME:cpu-$TAG"
echo "  GPU:        $FULL_IMAGE_NAME:gpu-$TAG"
echo "  Benchmark:  $FULL_IMAGE_NAME:benchmark-$TAG"
echo "  Production: $FULL_IMAGE_NAME:production-$TAG"
echo ""

# Optional: Push to registry
if [ "$2" = "--push" ] && [ -n "$REGISTRY" ]; then
    echo "📤 Pushing images to registry..."
    
    docker push "$FULL_IMAGE_NAME:cpu-$TAG"
    docker push "$FULL_IMAGE_NAME:cpu-latest"
    
    docker push "$FULL_IMAGE_NAME:gpu-$TAG"
    docker push "$FULL_IMAGE_NAME:gpu-latest"
    
    docker push "$FULL_IMAGE_NAME:benchmark-$TAG"
    docker push "$FULL_IMAGE_NAME:benchmark-latest"
    
    docker push "$FULL_IMAGE_NAME:production-$TAG"
    docker push "$FULL_IMAGE_NAME:production-latest"
    docker push "$FULL_IMAGE_NAME:$TAG"
    docker push "$FULL_IMAGE_NAME:latest"
    
    echo "✅ Images pushed to registry"
fi

echo "🎉 Docker build script complete!"
