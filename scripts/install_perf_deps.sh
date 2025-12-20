#!/usr/bin/env bash
# HYDRA Performance Optimization Setup Script
# ============================================
# This script installs all optional performance packages for maximum throughput.
#
# Usage:
#   ./scripts/install_perf_deps.sh [--all|--liger|--flash|--te]
#
# Options:
#   --all    Install all performance packages (recommended)
#   --liger  Install Liger Kernels only (best memory savings)
#   --flash  Install Flash Attention only (faster attention)
#   --te     Install Transformer Engine only (FP8 for Hopper+)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 HYDRA Performance Dependencies Installer"
echo "=========================================="

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo -e "${YELLOW}Warning: nvcc not found. CUDA version detection failed.${NC}"
    CUDA_VERSION="12.0"  # Assume modern CUDA
fi

# Detect GPU compute capability
GPU_INFO=$(python3 -c "import torch; print(torch.cuda.get_device_capability())" 2>/dev/null || echo "(0, 0)")
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name())" 2>/dev/null || echo "Unknown")
echo "Detected GPU: $GPU_NAME ($GPU_INFO)"

install_liger() {
    echo -e "\n${GREEN}Installing Liger Kernels...${NC}"
    echo "Benefits: 50-60% memory reduction, ~20% speedup"
    pip install liger-kernel
    echo -e "${GREEN}✅ Liger Kernels installed${NC}"
}

install_flash_attn() {
    echo -e "\n${GREEN}Installing Flash Attention...${NC}"
    echo "Benefits: Faster attention, O(n) memory for sequence length"
    
    # Flash Attention requires special build flags
    pip install flash-attn --no-build-isolation
    echo -e "${GREEN}✅ Flash Attention installed${NC}"
}

install_transformer_engine() {
    echo -e "\n${GREEN}Installing Transformer Engine...${NC}"
    echo "Benefits: FP8 training for 2x TFLOPS (Hopper+ GPUs)"
    
    # Check if GPU supports FP8 (sm_89+)
    SM_VERSION=$(echo $GPU_INFO | sed 's/[^0-9]//g')
    if [ "$SM_VERSION" -lt "90" ]; then
        echo -e "${YELLOW}Warning: Your GPU ($GPU_NAME) may not support FP8.${NC}"
        echo "Transformer Engine will still work in BF16 mode."
    fi
    
    pip install transformer-engine[pytorch]
    echo -e "${GREEN}✅ Transformer Engine installed${NC}"
}

install_xformers() {
    echo -e "\n${GREEN}Installing xFormers (fallback)...${NC}"
    pip install xformers
    echo -e "${GREEN}✅ xFormers installed${NC}"
}

install_all() {
    install_liger
    install_flash_attn
    install_transformer_engine
}

# Parse arguments
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage: $0 [--all|--liger|--flash|--te|--xformers]"
    echo ""
    echo "Options:"
    echo "  --all      Install all performance packages (recommended)"
    echo "  --liger    Install Liger Kernels (best memory savings)"
    echo "  --flash    Install Flash Attention (faster attention)"
    echo "  --te       Install Transformer Engine (FP8 for Hopper+)"
    echo "  --xformers Install xFormers (fallback if Flash doesn't work)"
    echo ""
    echo "Recommendation for your GPU ($GPU_NAME):"
    
    SM_VERSION=$(echo $GPU_INFO | sed 's/[^0-9,]//g' | cut -d',' -f1)
    if [ "$SM_VERSION" -ge "90" ]; then
        echo -e "  ${GREEN}→ Run: $0 --all${NC}"
        echo "  Your GPU supports FP8! Install everything for max performance."
    else
        echo -e "  ${GREEN}→ Run: $0 --liger --flash${NC}"
        echo "  Liger + Flash Attention is optimal for your GPU."
    fi
    exit 0
fi

for arg in "$@"; do
    case $arg in
        --all)
            install_all
            ;;
        --liger)
            install_liger
            ;;
        --flash)
            install_flash_attn
            ;;
        --te)
            install_transformer_engine
            ;;
        --xformers)
            install_xformers
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Installation complete!${NC}"
echo "=========================================="
echo ""
echo "Verify your setup with:"
echo "  python -c 'from hydra.kernels import print_perf_status; print_perf_status()'"
echo ""
