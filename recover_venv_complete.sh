#!/bin/bash
# VENV Recovery Script - COMPLETE VERSION
# For: CUDA 12.8 / RTX 5090 (Blackwell sm_100)
# Source: Extracted from broken venv llm.BROKEN.2025...0211
#
# Run with: bash recover_venv_complete.sh 2>&1 | tee venv_install.log

set -e  # Exit on error

VENV_NAME="llm"
VENV_PATH="$HOME/venvs/$VENV_NAME"
PYTHON_VERSION="python3"

echo "=== VENV Recovery Script (CUDA 12.8 / Blackwell) ==="
echo "=== Complete package list from broken venv ==="
echo ""

# =============================================================================
# STEP 1: Create new venv
# =============================================================================
echo "[1/8] Creating new virtual environment..."
if [ -d "$VENV_PATH" ]; then
    echo "ERROR: $VENV_PATH already exists!"
    echo "Remove it first: rm -rf $VENV_PATH"
    exit 1
fi

$PYTHON_VERSION -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""

# =============================================================================
# STEP 2: Build tools (MUST be first)
# =============================================================================
echo "[2/8] Installing build tools..."
pip install --upgrade pip==25.3 setuptools==80.9.0 wheel==0.45.1
pip install ninja==1.13.0 packaging==25.0

# =============================================================================
# STEP 3: PyTorch cu128 (MUST be before any CUDA-dependent packages)
# =============================================================================
echo "[3/8] Installing PyTorch for CUDA 12.8..."

# Your original had torch==2.7.1+cu128
# cu128 wheels may be on nightly or need specific index
# Try the nightly index first for cu128:
pip install --pre torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    || pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Verify
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cap[0]}.{cap[1]}')
"
echo ""

# =============================================================================
# STEP 4: Triton (MUST be before flash-attn and other kernel libs)
# =============================================================================
echo "[4/8] Installing Triton..."
pip install triton==3.3.1

python -c "import triton; print(f'Triton: {triton.__version__}')"
echo ""

# =============================================================================
# STEP 5: Flash Attention (NIGHTLY for Blackwell sm_100)
# =============================================================================
echo "[5/8] Installing Flash Attention (nightly for Blackwell)..."

# You said flash needs nightly for Blackwell
# Option 1: Build from source (most reliable for cutting edge)
echo "Building Flash Attention from source..."
pip install flash-attn --no-build-isolation \
    || (
        echo "Pre-built failed, building from GitHub..."
        pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation
    )

python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" || echo "Flash Attention install needs verification"
echo ""

# =============================================================================
# STEP 6: Other CUDA kernel packages
# =============================================================================
echo "[6/8] Installing CUDA kernel packages..."

# Liger Kernels (LinkedIn's fused BF16 kernels)
echo "Installing Liger Kernel..."
pip install liger-kernel==0.6.4

# Lightning Attention
echo "Installing Lightning Attention..."
pip install lightning-attn==0.0.5

# SageAttention
echo "Installing SageAttention..."
pip install sageattention==2.2.0

# bitsandbytes
echo "Installing bitsandbytes..."
pip install bitsandbytes==0.49.0

# DeepSpeed
echo "Installing DeepSpeed..."
DS_BUILD_OPS=1 pip install deepspeed==0.18.3

# Transformer Engine (NVIDIA) - dev build needs special handling
echo "Installing Transformer Engine..."
# Your version was 2.12.0.dev0+eb8e792 - that's a git commit build
# Try the latest release first, or build from source if you need that exact version
pip install transformer-engine[pytorch] \
    || pip install git+https://github.com/NVIDIA/TransformerEngine.git@main

echo ""

# =============================================================================
# STEP 7: All remaining packages (from your broken venv)
# =============================================================================
echo "[7/8] Installing remaining packages..."

# Core ML packages
pip install \
    transformers==4.57.3 \
    datasets==4.4.1 \
    tokenizers==0.22.1 \
    accelerate==1.12.0 \
    huggingface-hub==0.36.0 \
    safetensors==0.7.0 \
    einops==0.8.1 \
    peft==0.18.0 \
    trl==0.26.1 \
    evaluate==0.4.6

# Tokenization
pip install \
    tiktoken==0.12.0 \
    sentencepiece==0.2.1 \
    regex==2025.11.3

# Data processing
pip install \
    numpy==2.3.5 \
    pandas==2.3.3 \
    scipy==1.16.3 \
    pyarrow==21.0.0

# Visualization
pip install \
    matplotlib==3.10.8 \
    pillow==12.0.0

# Logging & monitoring
pip install \
    wandb==0.23.1 \
    tensorboard==2.20.0 \
    tensorboard-data-server==0.7.2 \
    rich==14.2.0 \
    tqdm==4.67.1

# ONNX
pip install \
    onnx==1.20.0 \
    onnx-ir==0.1.13 \
    onnxscript==0.5.7

# Config & CLI
pip install \
    PyYAML==6.0.3 \
    hjson==3.1.0 \
    typer-slim==0.20.0 \
    click==8.3.1 \
    shellingham==1.5.4

# HTTP & networking
pip install \
    requests==2.32.5 \
    urllib3==2.6.2 \
    httpx==0.28.1 \
    httpcore==1.0.9 \
    aiohttp==3.13.2 \
    aiofiles==25.1.0 \
    anyio==4.12.0

# Testing & dev tools
pip install \
    pytest==9.0.2 \
    pylint==4.0.4 \
    isort==7.0.0 \
    astroid==4.0.2

# Utilities
pip install \
    python-dateutil==2.9.0.post0 \
    pytz==2025.2 \
    tzdata==2025.2 \
    six==1.17.0 \
    filelock==3.20.0 \
    platformdirs==4.5.1 \
    humanize==4.15.0 \
    psutil==7.1.3

# Pydantic & typing
pip install \
    pydantic==2.12.5 \
    pydantic_core==2.41.5 \
    typing_extensions==4.15.0 \
    typing-inspection==0.4.2 \
    annotated-types==0.7.0

# Math & symbolic
pip install \
    sympy==1.14.0 \
    mpmath==1.3.0 \
    opt_einsum==3.4.0

# Serialization
pip install \
    protobuf==6.33.2 \
    msgpack==1.1.2 \
    simplejson==3.20.2 \
    dill==0.4.0

# JAX ecosystem (for chex)
pip install \
    chex==0.1.91 \
    etils==1.13.0 \
    ml_dtypes==0.5.4 \
    tensorstore==0.1.80 \
    toolz==1.1.0 \
    treescope==0.1.10

# Git tools
pip install \
    GitPython==3.1.45 \
    gitdb==4.0.12 \
    smmap==5.0.2

# Remaining packages
pip install \
    absl-py==2.3.1 \
    aiohappyeyeballs==2.6.1 \
    aiosignal==1.4.0 \
    attrs==25.4.0 \
    certifi==2025.11.12 \
    charset-normalizer==3.4.4 \
    contourpy==1.3.3 \
    cycler==0.12.1 \
    fonttools==4.61.1 \
    frozenlist==1.8.0 \
    fsspec==2025.10.0 \
    grpcio==1.76.0 \
    h11==0.16.0 \
    hf_transfer==0.1.9 \
    hf-xet==1.2.0 \
    idna==3.11 \
    importlib_metadata==8.7.0 \
    importlib_resources==6.5.2 \
    iniconfig==2.3.0 \
    Jinja2==3.1.6 \
    kiwisolver==1.4.9 \
    Markdown==3.10 \
    markdown-it-py==4.0.0 \
    MarkupSafe==3.0.3 \
    mccabe==0.7.0 \
    mdurl==0.1.2 \
    multidict==6.7.0 \
    multiprocess==0.70.18 \
    nest-asyncio==1.6.0 \
    networkx==3.6.1 \
    nvdlfw_inspect==0.2.2 \
    pluggy==1.6.0 \
    propcache==0.4.1 \
    py-cpuinfo==9.0.0 \
    Pygments==2.19.2 \
    pyparsing==3.2.5 \
    sentry-sdk==2.47.0 \
    tomlkit==0.13.3 \
    Werkzeug==3.1.4 \
    xxhash==3.6.0 \
    yarl==1.22.0 \
    zipp==3.23.0

echo ""

# =============================================================================
# STEP 8: Install HYDRA project
# =============================================================================
echo "[8/8] Installing HYDRA project..."
cd ~/Projects/LLM/HYDRA

if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e ".[dev]" 2>/dev/null || pip install -e .
    echo "HYDRA installed in editable mode"
else
    echo "No setup.py/pyproject.toml found - skipping project install"
fi

echo ""

# =============================================================================
# VERIFICATION
# =============================================================================
echo "=============================================="
echo "=== VERIFICATION ==="
echo "=============================================="
python << 'EOF'
import sys
print(f'Python: {sys.version}')
print()

packages = [
    ('torch', 'PyTorch'),
    ('triton', 'Triton'),
    ('flash_attn', 'Flash Attention'),
    ('liger_kernel', 'Liger Kernel'),
    ('lightning_attn', 'Lightning Attention'),
    ('sageattention', 'SageAttention'),
    ('bitsandbytes', 'bitsandbytes'),
    ('deepspeed', 'DeepSpeed'),
    ('transformer_engine', 'Transformer Engine'),
    ('transformers', 'Transformers'),
    ('datasets', 'Datasets'),
    ('accelerate', 'Accelerate'),
    ('einops', 'einops'),
    ('wandb', 'W&B'),
    ('hydra', 'HYDRA'),
]

for module, name in packages:
    try:
        m = __import__(module)
        version = getattr(m, '__version__', 'installed')
        print(f'✓ {name}: {version}')
    except ImportError as e:
        print(f'✗ {name}: {e}')

# Check CUDA
print()
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA: {torch.version.cuda}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    
    # Quick tensor test
    try:
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x)
        print(f'✓ CUDA tensor ops working')
    except Exception as e:
        print(f'✗ CUDA tensor test failed: {e}')
else:
    print('✗ CUDA not available')
EOF

echo ""
echo "=============================================="
echo "=== Recovery Complete ==="
echo "=============================================="
echo ""
echo "Activate with: source $VENV_PATH/bin/activate"
echo ""
echo "If any packages failed, check venv_install.log for details"
echo ""
echo "Known issues to watch for:"
echo "  - Flash Attention: May need manual build if pre-built wheels don't exist for sm_100"
echo "  - Transformer Engine: Dev version may need: pip install git+https://github.com/NVIDIA/TransformerEngine.git"
echo "  - Some nvidia-* packages come bundled with PyTorch cu128"
