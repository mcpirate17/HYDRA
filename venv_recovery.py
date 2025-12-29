#!/usr/bin/env python3
"""
VENV Recovery Discovery Script
Analyzes a broken venv and project to help rebuild dependencies.
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# =============================================================================
# CONFIGURATION - Update these paths for your system
# =============================================================================
BROKEN_VENV_PATH = Path.home() / "venvs" / "llm.BROKEN.20251224_160211"  # Update with exact name
PROJECT_PATH = Path.home() / "Projects" / "LLM" / "HYDRA"
OUTPUT_DIR = Path.home() / "venv_recovery_report"

# Known special packages that need specific installation methods
SPECIAL_PACKAGES = {
    "flash-attn": {
        "type": "compiled",
        "notes": "Requires CUDA toolkit, often needs: pip install flash-attn --no-build-isolation",
        "nightly_source": "https://github.com/Dao-AILab/flash-attention",
    },
    "triton": {
        "type": "compiled",
        "notes": "OpenAI Triton - may need nightly: pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly",
        "nightly_source": "https://github.com/openai/triton",
    },
    "torch": {
        "type": "compiled",
        "notes": "Check CUDA version! pip install torch --index-url https://download.pytorch.org/whl/cu121",
    },
    "xformers": {
        "type": "compiled",
        "notes": "Memory-efficient attention: pip install xformers",
    },
    "bitsandbytes": {
        "type": "compiled",
        "notes": "Quantization library, Linux preferred",
    },
    "apex": {
        "type": "compiled",
        "notes": "NVIDIA Apex - often needs building from source",
    },
    "deepspeed": {
        "type": "compiled",
        "notes": "May need: DS_BUILD_OPS=1 pip install deepspeed",
    },
    "ninja": {
        "type": "build_tool",
        "notes": "Required for building many CUDA extensions",
    },
}


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def extract_from_broken_venv(venv_path: Path) -> dict:
    """Extract all package information from a broken venv."""
    results = {
        "site_packages": [],
        "pip_freeze": None,
        "pip_list_json": None,
        "easy_install": [],
        "dist_info": [],
        "egg_info": [],
        "direct_url_packages": [],
        "errors": [],
    }
    
    if not venv_path.exists():
        results["errors"].append(f"Venv path does not exist: {venv_path}")
        return results
    
    # Find site-packages
    site_packages_paths = list(venv_path.glob("lib/python*/site-packages"))
    if not site_packages_paths:
        site_packages_paths = list(venv_path.glob("Lib/site-packages"))  # Windows
    
    if not site_packages_paths:
        results["errors"].append("Could not find site-packages directory")
        return results
    
    site_packages = site_packages_paths[0]
    print(f"[*] Found site-packages: {site_packages}")
    
    # Method 1: Parse .dist-info directories (most reliable)
    print("[*] Scanning .dist-info directories...")
    for dist_info in site_packages.glob("*.dist-info"):
        pkg_info = parse_dist_info(dist_info)
        if pkg_info:
            results["dist_info"].append(pkg_info)
    
    # Method 2: Parse .egg-info directories
    print("[*] Scanning .egg-info directories...")
    for egg_info in site_packages.glob("*.egg-info"):
        pkg_info = parse_egg_info(egg_info)
        if pkg_info:
            results["egg_info"].append(pkg_info)
    
    # Method 3: Try to read pip's cache/records
    pip_cache = venv_path / "pip-selfcheck.json"
    if pip_cache.exists():
        try:
            results["pip_cache"] = json.loads(pip_cache.read_text())
        except:
            pass
    
    # Method 4: Check for direct_url.json (editable installs, git installs)
    print("[*] Checking for direct URL installs (git, editable)...")
    for dist_info in site_packages.glob("*.dist-info"):
        direct_url = dist_info / "direct_url.json"
        if direct_url.exists():
            try:
                url_info = json.loads(direct_url.read_text())
                url_info["package"] = dist_info.name.replace(".dist-info", "")
                results["direct_url_packages"].append(url_info)
            except:
                pass
    
    # Method 5: List all top-level packages
    print("[*] Listing all installed packages...")
    for item in site_packages.iterdir():
        if item.is_dir() and not item.name.startswith("_") and not item.name.endswith((".dist-info", ".egg-info")):
            results["site_packages"].append(item.name)
    
    return results


def parse_dist_info(dist_info_path: Path) -> dict:
    """Parse a .dist-info directory for package metadata."""
    result = {
        "name": None,
        "version": None,
        "requires": [],
        "installer": None,
        "direct_url": None,
    }
    
    # Parse METADATA file
    metadata_file = dist_info_path / "METADATA"
    if metadata_file.exists():
        try:
            content = metadata_file.read_text(errors="ignore")
            for line in content.split("\n"):
                if line.startswith("Name:"):
                    result["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"):
                    result["version"] = line.split(":", 1)[1].strip()
                elif line.startswith("Requires-Dist:"):
                    result["requires"].append(line.split(":", 1)[1].strip())
        except Exception as e:
            result["error"] = str(e)
    
    # Check INSTALLER
    installer_file = dist_info_path / "INSTALLER"
    if installer_file.exists():
        try:
            result["installer"] = installer_file.read_text().strip()
        except:
            pass
    
    # Check direct_url.json for special installs
    direct_url_file = dist_info_path / "direct_url.json"
    if direct_url_file.exists():
        try:
            result["direct_url"] = json.loads(direct_url_file.read_text())
        except:
            pass
    
    return result if result["name"] else None


def parse_egg_info(egg_info_path: Path) -> dict:
    """Parse a .egg-info directory for package metadata."""
    result = {
        "name": None,
        "version": None,
        "requires": [],
    }
    
    # Parse PKG-INFO
    pkg_info = egg_info_path / "PKG-INFO"
    if pkg_info.exists():
        try:
            content = pkg_info.read_text(errors="ignore")
            for line in content.split("\n"):
                if line.startswith("Name:"):
                    result["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"):
                    result["version"] = line.split(":", 1)[1].strip()
        except:
            pass
    
    # Parse requires.txt
    requires_txt = egg_info_path / "requires.txt"
    if requires_txt.exists():
        try:
            result["requires"] = [
                line.strip() for line in requires_txt.read_text().split("\n")
                if line.strip() and not line.startswith("[")
            ]
        except:
            pass
    
    return result if result["name"] else None


def scan_project_imports(project_path: Path) -> dict:
    """Scan project files to find all imports."""
    imports = defaultdict(set)
    
    print(f"[*] Scanning project imports in: {project_path}")
    
    # Standard library modules to ignore
    stdlib = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else {
        'os', 'sys', 'json', 're', 'math', 'time', 'datetime', 'collections',
        'itertools', 'functools', 'pathlib', 'typing', 'dataclasses', 'enum',
        'abc', 'copy', 'warnings', 'logging', 'argparse', 'pickle', 'subprocess',
        'threading', 'multiprocessing', 'contextlib', 'inspect', 'traceback',
        'unittest', 'tempfile', 'shutil', 'glob', 'io', 'struct', 'random',
        'hashlib', 'base64', 'urllib', 'http', 'socket', 'asyncio', 'concurrent',
        'queue', 'heapq', 'bisect', 'array', 'weakref', 'gc', 'dis', 'ast',
        'token', 'tokenize', 'pdb', 'profile', 'timeit', 'trace', 'csv',
        'configparser', 'textwrap', 'string', 'difflib', 'operator', 'numbers',
        'decimal', 'fractions', 'statistics', 'cmath', 'secrets', 'hmac',
    }
    
    import_pattern = re.compile(
        r'^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))',
        re.MULTILINE
    )
    
    for py_file in project_path.rglob("*.py"):
        if "__pycache__" in str(py_file) or ".egg-info" in str(py_file):
            continue
        try:
            content = py_file.read_text(errors="ignore")
            for match in import_pattern.finditer(content):
                module = match.group(1) or match.group(2)
                if module:
                    top_level = module.split(".")[0]
                    if top_level not in stdlib:
                        imports[top_level].add(str(py_file.relative_to(project_path)))
        except Exception as e:
            print(f"    Warning: Could not parse {py_file}: {e}")
    
    return dict(imports)


def check_requirements_files(project_path: Path) -> list:
    """Find and parse all requirements files in the project."""
    req_files = []
    
    patterns = [
        "requirements*.txt",
        "requirements/*.txt",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
    ]
    
    for pattern in patterns:
        for f in project_path.glob(pattern):
            req_files.append({
                "path": str(f.relative_to(project_path)),
                "content": f.read_text(errors="ignore")[:5000],  # First 5KB
            })
    
    return req_files


def identify_special_packages(packages: list) -> list:
    """Identify packages that need special installation procedures."""
    special = []
    
    for pkg in packages:
        name = pkg.get("name", "").lower() if isinstance(pkg, dict) else pkg.lower()
        
        for special_name, info in SPECIAL_PACKAGES.items():
            if special_name in name:
                special.append({
                    "name": name,
                    "version": pkg.get("version") if isinstance(pkg, dict) else None,
                    "special_info": info,
                    "direct_url": pkg.get("direct_url") if isinstance(pkg, dict) else None,
                })
                break
    
    return special


def generate_recovery_script(venv_info: dict, project_imports: dict, special_pkgs: list) -> str:
    """Generate a bash script to recreate the venv."""
    
    # Collect all packages with versions
    packages = {}
    for pkg in venv_info.get("dist_info", []) + venv_info.get("egg_info", []):
        if pkg and pkg.get("name"):
            packages[pkg["name"].lower()] = pkg.get("version")
    
    script = '''#!/bin/bash
# VENV Recovery Script
# Generated: {timestamp}
# 
# IMPORTANT: Review this script before running!
# Some packages may need manual intervention.

set -e  # Exit on error

VENV_NAME="llm"
VENV_PATH="$HOME/venvs/$VENV_NAME"
PYTHON_VERSION="python3.11"  # Adjust as needed

echo "=== VENV Recovery Script ==="
echo ""

# Step 1: Create new venv
echo "[1/5] Creating new virtual environment..."
if [ -d "$VENV_PATH" ]; then
    echo "ERROR: $VENV_PATH already exists!"
    echo "Please remove or rename it first."
    exit 1
fi

$PYTHON_VERSION -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Step 2: Upgrade pip and install build tools
echo "[2/5] Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel
pip install ninja packaging  # Required for many CUDA builds

# Step 3: Install PyTorch (CRITICAL - do this first!)
echo "[3/5] Installing PyTorch..."
# IMPORTANT: Check your CUDA version with: nvidia-smi
# Then use the appropriate index URL:
# - CUDA 11.8: https://download.pytorch.org/whl/cu118
# - CUDA 12.1: https://download.pytorch.org/whl/cu121
# - CUDA 12.4: https://download.pytorch.org/whl/cu124

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is working
python -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')"

# Step 4: Install special packages (order matters!)
echo "[4/5] Installing special packages..."

'''.format(timestamp=datetime.now().isoformat())
    
    # Add special package installations
    for pkg in special_pkgs:
        name = pkg["name"]
        info = pkg["special_info"]
        script += f'''
# --- {name} ---
# Notes: {info["notes"]}
'''
        if "flash-attn" in name:
            script += '''# Flash Attention - may take a while to compile
# Option 1: Latest release
pip install flash-attn --no-build-isolation

# Option 2: From GitHub (if you need a specific version)
# pip install git+https://github.com/Dao-AILab/flash-attention.git

'''
        elif "triton" in name:
            version = pkg.get("version", "")
            if "nightly" in version.lower() or not version:
                script += '''# Triton Nightly
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

'''
            else:
                script += f'''# Triton (version: {version})
pip install triton=={version}

'''
        elif "xformers" in name:
            script += '''# xformers - memory efficient attention
pip install xformers

'''
        elif "deepspeed" in name:
            script += '''# DeepSpeed
DS_BUILD_OPS=1 pip install deepspeed

'''
        elif "bitsandbytes" in name:
            script += '''# bitsandbytes - quantization
pip install bitsandbytes

'''
    
    # Add regular packages
    script += '''
# Step 5: Install remaining packages
echo "[5/5] Installing remaining packages..."

# Core ML packages
pip install transformers datasets tokenizers accelerate
pip install einops safetensors
pip install wandb tensorboard

'''
    
    # Add discovered packages (excluding special ones and torch)
    regular_packages = []
    special_names = {p["name"].lower() for p in special_pkgs}
    special_names.update(["torch", "torchvision", "torchaudio", "nvidia-", "cuda-"])
    
    for name, version in sorted(packages.items()):
        if not any(s in name.lower() for s in special_names):
            if version:
                regular_packages.append(f"{name}=={version}")
            else:
                regular_packages.append(name)
    
    if regular_packages:
        script += "# Discovered packages from broken venv:\n"
        script += "pip install \\\n    " + " \\\n    ".join(regular_packages[:50])  # Limit to 50
        if len(regular_packages) > 50:
            script += f"\n\n# ... and {len(regular_packages) - 50} more packages (see full list in report)"
    
    script += '''

# Step 6: Install project in editable mode (if setup.py exists)
echo "[6/6] Installing project..."
cd ~/Projects/LLM/HYDRA
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e .
fi

echo ""
echo "=== Recovery Complete ==="
echo "Activate with: source $VENV_PATH/bin/activate"
'''
    
    return script


def generate_report(venv_info: dict, project_imports: dict, special_pkgs: list, 
                   req_files: list, output_dir: Path) -> Path:
    """Generate a comprehensive recovery report."""
    
    report_path = output_dir / "recovery_report.md"
    
    with open(report_path, "w") as f:
        f.write("# VENV Recovery Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- Packages found in broken venv: {len(venv_info.get('dist_info', []))}\n")
        f.write(f"- Unique imports in project: {len(project_imports)}\n")
        f.write(f"- Special packages needing attention: {len(special_pkgs)}\n\n")
        
        # Special packages section
        f.write("## ⚠️ Special Packages (Need Manual Attention)\n\n")
        f.write("These packages require special installation procedures:\n\n")
        
        for pkg in special_pkgs:
            f.write(f"### {pkg['name']}\n")
            f.write(f"- **Version found**: {pkg.get('version', 'Unknown')}\n")
            f.write(f"- **Type**: {pkg['special_info']['type']}\n")
            f.write(f"- **Notes**: {pkg['special_info']['notes']}\n")
            if pkg.get('direct_url'):
                f.write(f"- **Installed from**: `{json.dumps(pkg['direct_url'])}`\n")
            f.write("\n")
        
        # Git/Editable installs
        if venv_info.get("direct_url_packages"):
            f.write("## 📦 Git/Editable Installs\n\n")
            f.write("These packages were installed from Git or in editable mode:\n\n")
            for pkg in venv_info["direct_url_packages"]:
                f.write(f"- **{pkg.get('package', 'Unknown')}**\n")
                if pkg.get("url"):
                    f.write(f"  - URL: `{pkg['url']}`\n")
                if pkg.get("vcs_info"):
                    f.write(f"  - VCS: `{json.dumps(pkg['vcs_info'])}`\n")
            f.write("\n")
        
        # All packages
        f.write("## 📋 All Packages Found\n\n")
        f.write("| Package | Version | Installer |\n")
        f.write("|---------|---------|----------|\n")
        for pkg in sorted(venv_info.get("dist_info", []), key=lambda x: x.get("name", "").lower()):
            if pkg:
                f.write(f"| {pkg.get('name', 'Unknown')} | {pkg.get('version', '-')} | {pkg.get('installer', '-')} |\n")
        f.write("\n")
        
        # Project imports
        f.write("## 🔍 Project Import Analysis\n\n")
        f.write("Imports found in project code:\n\n")
        for module, files in sorted(project_imports.items()):
            f.write(f"- **{module}** (used in {len(files)} files)\n")
        f.write("\n")
        
        # Missing packages
        found_packages = {p.get("name", "").lower() for p in venv_info.get("dist_info", []) if p}
        missing = set(project_imports.keys()) - found_packages
        
        # Package name mapping (import name -> pip name)
        import_to_pip = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "sklearn": "scikit-learn",
            "yaml": "pyyaml",
            "skimage": "scikit-image",
        }
        
        adjusted_missing = set()
        for m in missing:
            pip_name = import_to_pip.get(m, m)
            if pip_name.lower() not in found_packages:
                adjusted_missing.add(f"{m} (pip: {pip_name})" if m != pip_name else m)
        
        if adjusted_missing:
            f.write("## ⚠️ Potentially Missing Packages\n\n")
            f.write("These imports are used in the project but weren't found in the venv:\n\n")
            for m in sorted(adjusted_missing):
                f.write(f"- {m}\n")
            f.write("\n")
        
        # Requirements files
        if req_files:
            f.write("## 📄 Requirements Files Found\n\n")
            for rf in req_files:
                f.write(f"### {rf['path']}\n\n")
                f.write("```\n")
                f.write(rf['content'][:2000])
                if len(rf['content']) > 2000:
                    f.write("\n... (truncated)")
                f.write("\n```\n\n")
    
    return report_path


def main():
    print("=" * 60)
    print("VENV Recovery Discovery Script")
    print("=" * 60)
    print()
    
    # Verify paths
    print(f"[*] Broken venv path: {BROKEN_VENV_PATH}")
    print(f"[*] Project path: {PROJECT_PATH}")
    
    if not BROKEN_VENV_PATH.exists():
        print(f"\n[!] ERROR: Broken venv not found at {BROKEN_VENV_PATH}")
        print("[!] Please update BROKEN_VENV_PATH at the top of this script.")
        
        # Try to find it
        venvs_dir = Path.home() / "venvs"
        if venvs_dir.exists():
            print(f"\n[*] Found these venvs in {venvs_dir}:")
            for v in venvs_dir.iterdir():
                if v.is_dir():
                    print(f"    - {v.name}")
        return
    
    if not PROJECT_PATH.exists():
        print(f"\n[!] WARNING: Project path not found at {PROJECT_PATH}")
        print("[!] Will skip project import analysis.")
    
    output_dir = ensure_output_dir()
    print(f"[*] Output directory: {output_dir}")
    print()
    
    # Extract info from broken venv
    print("[STEP 1] Analyzing broken venv...")
    venv_info = extract_from_broken_venv(BROKEN_VENV_PATH)
    
    if venv_info["errors"]:
        for err in venv_info["errors"]:
            print(f"[!] Error: {err}")
    
    print(f"    Found {len(venv_info['dist_info'])} packages via .dist-info")
    print(f"    Found {len(venv_info['egg_info'])} packages via .egg-info")
    print(f"    Found {len(venv_info['direct_url_packages'])} git/editable installs")
    print()
    
    # Scan project imports
    print("[STEP 2] Scanning project imports...")
    if PROJECT_PATH.exists():
        project_imports = scan_project_imports(PROJECT_PATH)
        print(f"    Found {len(project_imports)} unique import modules")
    else:
        project_imports = {}
    print()
    
    # Check requirements files
    print("[STEP 3] Checking requirements files...")
    if PROJECT_PATH.exists():
        req_files = check_requirements_files(PROJECT_PATH)
        print(f"    Found {len(req_files)} requirements/config files")
    else:
        req_files = []
    print()
    
    # Identify special packages
    print("[STEP 4] Identifying special packages...")
    all_packages = venv_info.get("dist_info", []) + venv_info.get("egg_info", [])
    special_pkgs = identify_special_packages(all_packages)
    print(f"    Found {len(special_pkgs)} packages needing special installation")
    for pkg in special_pkgs:
        print(f"      - {pkg['name']}: {pkg['special_info']['notes'][:60]}...")
    print()
    
    # Generate recovery script
    print("[STEP 5] Generating recovery script...")
    recovery_script = generate_recovery_script(venv_info, project_imports, special_pkgs)
    script_path = output_dir / "recover_venv.sh"
    script_path.write_text(recovery_script)
    script_path.chmod(0o755)
    print(f"    Saved: {script_path}")
    print()
    
    # Generate report
    print("[STEP 6] Generating recovery report...")
    report_path = generate_report(venv_info, project_imports, special_pkgs, req_files, output_dir)
    print(f"    Saved: {report_path}")
    print()
    
    # Save raw data as JSON
    json_path = output_dir / "venv_data.json"
    with open(json_path, "w") as f:
        # Convert sets to lists for JSON serialization
        project_imports_serializable = {k: list(v) for k, v in project_imports.items()}
        json.dump({
            "venv_info": venv_info,
            "project_imports": project_imports_serializable,
            "special_packages": special_pkgs,
            "requirements_files": req_files,
        }, f, indent=2, default=str)
    print(f"    Saved raw data: {json_path}")
    print()
    
    # Summary
    print("=" * 60)
    print("RECOVERY SUMMARY")
    print("=" * 60)
    print(f"Total packages found: {len(venv_info['dist_info'])}")
    print(f"Special packages: {len(special_pkgs)}")
    print()
    print("Next steps:")
    print(f"  1. Review the report: {report_path}")
    print(f"  2. Edit the recovery script as needed: {script_path}")
    print(f"  3. Check your CUDA version: nvidia-smi")
    print(f"  4. Run the recovery script: bash {script_path}")
    print()
    
    # Quick list of special packages
    if special_pkgs:
        print("⚠️  PACKAGES REQUIRING SPECIAL ATTENTION:")
        for pkg in special_pkgs:
            print(f"   • {pkg['name']}")


if __name__ == "__main__":
    main()
