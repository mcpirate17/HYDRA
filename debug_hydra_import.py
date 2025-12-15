#!/usr/bin/env python3
"""Debug script to diagnose hydra import issues."""

import sys
import os

print("=" * 60)
print("HYDRA IMPORT DEBUG")
print("=" * 60)

# 1. Check Python path
print("\n[1] Python executable:")
print(f"    {sys.executable}")

print("\n[2] sys.path (import search order):")
for i, p in enumerate(sys.path):
    print(f"    {i}: {p}")

# 2. Check if hydra is importable and from where
print("\n[3] Looking for 'hydra' module:")
import importlib.util
spec = importlib.util.find_spec("hydra")
if spec:
    print(f"    Found: {spec.origin}")
    print(f"    Submodule search locations: {spec.submodule_search_locations}")
else:
    print("    NOT FOUND in sys.path")

# 3. Check current directory
print("\n[4] Current directory:")
print(f"    {os.getcwd()}")

# 4. Check if local hydra folder exists
local_hydra = os.path.join(os.getcwd(), "hydra")
print(f"\n[5] Local ./hydra folder exists: {os.path.isdir(local_hydra)}")
if os.path.isdir(local_hydra):
    print("    Contents:")
    for item in sorted(os.listdir(local_hydra)):
        full_path = os.path.join(local_hydra, item)
        if os.path.isdir(full_path):
            print(f"      📁 {item}/")
        else:
            print(f"      📄 {item}")

# 5. Check for __pycache__ directories
print("\n[6] __pycache__ directories found:")
pycache_count = 0
for root, dirs, files in os.walk(local_hydra):
    if "__pycache__" in dirs:
        pycache_count += 1
        pycache_path = os.path.join(root, "__pycache__")
        print(f"    {pycache_path}")
if pycache_count == 0:
    print("    None found")

# 6. Check pip installed hydra packages
print("\n[7] Pip packages containing 'hydra':")
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
hydra_packages = [line for line in result.stdout.split('\n') if 'hydra' in line.lower()]
if hydra_packages:
    for pkg in hydra_packages:
        print(f"    ⚠️  {pkg}")
    print("    WARNING: These may conflict with local hydra/")
else:
    print("    None (good)")

# 7. Try importing step by step
print("\n[8] Step-by-step import test:")

def try_import(module_name, from_module=None, item=None):
    try:
        if from_module:
            mod = __import__(from_module, fromlist=[item])
            obj = getattr(mod, item)
            print(f"    ✅ from {from_module} import {item}")
            return True
        else:
            mod = __import__(module_name)
            print(f"    ✅ import {module_name} -> {getattr(mod, '__file__', 'built-in')}")
            return True
    except ImportError as e:
        print(f"    ❌ {module_name}: {e}")
        return False
    except AttributeError as e:
        print(f"    ❌ {module_name}.{item}: {e}")
        return False

# Add current dir to path if not there
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print(f"    (Added {os.getcwd()} to sys.path)")

try_import("hydra")
try_import("hydra.kernels")
try_import("hydra.kernels.fused_ops")
try_import("hydra.kernels.fused_ops", "hydra.kernels.fused_ops", "get_kernel_status")
try_import("hydra.kernels", "hydra.kernels", "get_kernel_status")

# 8. Check actual file contents
print("\n[9] Checking hydra/kernels/__init__.py:")
init_path = os.path.join(local_hydra, "kernels", "__init__.py")
if os.path.exists(init_path):
    with open(init_path, 'r') as f:
        content = f.read()
    if "get_kernel_status" in content:
        print("    ✅ 'get_kernel_status' IS in the file")
    else:
        print("    ❌ 'get_kernel_status' NOT in the file")
    print("\n    File contents:")
    print("    " + "-" * 40)
    for i, line in enumerate(content.split('\n'), 1):
        print(f"    {i:3}: {line}")
else:
    print(f"    ❌ File not found: {init_path}")

print("\n[10] Checking hydra/kernels/fused_ops.py for get_kernel_status:")
fused_path = os.path.join(local_hydra, "kernels", "fused_ops.py")
if os.path.exists(fused_path):
    with open(fused_path, 'r') as f:
        content = f.read()
    if "def get_kernel_status" in content:
        print("    ✅ 'def get_kernel_status' IS defined in the file")
    else:
        print("    ❌ 'def get_kernel_status' NOT defined in the file")
    if "'get_kernel_status'" in content or '"get_kernel_status"' in content:
        print("    ✅ 'get_kernel_status' IS in __all__")
    else:
        print("    ❌ 'get_kernel_status' NOT in __all__")
else:
    print(f"    ❌ File not found: {fused_path}")

print("\n" + "=" * 60)
print("END DEBUG")
print("=" * 60)
