#!/usr/bin/env python3
"""
HYDRA Diagnostics - Single Entry Point

Industry-standard diagnostic tools for HYDRA MoD/MoR model debugging.
Uses pytest, torch.profiler, and memory-profiler for comprehensive analysis.

Usage:
    # Run all tests
    python -m diagnostics

    # Run specific tests
    pytest diagnostics/test_mod_mor_routing.py -v
    
    # Run with memory profiling
    mprof run python -m diagnostics.deep_diagnosis
    mprof plot  # View memory usage

    # Run scaling analysis
    python diagnostics/scaling_analysis.py --predict-4b

    # Profile training performance
    python -m torch.utils.bottleneck trainer.py --max_steps 50
"""

import sys
import subprocess
from pathlib import Path


def run_routing_tests():
    """Run MoD/MoR routing health tests."""
    print("🔍 Running MoD/MoR routing tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "diagnostics/test_mod_mor_routing.py", "-v"
    ], cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_gradient_diagnosis():
    """Run gradient flow analysis."""
    print("🧠 Running gradient flow analysis...")
    result = subprocess.run([
        sys.executable, "diagnostics/deep_diagnosis.py"
    ], cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_scaling_analysis():
    """Run scaling predictions."""
    print("📈 Running scaling analysis...")
    result = subprocess.run([
        sys.executable, "diagnostics/scaling_analysis.py", "--predict-4b"
    ], cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def print_external_tools():
    """Show external debugging tools."""
    print("\n🛠️  External Tools (install separately):")
    print("   pip install memory-profiler matplotlib")
    print("   # Memory profiling:")
    print("   mprof run python trainer.py --max_steps 50")
    print("   mprof plot")
    print("   ")
    print("   # PyTorch bottleneck analysis:")
    print("   python -m torch.utils.bottleneck trainer.py --max_steps 10")
    print("   ")
    print("   # TensorBoard profiling (add to training script):")
    print("   from torch.profiler import profile, ProfilerActivity, schedule")
    print("   # See: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")


def main():
    """Run all diagnostics."""
    print("🚀 HYDRA Diagnostics Suite")
    print("=" * 50)
    
    success = True
    
    # Run tests
    if not run_routing_tests():
        success = False
        
    if not run_gradient_diagnosis():
        success = False
        
    if not run_scaling_analysis():
        success = False
    
    # Show external tools
    print_external_tools()
    
    if success:
        print("\n✅ All diagnostics completed successfully!")
    else:
        print("\n❌ Some diagnostics failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()