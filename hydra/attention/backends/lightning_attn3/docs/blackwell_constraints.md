# Blackwell SM 12.x Kernel Constraints

## Hardware Limits

| Resource | Blackwell (SM 12.x) | Notes |
|----------|---------------------|-------|
| Max shared memory per block | **101,376 bytes** | Opt-in via `cudaFuncSetAttribute` |
| Default shared memory | 48KB | Without opt-in |
| Max registers per thread | 255 | |
| Max threads per block | 1024 | |
| Warp size | 32 | |

---

## Shared Memory Budget Breakdown

**Target**: Stay under **96KB** (leave 5KB headroom for Triton overhead/alignment)

### Buffer Categories

#### 1. Input Tiles (read from HBM)
| Buffer | Shape | Formula |
|--------|-------|---------|
| `q_tile` | [CBLOCK, d] | `CBLOCK × d × dtype_size` |
| `k_tile` | [CBLOCK, d] | `CBLOCK × d × dtype_size` |
| `v_tile` | [CBLOCK, d] | `CBLOCK × d × dtype_size` |
| `do_tile` | [CBLOCK, d] | `CBLOCK × d × dtype_size` |

#### 2. Output/Accumulator Tiles (written to HBM)
| Buffer | Shape | Formula |
|--------|-------|---------|
| `dq_acc` | [CBLOCK, d] | `CBLOCK × d × 4` (always fp32) |
| `dk_acc` | [CBLOCK, d] | `CBLOCK × d × 4` |
| `dv_acc` | [CBLOCK, d] | `CBLOCK × d × 4` |

#### 3. Intermediate Tiles (recomputed, not stored long-term)
| Buffer | Shape | Formula |
|--------|-------|---------|
| `attn` | [CBLOCK, CBLOCK] | `CBLOCK² × 4` (fp32 for precision) |
| `dattn` | [CBLOCK, CBLOCK] | `CBLOCK² × 4` |

#### 4. Triton Pipeline Buffers
- `num_stages` multiplies input tile memory
- Each stage needs its own copy of input tiles

---

## Constraint Derivation

### Total Shared Memory Formula

```
SRAM_total = num_stages × (4 × CBLOCK × d × dtype_size)    # Input tiles
           + 3 × CBLOCK × d × 4                            # Accumulators (fp32)
           + 2 × CBLOCK² × 4                               # Attention tiles (fp32)
           + overhead                                       # ~4KB alignment/metadata
```

### For dtype = fp16 (2 bytes), d = 64:

```
SRAM = num_stages × (4 × CBLOCK × 64 × 2)   # = num_stages × 512 × CBLOCK
     + 3 × CBLOCK × 64 × 4                  # = 768 × CBLOCK  
     + 2 × CBLOCK² × 4                      # = 8 × CBLOCK²
     + 4096                                 # overhead

SRAM = 512 × num_stages × CBLOCK + 768 × CBLOCK + 8 × CBLOCK² + 4096
```

### Solving for CBLOCK with num_stages = 1:

```
96000 ≥ 512 × CBLOCK + 768 × CBLOCK + 8 × CBLOCK² + 4096
91904 ≥ 1280 × CBLOCK + 8 × CBLOCK²
```

Solving quadratic: **CBLOCK ≤ 56** (round down to power of 2: **CBLOCK = 32**)

### Solving for CBLOCK with num_stages = 2:

```
96000 ≥ 1024 × CBLOCK + 768 × CBLOCK + 8 × CBLOCK² + 4096
91904 ≥ 1792 × CBLOCK + 8 × CBLOCK²
```

Solving: **CBLOCK ≤ 40** (round down: **CBLOCK = 32**)

---

## Blackwell-Safe Configurations

### Conservative (Recommended)
```python
CBLOCK = 16
num_stages = 1
num_warps = 4
# SRAM ≈ 512×16 + 768×16 + 8×256 + 4096 = 8192 + 12288 + 2048 + 4096 = 26.6KB ✓
```

### Moderate
```python
CBLOCK = 32
num_stages = 1  
num_warps = 4
# SRAM ≈ 512×32 + 768×32 + 8×1024 + 4096 = 16384 + 24576 + 8192 + 4096 = 53.2KB ✓
```

### Aggressive (Near Limit)
```python
CBLOCK = 32
num_stages = 2
num_warps = 8
# SRAM ≈ 1024×32 + 768×32 + 8×1024 + 4096 = 32768 + 24576 + 8192 + 4096 = 69.6KB ✓
```

### INVALID (Exceeds Limit)
```python
CBLOCK = 64
num_stages = 2
num_warps = 8
# SRAM ≈ 1024×64 + 768×64 + 8×4096 + 4096 = 65536 + 49152 + 32768 + 4096 = 151.5KB ✗
```

---

## Code Enforcement Checklist

```python
# ============================================================================
# BLACKWELL KERNEL CONSTRAINTS - SM 12.x (RTX 5090, etc.)
# ============================================================================

# Hardware limit
BLACKWELL_SRAM_LIMIT = 101_376  # bytes

# Safety margin for Triton overhead
SRAM_BUDGET = 96_000  # bytes

def validate_blackwell_config(
    CBLOCK: int,
    d: int,
    num_stages: int,
    dtype_size: int = 2,  # fp16
) -> tuple[bool, int]:
    """
    Validate kernel config fits Blackwell shared memory.
    
    Returns:
        (is_valid, estimated_sram_bytes)
    """
    # Input tiles (pipelined)
    input_tiles = num_stages * 4 * CBLOCK * d * dtype_size
    
    # Accumulators (always fp32)
    accumulators = 3 * CBLOCK * d * 4
    
    # Attention intermediates (fp32)
    attention = 2 * CBLOCK * CBLOCK * 4
    
    # Overhead
    overhead = 4096
    
    total = input_tiles + accumulators + attention + overhead
    
    return (total <= SRAM_BUDGET, total)


# ============================================================================
# ENFORCED CONSTRAINTS
# ============================================================================

BLACKWELL_CONSTRAINTS = {
    # Tile sizes
    "CBLOCK_MAX": 32,          # Never exceed
    "CBLOCK_RECOMMENDED": 16,  # Safe default
    
    # Pipeline stages
    "NUM_STAGES_MAX": 2,       # More stages = more SRAM
    "NUM_STAGES_RECOMMENDED": 1,
    
    # Warps
    "NUM_WARPS_MAX": 8,
    "NUM_WARPS_RECOMMENDED": 4,
    
    # Head dimension limits (for d > 128, use head splitting)
    "HEAD_DIM_MAX": 128,
    
    # Constraint: CBLOCK² must fit in registers for attention tile
    # 32×32 = 1024 elements × 4 bytes = 4KB (ok)
    # 64×64 = 4096 elements × 4 bytes = 16KB (too large for shared)
    "CBLOCK_SQUARED_LIMIT": 1024,  # elements
}


def get_blackwell_safe_config(d: int) -> dict:
    """Get safe kernel config for Blackwell GPUs."""
    if d <= 64:
        return {
            "CBLOCK": 32,
            "num_stages": 1,
            "num_warps": 4,
        }
    elif d <= 128:
        return {
            "CBLOCK": 16,
            "num_stages": 1,
            "num_warps": 4,
        }
    else:
        raise ValueError(f"Head dim {d} > 128 requires head splitting")


# ============================================================================
# RUNTIME VALIDATION (call before kernel launch)
# ============================================================================

def assert_blackwell_constraints(
    CBLOCK: int,
    d: int,
    num_stages: int,
    num_warps: int,
):
    """Raise if config violates Blackwell constraints."""
    C = BLACKWELL_CONSTRAINTS
    
    assert CBLOCK <= C["CBLOCK_MAX"], \
        f"CBLOCK={CBLOCK} exceeds max {C['CBLOCK_MAX']}"
    
    assert CBLOCK * CBLOCK <= C["CBLOCK_SQUARED_LIMIT"], \
        f"CBLOCK²={CBLOCK**2} exceeds limit {C['CBLOCK_SQUARED_LIMIT']}"
    
    assert num_stages <= C["NUM_STAGES_MAX"], \
        f"num_stages={num_stages} exceeds max {C['NUM_STAGES_MAX']}"
    
    assert num_warps <= C["NUM_WARPS_MAX"], \
        f"num_warps={num_warps} exceeds max {C['NUM_WARPS_MAX']}"
    
    assert d <= C["HEAD_DIM_MAX"], \
        f"head_dim={d} exceeds max {C['HEAD_DIM_MAX']} (use head splitting)"
    
    is_valid, sram = validate_blackwell_config(CBLOCK, d, num_stages)
    assert is_valid, \
        f"Config requires {sram} bytes SRAM, exceeds budget {SRAM_BUDGET}"
```

---

## Quick Reference Card

| Constraint | Value | Rationale |
|------------|-------|-----------|
| `CBLOCK` | ≤ 32 | Keeps attention tile small |
| `CBLOCK²` | ≤ 1024 | Fits in shared memory |
| `num_stages` | ≤ 2 | Each stage duplicates input tiles |
| `num_warps` | ≤ 8 | Diminishing returns above this |
| `head_dim` | ≤ 128 | Split larger heads |
| **Total SRAM** | ≤ 96KB | Leave 5KB headroom |

### Safe Defaults for Blackwell
```python
CBLOCK = 16
num_stages = 1
num_warps = 4
# → ~27KB SRAM (72% headroom)
```
