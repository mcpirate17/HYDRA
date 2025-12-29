# hydra.model

This package contains HYDRA model wiring (blocks, routing composition, and top-level architectures).

## What belongs here

- Model blocks and composition logic (e.g. MoD/MoR wrapping, block stacks)
- Public model constructors / factories
- Configuration-to-module wiring

## What should *not* live here

- Backend selection logic (belongs in `hydra.attention.registry` / `hydra.attention.factory`)
- Low-level fused primitives (belongs in `hydra.kernels`)
- Backend implementations (belongs in `hydra.attention.*`)

## Current files

- `framework/`: main model wiring for the CCGQA/MoD/MoR-style architecture + routing integrations.
  The attention implementation itself is in `hydra.attention.ccqa.CCGQAAttention`.
- `hybrid_attention_variants.py`: adapter modules for hybrid attention backends
  (including LLA3 integration with lazy imports).
