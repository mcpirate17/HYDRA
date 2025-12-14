# HYDRA Source Reference Index

This document indexes all files in the HYDRA project and tracks their origins from the DMTA2 project.

---

## File Index

### Core Package (`hydra/`)

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `__init__.py` | *New* | Package exports for HYDRA |
| `optimization.py` | `numeric_optimization.py` | Optuna-based hyperparameter optimization |

### Model Components (`hydra/model/`)

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `__init__.py` | *New* | Model component exports |
| `ccgqa.py` | `templates/attention/attention_ccgqa.py` | CCGQA + MoD + MoR implementation (1565 lines) |

### Routing Components (`hydra/routing/`)

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `__init__.py` | *New* | Routing component exports |
| `mixture_of_depths.py` | `templates/attention/mixture_of_depths.py` | Standalone MoD router |

### Tests (`tests/`)

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `__init__.py` | *New* | Test package init |
| `conftest.py` | *New* | Pytest fixtures and configuration |
| `test_paper_compliance.py` | `tests/test_paper_compliance.py` | 64 paper compliance tests |

### Diagnostics (`diagnostics/`)

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `__init__.py` | *New* | Diagnostics package init |
| `scaling_analysis.py` | `scaling_analysis.py` | Multi-scale curve fitting and 4B predictions |
| `run_variant_diagnostics.py` | `run_variant_diagnostics.py` | Variant diagnostic runner |
| `deep_diagnosis.py` | `deep_diagnosis.py` | Deep model diagnostics |

### Configuration (`configs/`)

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `variants.yaml` | *New* (from test data) | Model variant definitions 100M→4B |

### Reports (`reports/`)

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `scaling_analysis.png` | `reports/scaling_analysis.png` | 4-panel scaling plots |
| `scaling_summary_table.png` | `reports/scaling_summary_table.png` | Summary table image |
| `scaling_analysis_results.json` | `reports/scaling_analysis_results.json` | Full analysis data |

### Documentation (`docs/`)

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `ARCHITECTURE.md` | *New* | Deep dive into HYDRA architecture |

### Root Files

| File | Origin (DMTA2) | Description |
|------|----------------|-------------|
| `README.md` | *New* | Project overview and quick start |
| `setup.py` | *New* | Package installation script |
| `requirements.txt` | *New* | Python dependencies |
| `pytest.ini` | `pytest.ini` | Test configuration |
| `LICENSE` | *New* | MIT License |

---

## DMTA2 Template References

The following DMTA2 templates are related to HYDRA but remain in DMTA2:

### Attention Templates (`templates/attention/`)

| File | Relevance | Notes |
|------|-----------|-------|
| `attention_ccgqa.py` | **Primary** | Source for `hydra/model/ccgqa.py` |
| `mixture_of_depths.py` | **Primary** | Source for `hydra/routing/mixture_of_depths.py` |
| `attention_causal_gqa.py` | Related | Base GQA implementation |
| `attention_causal_mha.py` | Related | Base MHA implementation |
| `attention_flash_sdp.py` | Related | Flash attention variant |
| `attention_functional.py` | Related | Functional attention utilities |
| `attention_gated.py` | Related | Gated attention mechanisms |
| `attention_linear.py` | Related | Linear attention variant |
| `attention_sliding_window.py` | Related | Sliding window attention |
| `attention_topk.py` | Related | Top-k sparse attention |

### MLP Templates (`templates/mlp/`)

| File | Relevance | Notes |
|------|-----------|-------|
| `mlp_*.py` | Related | FFN implementations used by CCGQA blocks |

### Norm Templates (`templates/norm/`)

| File | Relevance | Notes |
|------|-----------|-------|
| `norm_*.py` | Related | Normalization layers used by CCGQA |

### Model Templates (`templates/model/`)

| File | Relevance | Notes |
|------|-----------|-------|
| `transformer_*.py` | Related | Base transformer structures |

---

## Key Classes and Functions

### From `hydra/model/ccgqa.py`

| Class/Function | Purpose |
|----------------|---------|
| `CCGQAAttention` | Compressed Convolutional GQA attention layer |
| `CCGQABlock` | Single attention + FFN block |
| `CCGQAMoRBlock` | MoR-enabled recursive block |
| `CCGQAMoDBlockWrapper` | MoD-enabled block wrapper |
| `CCGQAMoDMoRModel` | Complete model with MoD + MoR |
| `create_ccgqa_mod_mor_model()` | Factory function for model creation |

### From `hydra/optimization.py`

| Class/Function | Purpose |
|----------------|---------|
| `optimize_mlp_hyperparams()` | Optimize MLP layer parameters |
| `optimize_attention_hyperparams()` | Optimize attention parameters |
| `optimize_norm_hyperparams()` | Optimize normalization parameters |
| `optimize_all_components()` | Full component optimization |
| `scalar_objective()` | Objective function for Optuna |

### From `diagnostics/scaling_analysis.py`

| Class/Function | Purpose |
|----------------|---------|
| `ScalingDataPoint` | Data class for scaling metrics |
| `ScalingFit` | Curve fitting results |
| `analyze_scaling()` | Main scaling analysis function |
| `fit_polynomial()` | Polynomial curve fitting |
| `fit_power_law()` | Power law curve fitting |
| `plot_scaling_analysis()` | Generate scaling plots |

---

## Paper References

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| CCGQA | [2510.04476](https://arxiv.org/abs/2510.04476) | Compressed attention in latent space |
| MoD | [2404.02258](https://arxiv.org/abs/2404.02258) | Token-level dynamic compute routing |
| MoR | [2507.10524](https://arxiv.org/abs/2507.10524) | Layer-level adaptive depth |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-07 | 0.1.0 | Initial project creation from DMTA2 |

---

*Last updated: December 7, 2025*
