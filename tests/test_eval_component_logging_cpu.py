import pytest
import torch


class _DummyLoader:
    def __init__(self, *, batch_size: int, seq_len: int, vocab_size: int):
        self._batch_size = int(batch_size)
        self._seq_len = int(seq_len)
        self._vocab_size = int(vocab_size)

    def set_max_steps(self, _max_steps: int) -> None:
        return None

    def get_batch(self) -> dict:
        x = torch.randint(0, self._vocab_size, (self._batch_size, self._seq_len), dtype=torch.long)
        # Next-token style labels are fine for this smoke.
        y = x.clone()
        return {"input_ids": x, "labels": y, "attention_mask": None}


class _DummyCurriculum:
    def step(self, **_kwargs):
        return []


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 8)
        self.proj = torch.nn.Linear(8, vocab_size)

    def forward(self, x, mask=None, return_losses: bool = False):
        _ = mask
        h = self.emb(x)
        logits = self.proj(h)
        if return_losses:
            return logits, {"aux_loss": torch.tensor(0.0), "ponder_loss": torch.tensor(0.0)}
        return logits


def test_eval_component_logging_path_cpu(monkeypatch, tmp_path):
    """CPU-only: ensure per-component eval path runs for mixed eval presets.

    This stubs data loading + eval to avoid any HF/network access.
    """

    # Force CPU even if a GPU exists.
    monkeypatch.setattr("hydra.training.trainer.torch.cuda.is_available", lambda: False)

    # Minimal mixed eval preset with two components.
    monkeypatch.setattr(
        "hydra.training.trainer.DATASET_CONFIGS",
        {
            "pretrain_default_eval": {
                "mixed": True,
                "sources": [
                    {"name": "fineweb_edu", "weight": 0.8},
                    {"name": "wikitext2", "weight": 0.2},
                ],
            }
        },
        raising=False,
    )

    # Stub loader for train + eval + component eval.
    def _stub_create_universal_loader(*, dataset, batch_size, seq_len, vocab_size, device, tokenizer_name, max_steps=None):
        _ = (dataset, device, tokenizer_name, max_steps)
        return _DummyLoader(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size)

    monkeypatch.setattr("hydra.training.trainer.create_universal_loader", _stub_create_universal_loader)

    # Avoid heavy model construction; install a tiny CPU model + dummy curriculum.
    def _stub_setup_model(self):
        self.model = _TinyModel(vocab_size=self.config.vocab_size).to(self.device)
        self._use_mod_mor = False
        self._curriculum_controller = _DummyCurriculum()

    monkeypatch.setattr("hydra.training.trainer.Trainer._setup_model", _stub_setup_model)

    # Make eval fire and ensure per-component eval uses our stub.
    monkeypatch.setattr("hydra.training.trainer.maybe_run_fixed_eval", lambda **_kwargs: 1.234)

    def _stub_evaluate_fixed_batches(**_kwargs):
        fixed = _kwargs.get("fixed_eval_batches") or []
        # Return a different loss depending on which fixed batches list we got.
        # (The component loops pass a per-component list, so lengths differ from the mixed eval.)
        return float(len(fixed)) + 0.01

    monkeypatch.setattr("hydra.training.trainer.evaluate_fixed_batches", _stub_evaluate_fixed_batches)

    # Avoid checkpoint/report side-effects.
    monkeypatch.setattr("hydra.training.trainer.Trainer._save_checkpoint", lambda *args, **kwargs: None)

    from hydra.training.config import TrainingConfig
    from hydra.training.trainer import Trainer

    cfg = TrainingConfig(
        mode="testing",
        dataset_name="pretrain_default",
        vocab_size=128,
        batch_size=2,
        grad_accum_steps=1,
        use_compile=False,
        use_triton_kernels=False,
        use_chunked_ce=False,
        gradient_checkpointing=False,
        checkpoint_dir=str(tmp_path),
        report_dir=str(tmp_path),
    )
    # TrainingConfig.__post_init__ overrides some fields based on mode; force the smoke to be 1 step.
    cfg.max_steps = 1
    cfg.max_seq_len = 16
    cfg.eval_interval = 1
    cfg.log_interval = 1
    cfg.save_interval = 10**9

    trainer = Trainer(cfg)

    captured: list[str] = []
    _orig_info = trainer.logger.info

    def _capture_info(msg: str) -> None:
        captured.append(str(msg))
        _orig_info(msg)

    trainer.logger.info = _capture_info  # type: ignore[method-assign]

    trainer.train()

    assert any("[EVAL_COMPONENTS]" in m for m in captured)
