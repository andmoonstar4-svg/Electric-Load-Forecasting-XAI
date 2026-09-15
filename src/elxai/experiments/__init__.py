"""Experiment orchestration: one function per paper table or figure."""

from elxai.experiments.runner import (
    ModelRun,
    run_loss_ablation,
    run_model_comparison,
    run_xai,
    summarise,
    train_one,
)

__all__ = [
    "ModelRun",
    "run_loss_ablation",
    "run_model_comparison",
    "run_xai",
    "summarise",
    "train_one",
]
