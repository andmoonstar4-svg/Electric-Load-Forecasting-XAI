"""Explainability layer: SHAP attribution with explicit, auditable settings."""

from elxai.xai.shap_explainer import (
    XAIReport,
    auto_nsamples,
    compare_reports,
    explain,
    explain_from_config,
    plot_importance,
    plot_lag_profile,
    resolve_nsamples,
    resolve_target_steps,
)

__all__ = [
    "XAIReport",
    "auto_nsamples",
    "compare_reports",
    "explain",
    "explain_from_config",
    "plot_importance",
    "plot_lag_profile",
    "resolve_nsamples",
    "resolve_target_steps",
]
