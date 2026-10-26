"""
ÉTAPE Évaluation — Mesure honnête des performances sur le TEST.

Produit :
  - reports/metrics.json  (AUC, précision, rappel, F1, ...)
  - figures : matrice de confusion, courbe ROC, importance des variables
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.config import get_logger, load_config, resolve

log = get_logger("evaluate")
plt.rcParams["figure.dpi"] = 120


def evaluate(model, X_te, y_te, best_params, cv_auc,
             baseline_scores=None, champion="xgboost", cfg: dict | None = None):
    cfg = cfg or load_config()
    thr = cfg["business"]["alert_threshold"]
    figdir = resolve(cfg["paths"]["figures"])
    figdir.mkdir(parents=True, exist_ok=True)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= thr).astype(int)

    metrics = {
        "roc_auc_cv": round(float(cv_auc), 4),
        "roc_auc_test": round(float(roc_auc_score(y_te, proba)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "threshold": thr,
        "champion_deployed": champion,
        "best_params": best_params,
        "n_test": int(len(y_te)),
        "churn_rate_test": round(float(y_te.mean()), 4),
    }
    if baseline_scores:
        metrics["baseline_logistic_regression"] = baseline_scores
        metrics["gain_vs_baseline_test"] = round(
            metrics["roc_auc_test"] - baseline_scores["roc_auc_test"], 4)
        _plot_baseline_comparison(metrics, figdir, champion)
    with open(resolve(cfg["paths"]["metrics"]), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("ROC-AUC (test) : %.3f", metrics["roc_auc_test"])
    log.info("Précision : %.3f | Rappel : %.3f | F1 : %.3f",
             metrics["precision"], metrics["recall"], metrics["f1"])
    print("\n" + classification_report(y_te, pred,
          target_names=["Actif (0)", "Churn (1)"]))

    # --- Matrice de confusion ---
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=["Actif", "Churn"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Matrice de confusion (test)")
    fig.tight_layout(); fig.savefig(figdir / "confusion_matrix.png"); plt.close(fig)

    # --- Courbe ROC ---
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_te, proba, ax=ax, name="Modèle")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title("Courbe ROC (test)")
    fig.tight_layout(); fig.savefig(figdir / "roc_curve.png"); plt.close(fig)

    # --- Importance des variables ---
    _plot_importance(model, figdir)

    log.info("Figures écrites dans %s", figdir)
    return metrics


def _plot_baseline_comparison(metrics: dict, figdir, champion="xgboost"):
    """Barres comparant baseline et modèle final (AUC test)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    final_label = {"xgboost": "XGBoost", "random_forest": "Random Forest",
                   "logistic_regression": "Rég. logistique"}.get(champion, champion)
    names = ["Régression logistique\n(baseline)", f"{final_label}\n(modèle final)"]
    vals = [metrics["baseline_logistic_regression"]["roc_auc_test"],
            metrics["roc_auc_test"]]
    bars = ax.bar(names, vals, color=["#a0aec0", "#2b6cb0"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.text(0.02, 0.505, "hasard (0.5)", color="grey", fontsize=8)
    ax.set_ylabel("ROC-AUC (test)")
    ax.set_title("Baseline vs modèle final")
    fig.tight_layout(); fig.savefig(figdir / "baseline_comparison.png"); plt.close(fig)


def _plot_importance(model, figdir):
    """Importance des variables lisible malgré le one-hot encoding."""
    try:
        pre = model.named_steps["pre"]
        names = pre.get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        imp = pd.Series(importances, index=names).sort_values(ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        imp.plot.barh(ax=ax, color="#2b6cb0")
        ax.set_title("Importance des variables")
        fig.tight_layout(); fig.savefig(figdir / "feature_importance.png"); plt.close(fig)
    except Exception as e:   # noqa
        log.warning("Importance non disponible : %s", e)


if __name__ == "__main__":
    from src.train import train
    cfg = load_config()
    model, X_te, y_te, bp, auc, bl, champ = train(cfg)
    evaluate(model, X_te, y_te, bp, auc, bl, champ, cfg)
