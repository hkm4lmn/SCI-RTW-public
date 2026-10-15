"""
SCI-RTW Predictor: evaluation figures for the report (Chapter 4)

Reads models/layer{1,2}_results.json and models/evaluation_report.json
(both written by train_models.py) and saves PNG figures to docs/figures/.
Nothing is recomputed here, so the figures always match the reported numbers.

Run:  python src/train_models.py   (first, if the models are not trained yet)
      python src/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["xgboost", "random_forest", "logistic_regression"]
MODEL_LABELS = {"xgboost": "XGBoost", "random_forest": "Random Forest",
                "logistic_regression": "Logistic Regression"}
# One fixed colour per model in every figure. The three were checked together
# for colour-blind separation; line style and marker repeat the distinction
# so the figures also survive black-and-white printing.
MODEL_COLORS = {"xgboost": "#2a78d6", "random_forest": "#eb6834", "logistic_regression": "#1baf7a"}
MODEL_LINESTYLES = {"xgboost": "-", "random_forest": "--", "logistic_regression": "-."}
MODEL_MARKERS = {"xgboost": "o", "random_forest": "s", "logistic_regression": "^"}

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE = "#2a78d6"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

FIELD_LABELS = {
    "sex": "Biological sex", "age_at_injury": "Age at injury", "education_level": "Education level",
    "marital_status": "Marital status", "veteran_status": "Veteran status",
    "pre_injury_employment_status": "Pre-injury employment status", "injury_cause": "Cause of injury",
    "vertebral_injury_level": "Vertebral fracture/dislocation", "associated_injury": "Associated injury",
    "work_related_injury": "Work-related injury", "neuro_level_discharge": "Neurologic level (discharge)",
    "aisgrade_discharge": "AIS grade (discharge)", "injury_completeness_discharge": "Injury completeness",
    "asia_motor_total_discharge": "ASIA motor total score",
}
TIMING_ORDER = ["Year 1", "Year 5", "By Year 10"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": AXIS,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path.relative_to(PROJECT_ROOT))


def roc_figure(l1_results, l1_eval):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=":", label="No skill (AUC 0.500)")
    for name in MODEL_ORDER:
        curve = l1_eval["roc_curves"][name]
        auc = l1_results[name]["metrics"]["roc_auc"]
        ax.plot(curve["fpr"], curve["tpr"], color=MODEL_COLORS[name], linewidth=2,
                linestyle=MODEL_LINESTYLES[name], label=f"{MODEL_LABELS[name]} (AUC {auc:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.005), aspect="equal",
           xlabel="False positive rate (Non-RTW patients predicted RTW)",
           ylabel="True positive rate (RTW patients predicted RTW)",
           title="Layer 1: ROC curves on the held-out test set")
    ax.legend(loc="lower right")
    save(fig, "layer1_roc_curves.png")


def confusion_figure(results, class_labels, display_order, title, filename):
    """Each row is the true class; cells are shaded by the share of that row."""
    idx = [class_labels.index(c) for c in display_order]
    fig, axes = plt.subplots(1, 3, figsize=(3.5 * 3 + 0.6, 3.9), sharey=True)
    for ax, name in zip(axes, MODEL_ORDER):
        cm = np.array(results[name]["metrics"]["confusion_matrix"])[np.ix_(idx, idx)]
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
        for i in range(len(idx)):
            for j in range(len(idx)):
                ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9.5,
                        color="#ffffff" if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(idx)), display_order)
        ax.set_yticks(range(len(idx)), display_order)
        ax.set_xlabel("Predicted")
        ax.set_title(f"{MODEL_LABELS[name]}  ·  accuracy {results[name]['metrics']['accuracy']:.1%}",
                     fontsize=10, fontweight="normal", color=INK)
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_frame_on(False)
        # 2px surface gap between cells
        ax.set_xticks(np.arange(-0.5, len(idx)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(idx)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    axes[0].set_ylabel("Actual")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.text(0.01, -0.04, "Cells show the number of test patients and the share of each actual class (row).",
             fontsize=8.5, color=INK_SECONDARY)
    save(fig, filename)


def calibration_figure(l1_eval):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=":", label="Perfect calibration")
    for name in MODEL_ORDER:
        cal = l1_eval["calibration"][name]
        ax.plot([c["mean_predicted"] for c in cal["curve"]], [c["observed_rate"] for c in cal["curve"]],
                color=MODEL_COLORS[name], linewidth=2, linestyle=MODEL_LINESTYLES[name],
                marker=MODEL_MARKERS[name], markersize=5, markeredgecolor=SURFACE, markeredgewidth=1,
                label=f"{MODEL_LABELS[name]} (Brier {cal['brier_score']:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           xlabel="Predicted probability of RTW (mean of each tenth of patients)",
           ylabel="Observed share who returned to work",
           title="Layer 1: calibration on the held-out test set")
    ax.legend(loc="upper left")
    save(fig, "layer1_calibration.png")


def factor_importance_figure(layer_eval, title, filename):
    fields = layer_eval["factor_importance"]["fields"][::-1]
    values = [max(f["auc_drop_mean"], 0) for f in fields]
    errors = [f["auc_drop_std"] for f in fields]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(fields) + 1.4))
    ax.barh([FIELD_LABELS[f["field"]] for f in fields], values, xerr=errors, color=BLUE, height=0.62,
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 2})
    for y, (f, v, e) in enumerate(zip(fields, values, errors)):
        ax.text(v + e + max(values) * 0.015, y, f"{v:.3f}", va="center", fontsize=8.5,
                color=INK_SECONDARY)
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel(f"Drop in ROC-AUC when the field is shuffled (mean of {layer_eval['factor_importance']['repeats']}"
                  " shuffles; bars show ±1 SD)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, filename)


def accuracy_vs_baseline_figure(results_by_layer, evaluation):
    """Test accuracy with its 95% interval, against the majority-class baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), sharey=True)
    titles = {"layer1": "Layer 1: RTW status", "layer2": "Layer 2: RTW timing"}
    for ax, key in zip(axes, ("layer1", "layer2")):
        ev = evaluation[key]
        baseline = ev["majority_baseline"]["accuracy"]
        ax.axvline(baseline, color=INK_MUTED, linewidth=1.2, linestyle="--")
        ax.text(baseline, len(MODEL_ORDER) - 0.42, f" Always predict\n \"{ev['majority_baseline']['always_predicts']}\""
                f": {baseline:.1%}", fontsize=8.5, color=INK_SECONDARY, va="top")
        for y, name in enumerate(MODEL_ORDER[::-1]):
            acc = results_by_layer[key][name]["metrics"]["accuracy"]
            lo, hi = ev["test_ci"][name]["accuracy"]
            ax.plot([lo, hi], [y, y], color=MODEL_COLORS[name], linewidth=2, solid_capstyle="round")
            ax.plot(acc, y, marker=MODEL_MARKERS[name], markersize=8, color=MODEL_COLORS[name],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(hi + 0.004, y, f"{acc:.1%}", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[n] for n in MODEL_ORDER[::-1]])
        ax.set_ylim(-0.6, len(MODEL_ORDER) - 0.2)
        ax.set_xlim(0.48, 0.74)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("Test accuracy (line = 95% bootstrap interval)")
        ax.set_title(titles[key])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "accuracy_vs_baseline.png")


if __name__ == "__main__":
    evaluation = json.load(open(MODEL_DIR / "evaluation_report.json"))
    l1 = json.load(open(MODEL_DIR / "layer1_results.json"))
    l2 = json.load(open(MODEL_DIR / "layer2_results.json"))

    accuracy_vs_baseline_figure({"layer1": l1, "layer2": l2}, evaluation)
    roc_figure(l1, evaluation["layer1"])
    calibration_figure(evaluation["layer1"])
    confusion_figure(l1, evaluation["layer1"]["class_labels"], ["Non-RTW", "RTW"],
                     "Layer 1: confusion matrices on the held-out test set", "layer1_confusion_matrices.png")
    confusion_figure(l2, evaluation["layer2"]["class_labels"], TIMING_ORDER,
                     "Layer 2: confusion matrices on the held-out test set", "layer2_confusion_matrices.png")
    factor_importance_figure(evaluation["layer1"], "Layer 1: which factors the RTW prediction relies on",
                             "layer1_factor_importance.png")
    factor_importance_figure(evaluation["layer2"], "Layer 2: which factors the timing prediction relies on",
                             "layer2_factor_importance.png")
