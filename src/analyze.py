from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif", "text.latex.preamble": r"\usepackage{OldStandard}"})
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"

cases = []
raw = []
for index in range(4):
    orientations = {}
    for orientation in ["original", "semantic_swapped"]:
        result = json.loads((RESULTS / "test" / f"{index:02d}_{orientation}.json").read_text())
        active = {round(float(point["target"]), 3): point for point in result["active"]}
        sham = {round(float(point["matched_target"]), 3): point for point in result["sham"]}
        orientations[orientation] = {"active": active, "sham": sham}
        raw.append((index, orientation, active, sham))
    row = {"case": index + 1}
    for label, field in [("numeric", "confidence_number"), ("verbal", "confidence_word")]:
        strong = float(np.mean([orientations[o]["active"][-3.0][field] for o in orientations]))
        near = float(np.mean([orientations[o]["active"][-0.1][field] for o in orientations]))
        sham_strong = float(np.mean([orientations[o]["sham"][-3.0][field] for o in orientations]))
        sham_near = float(np.mean([orientations[o]["sham"][-0.1][field] for o in orientations]))
        row[f"{label}_strong"] = strong
        row[f"{label}_near"] = near
        row[f"{label}_near_minus_strong"] = near - strong
        row[f"{label}_strong_minus_near"] = strong - near
        row[f"{label}_sham_strong"] = sham_strong
        row[f"{label}_sham_near"] = sham_near
        row[f"{label}_near_active_minus_sham"] = near - sham_near
    row["margin_strong"] = float(np.mean([orientations[o]["active"][-3.0]["original_margin"] for o in orientations]))
    row["margin_near"] = float(np.mean([orientations[o]["active"][-0.1]["original_margin"] for o in orientations]))
    cases.append(row)

summary = json.loads((RESULTS / "summary.json").read_text())
summary["preflip_a_side"] = {
    "case_level_orientation_averaged": cases,
    "numeric_mean_strong": float(np.mean([row["numeric_strong"] for row in cases])),
    "numeric_mean_near": float(np.mean([row["numeric_near"] for row in cases])),
    "numeric_mean_strong_minus_near": float(np.mean([row["numeric_strong_minus_near"] for row in cases])),
    "numeric_mean_near_minus_strong": float(np.mean([row["numeric_near_minus_strong"] for row in cases])),
    "numeric_opposite_direction_cases": int(sum(row["numeric_near_minus_strong"] > 0 for row in cases)),
    "verbal_mean_strong": float(np.mean([row["verbal_strong"] for row in cases])),
    "verbal_mean_near": float(np.mean([row["verbal_near"] for row in cases])),
    "verbal_mean_strong_minus_near": float(np.mean([row["verbal_strong_minus_near"] for row in cases])),
    "verbal_mean_near_minus_strong": float(np.mean([row["verbal_near_minus_strong"] for row in cases])),
    "verbal_opposite_direction_cases": int(sum(row["verbal_near_minus_strong"] > 0 for row in cases)),
    "mean_verified_margin_strong": float(np.mean([row["margin_strong"] for row in cases])),
    "mean_verified_margin_near": float(np.mean([row["margin_near"] for row in cases]))
}
summary["preflip_sham"] = {
    "numeric_active_strong": float(np.mean([row["numeric_strong"] for row in cases])),
    "numeric_sham_strong": float(np.mean([row["numeric_sham_strong"] for row in cases])),
    "numeric_active_near": float(np.mean([row["numeric_near"] for row in cases])),
    "numeric_sham_near": float(np.mean([row["numeric_sham_near"] for row in cases])),
    "numeric_near_active_minus_sham": float(np.mean([row["numeric_near_active_minus_sham"] for row in cases])),
    "numeric_near_active_minus_sham_cases": [float(row["numeric_near_active_minus_sham"]) for row in cases],
    "verbal_active_strong": float(np.mean([row["verbal_strong"] for row in cases])),
    "verbal_sham_strong": float(np.mean([row["verbal_sham_strong"] for row in cases])),
    "verbal_active_near": float(np.mean([row["verbal_near"] for row in cases])),
    "verbal_sham_near": float(np.mean([row["verbal_sham_near"] for row in cases])),
    "verbal_near_active_minus_sham": float(np.mean([row["verbal_near_active_minus_sham"] for row in cases])),
    "verbal_near_active_minus_sham_cases": [float(row["verbal_near_active_minus_sham"]) for row in cases]
}
summary["selection"] = {
    "test_source": "first four rows of the pre-existing test candidate file",
    "test_case_count": 4,
    "test_exclusions": 0,
    "confidence_used_for_test_selection": False,
    "sample_size_reason": "CPU runtime limit for the corrected shared-prefix experiment",
    "validation_confidence_queried": False,
    "controller_frozen_after_validation": True
}
summary["controller_details"] = {
    "max_iterations": 12,
    "target_tolerance": 0.15,
    "position_tolerance": 0.15,
    "margin_step_clip": 1.5,
    "position_step_clip": 0.75,
    "update_norm_cap_fraction": 0.08,
    "damping": 1e-8,
    "gradient_recomputed_each_iteration": True,
    "test_states_over_tolerance": 1,
    "over_tolerance_state": {"case": 4, "orientation": "semantic_swapped", "target": -0.1, "verified_margin": 0.0, "position_drift": 0.15625, "iterations": 12},
    "test_states_excluded": 0
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(RESULTS / "preflip_case_table.json").write_text(json.dumps(cases, indent=2) + "\n")

fig = plt.figure(figsize=(7.0, 3.35))
for column, (measure, title, limits) in enumerate([
    ("numeric", "Numeric confidence", [1.45, 2.35]),
    ("verbal", "Verbal confidence", [2.55, 3.65])
]):
    ax = fig.add_subplot(1, 2, column + 1)
    strong = []
    near = []
    for row in cases:
        left = row[f"{measure}_strong"]
        right = row[f"{measure}_near"]
        strong.append(left)
        near.append(right)
        ax.plot([0, 1], [left, right], marker="o", lw=1.0, alpha=0.8)
        ax.text(-0.055, left, str(row["case"]), fontsize=7.2, ha="right", va="center")
    mean_strong = float(np.mean(strong))
    mean_near = float(np.mean(near))
    ax.plot([0, 1], [mean_strong, mean_near], marker="s", lw=2.5, markersize=5.5)
    ax.set_xticks([0, 1], [r"Strong A  $R\approx-3$", r"Near flip A  $R\approx-0.1$"])
    ax.set_ylabel("Reported confidence")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(*limits)
    ax.grid(axis="y", alpha=0.18)
    ax.text(0.5, 0.04, rf"mean change  ${mean_near - mean_strong:+.3f}$", transform=ax.transAxes, ha="center", fontsize=8)

fig.tight_layout(w_pad=2.0)
fig.savefig(FIGURES / "preflip_primary.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "preflip_primary.png", dpi=240, bbox_inches="tight")
