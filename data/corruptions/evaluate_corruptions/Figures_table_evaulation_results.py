"""
visualize_results.py
--------------------
Reads evaluation_results.csv and produces:
  1. Figure 1 – mean_cd1   vs. severity (one line per corruption type)
  2. Figure 2 – mean_cd    vs. severity (one line per corruption type)
  3. Figure 3 – mean_emd   vs. severity (one line per corruption type)
  4. A console table (and CSV export) showing, per corruption,
     the mean of every metric averaged across all 5 severity levels,
     plus a reference row for "clean".

Usage
-----
    python visualize_results.py [path/to/evaluation_results.csv]

If no path is given the script looks for evaluation_results.csv in the
current working directory.
"""

import sys
import os
import re
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Reference values from the paper (Table in the prompt) ──────────────────────
PAPER_REFERENCE = {
    "real": {
        "mean_x": 2.2,  "mean_y": -1.3, "mean_z": 1.2,
        "min_x": -100.0,"min_y": -100.0,"min_z": -12.0,
        "max_x": 100.0, "max_y": 100.0, "max_z": 25.1,
        "mean_n_points": 73_123, "min_n_points": 52_762, "max_n_points": 79_690,
    },
    "sim": {
        "mean_x": 2.1,  "mean_y": -0.5, "mean_z": 1.9,
        "min_x": -100.0,"min_y": -87.0, "min_z": -1.5,
        "max_x": 100.0, "max_y": 100.0, "max_z": 23.3,
        "mean_n_points": 78_776, "min_n_points": 52_695, "max_n_points": 81_538,
    },
}

# ── Colour palette (one colour per corruption family) ──────────────────────────
PALETTE = {
    "clean":               "#333333",
    "uniform":             "#1f77b4",
    "gaussian":            "#ff7f0e",
    "background_noise":    "#2ca02c",
    "global_density_dec":  "#d62728",
    "local_density_dec":   "#9467bd",
    "cutout":              "#8c564b",
    "occlusion":           "#e377c2",
}

MARKER = {
    "clean":               "D",
    "uniform":             "o",
    "gaussian":            "s",
    "background_noise":    "^",
    "global_density_dec":  "v",
    "local_density_dec":   "P",
    "cutout":              "X",
    "occlusion":           "*",
}

PRETTY = {
    "clean":               "Clean",
    "uniform":             "Uniform noise",
    "gaussian":            "Gaussian noise",
    "background_noise":    "Background noise",
    "global_density_dec":  "Global density ↓",
    "local_density_dec":   "Local density ↓",
    "cutout":              "Cutout",
    "occlusion":           "Occlusion",
}

# Metrics used in the per-corruption summary table
SUMMARY_METRICS = [
    "mean_cd1", "mean_cd2", "mean_cd", "mean_emd",
    "mean_mean_x", "mean_mean_y", "mean_mean_z",
    "mean_min_x",  "mean_min_y",  "mean_min_z",
    "mean_max_x",  "mean_max_y",  "mean_max_z",
    "mean_n_points", "min_n_points", "max_n_points",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_label(label: str):
    """Return (corruption_name, severity_int) or (label, 0) for 'clean'."""
    m = re.match(r"^(.+)_s(\d+)$", label)
    if m:
        return m.group(1), int(m.group(2))
    return label, 0          # "clean" → severity 0


def load_and_enrich(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[["corruption", "severity"]] = pd.DataFrame(
        df["label"].map(parse_label).tolist(), index=df.index
    )
    return df


def corruption_order(df: pd.DataFrame):
    """Return corruption names ordered: clean first, rest alphabetically."""
    names = sorted(df["corruption"].unique())
    names = ["clean"] + [n for n in names if n != "clean"]
    return names


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_metric_vs_severity(df: pd.DataFrame, metric: str,
                             ylabel: str, title: str,
                             save_path: str):
    """
    Line plot: x = severity (1–5), y = <metric>.
    'Clean' is drawn as a horizontal dashed reference line at severity=0,
    extended as a faint band across the whole plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    corruptions = corruption_order(df)

    # ── clean reference ───────────────────────────────────────────────────────
    clean_val = df.loc[df["corruption"] == "clean", metric].values
    if len(clean_val):
        ax.axhline(clean_val[0],
                   color=PALETTE["clean"], linewidth=1.4,
                   linestyle="--", label=PRETTY["clean"], zorder=1)

    # ── one line per corruption ───────────────────────────────────────────────
    for corr in corruptions:
        if corr == "clean":
            continue
        sub = df[df["corruption"] == corr].sort_values("severity")
        if sub.empty:
            continue
        color  = PALETTE.get(corr, "#888888")
        marker = MARKER.get(corr,  "o")
        ax.plot(
            sub["severity"], sub[metric],
            color=color, marker=marker,
            linewidth=1.8, markersize=6,
            label=PRETTY.get(corr, corr),
            zorder=2,
        )

    ax.set_xlabel("Severity level", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

# Raw metric column names (without the "mean_" prefix aggregation)
RAW_METRICS = [
    "cd1", "cd2", "cd", "emd",
    "mean_x", "mean_y", "mean_z",
    "min_x",  "min_y",  "min_z",
    "max_x",  "max_y",  "max_z",
    "n_points",
]

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    clean_row = df[df["corruption"] == "clean"][SUMMARY_METRICS]
    if not clean_row.empty:
        rec = {"corruption": "clean (ref)", "severities": "–"}
        rec.update(clean_row.iloc[0].to_dict())
        rows.append(rec)

    for corr in sorted(df["corruption"].unique()):
        if corr == "clean":
            continue
        sub = df[df["corruption"] == corr]
        rec = {
            "corruption": PRETTY.get(corr, corr),
            "severities": f"s1–s{sub['severity'].max()}",
        }

        # apply the semantically correct aggregation per column
        for col in SUMMARY_METRICS:
            if col.startswith("min_"):
                rec[col] = sub[col].min()
            elif col.startswith("max_"):
                rec[col] = sub[col].max()
            else:                          # mean_* columns
                rec[col] = sub[col].mean()

        rows.append(rec)

    return pd.DataFrame(rows).set_index("corruption")


def print_summary_table(summary: pd.DataFrame):
    """Pretty-print the summary table to stdout."""
    # Format floats sensibly
    float_cols = [c for c in summary.columns if summary[c].dtype == float]
    fmt = {}
    for c in float_cols:
        absmax = summary[c].abs().max()
        if absmax == 0:
            fmt[c] = lambda v: "0.000"
        elif absmax < 0.01:
            fmt[c] = lambda v, d=3: f"{v:.2e}"
        elif absmax < 10:
            fmt[c] = lambda v: f"{v:.4f}"
        elif absmax < 1000:
            fmt[c] = lambda v: f"{v:.1f}"
        else:
            fmt[c] = lambda v: f"{v:,.0f}"

    # Build display frame
    display = summary.copy()
    for c, f in fmt.items():
        display[c] = display[c].apply(f)

    # Wrap column names for readability
    col_map = {
        "severities":       "Sev.",
        "mean_cd1":         "CD1↓",
        "mean_cd2":         "CD2↓",
        "mean_cd":          "CD↓",
        "mean_emd":         "EMD↓",
        "mean_mean_x":      "μx",
        "mean_mean_y":      "μy",
        "mean_mean_z":      "μz",
        "mean_min_x":       "min_x",
        "mean_min_y":       "min_y",
        "mean_min_z":       "min_z",
        "mean_max_x":       "max_x",
        "mean_max_y":       "max_y",
        "mean_max_z":       "max_z",
        "mean_n_points":    "μ#pts",
        "min_n_points":     "min#pts",
        "max_n_points":     "max#pts",
    }
    display.rename(columns=col_map, inplace=True)

    print("\n" + "═" * 120)
    print("  SUMMARY TABLE  –  mean metric value per corruption type (averaged across severities s1–s5)")
    print("  Reference values (paper):  Real → n_pts mean=73,123 | Sim → n_pts mean=78,776")
    print("═" * 120)
    print(display.to_string())
    print("═" * 120 + "\n")


def add_paper_reference_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Append the paper's reference rows for real / sim point clouds."""
    for domain, vals in PAPER_REFERENCE.items():
        rec = {"severities": "paper"}
        for k, v in vals.items():
            # map paper key → CSV column name
            col = f"mean_{k}" if not k.startswith("min") and not k.startswith("max") else k
            if col in summary.columns:
                rec[col] = v
        row = pd.DataFrame([rec], index=[f"[paper] {domain}"])
        summary = pd.concat([summary, row])
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results.csv"
    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] File not found: {csv_path}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nLoading  : {csv_path}")
    df = load_and_enrich(csv_path)
    print(f"Rows     : {len(df)}  |  Corruptions: {sorted(df['corruption'].unique())}")

    # ── Figure 1: mean_cd1 ────────────────────────────────────────────────────
    print("\n[1/3] Plotting mean_cd1 vs. severity …")
    plot_metric_vs_severity(
        df, metric="mean_cd1",
        ylabel="Mean CD₁  (one-sided Chamfer Distance ↓)",
        title="CD₁ per corruption type vs. severity level",
        save_path=os.path.join(out_dir, "fig1_mean_cd1_vs_severity.png"),
    )

    # ── Figure 2: mean_cd ─────────────────────────────────────────────────────
    print("[2/3] Plotting mean_cd vs. severity …")
    plot_metric_vs_severity(
        df, metric="mean_cd",
        ylabel="Mean CD  (symmetric Chamfer Distance ↓)",
        title="CD per corruption type vs. severity level",
        save_path=os.path.join(out_dir, "fig2_mean_cd_vs_severity.png"),
    )

    # ── Figure 3: mean_emd ────────────────────────────────────────────────────
    print("[3/3] Plotting mean_emd vs. severity …")
    plot_metric_vs_severity(
        df, metric="mean_emd",
        ylabel="Mean EMD  (Earth Mover's Distance ↓)",
        title="EMD per corruption type vs. severity level",
        save_path=os.path.join(out_dir, "fig3_mean_emd_vs_severity.png"),
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("[4/4] Building summary table …")
    summary = build_summary_table(df)
    #summary = add_paper_reference_rows(summary)
    print_summary_table(summary)

    # Save summary as CSV
    csv_out = os.path.join(out_dir, "summary_per_corruption.csv")
    summary.to_csv(csv_out)
    print(f"  Summary CSV saved → {csv_out}\n")

    print("Done ✓")


if __name__ == "__main__":
    main()