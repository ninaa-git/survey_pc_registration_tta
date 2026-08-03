import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- config ----------
method = "PEA_TTA"
base_path   = f"../../../PARENet/output/P2ILReg/{method}/"
momentum_path  = os.path.join(base_path, "ft_validation")

VIS_LO, VIS_HI = 0.0, 1.0


momentums = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1.0] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
#momentums = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5,]
metric_to_plot = "RMSE"

# ---------- helpers ----------
def load_json_results(json_path, metric):
    if not os.path.isfile(json_path):
        return None
    values = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if metric in obj:
                values.append(obj[metric])
    arr = np.array(values) if values else None
    return arr


def mean_std(arr):
    return (float(np.mean(arr)), float(np.std(arr))) if (arr is not None and arr.size > 0) else (None, None)


def apply_mask(arr, mask):
    if arr is None:
        return None
    n = min(arr.shape[0], mask.shape[0])
    return arr[:n][mask[:n]]


def get_json_path(momentum):
    fname = f"val_{momentum}.json"
    return os.path.join(momentum_path, fname)


# ---------- gather results: per_momentum[momentum] ----------
per_momentum = {}
for momentum in momentums:
    path   = get_json_path(momentum)
    arr    = load_json_results(path, metric_to_plot)
    per_momentum[momentum] = arr          # <-- must be outside the if block

    n      = 0 if arr is None else int(arr.size)
    status = f"FOUND  {path}" if arr is not None else f"MISSING {path}"
    print(f"[momentum={momentum}] -> n={n}  ({status})")

# ---------- compute statistics ----------
means = []
stds  = []
for momentum in momentums:
    m, sd = mean_std(per_momentum[momentum])
    means.append(m)
    stds.append(sd)


# ---------- font sizes ----------
SMALL_SIZE  = 6 * 1
MEDIUM_SIZE = 6 * 1.5
BIGGER_SIZE = 6 * 2

plt.rc("font",   size=SMALL_SIZE)
plt.rc("axes",   titlesize=SMALL_SIZE)
plt.rc("axes",   labelsize=MEDIUM_SIZE)
plt.rc("xtick",  labelsize=SMALL_SIZE)
plt.rc("ytick",  labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)

# ---------- plotting ----------
fig, ax = plt.subplots(figsize=(10, 6))

x_ticks = list(momentums)

xs_plot = [x for x, y in zip(x_ticks, means) if y is not None]
ys_plot = [y for y in means if y is not None]
sd_plot = [s for s, y in zip(stds, means) if y is not None]

if xs_plot:
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.errorbar(
        xs_plot, ys_plot,
        yerr=sd_plot,
        marker="o",
        linewidth=1.5,
        markersize=SMALL_SIZE * 1.5,
        capsize=4,
        label=metric_to_plot,
        color=color,
        zorder=2,
    )

    # Highlight the minimum value point with a black circle
    min_idx = int(np.argmin(ys_plot))
    ax.plot(
        xs_plot[min_idx], ys_plot[min_idx],
        "o",
        markersize=SMALL_SIZE * 1.2,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=2.0,
        zorder=3,
        label=f"Best ({xs_plot[min_idx]}): {ys_plot[min_idx]:.3f}",
    )
else:
    print("No data to plot.")

ax.set_xlabel("Momentum")
ax.set_ylabel(metric_to_plot)
#ax.set_xscale("log")
ax.set_xticks(x_ticks)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(v)))
ax.set_xticklabels([str(m) for m in x_ticks], rotation=45, ha="right")

all_vals = [y for y in ys_plot if np.isfinite(y)]
if all_vals:
    y_min    = float(np.min(all_vals))
    y_max    = float(np.max(all_vals))
    spread   = y_max - y_min if y_max != y_min else max(1.0, 0.1 * y_max)
    padding  = 0.15 * spread
    ax.set_ylim(max(0.0, y_min - padding), y_max + padding)
else:
    ax.set_ylim(0, 25)

ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax.grid(True, axis="y", linewidth=0.6, alpha=0.4)
ax.legend(loc="upper right")
ax.set_title(
    f"{metric_to_plot} vs momentum "
)
plt.tight_layout()

out_dir  = "./Figs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"metrics_{method}_{metric_to_plot}.png")
plt.savefig(out_path, dpi=300)
plt.show()
print(f"Saved: {out_path}")