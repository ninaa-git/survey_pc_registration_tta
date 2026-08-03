# Validation-set-size analysis

This folder contains a single script, `analyze_validset.py`, that answers the
question: **what validation-set size gives the most stable model selection?**

It assumes you have already run the experimental grid you described:

| Knob                     | Value                                  |
| ------------------------ | -------------------------------------- |
| Validation-set sizes     | `{100, 150, 200, 250, 300, 350, 400}`  |
| Repetition seeds         | `{0, 10, 20, 30, 40, 50}`              |
| Epochs per run           | `150`                                  |
| Optimizer / lr / batch   | already fixed (see your config below)  |

## Training config (already used in your runs)

The header of every training log already records the configuration you ran
with — quoting the relevant block from `train-20260507-174733.log`:

```
"train": {
    "batch_size": 16,
    "num_workers": 4,
    "point_limit": 30000,
    "use_augmentation": true,
    ...
}
"test":  { "batch_size": 8, ... }
seed: 7351                   # global training seed
lr scheduling: 8.0e-4 → ...  # visible in epoch lines, e.g. lr: 8.000e-04
                             # at epoch 1, 7.600e-04 at epoch 2 (decay ~0.95/epoch)
```

That is a reasonable starting point: an 8e-4 initial learning rate with a
gentle exponential decay is a standard, neither aggressive nor timid choice
for fine-tuning. Keep these settings constant across every (size, seed) cell
of the grid so the only variable being measured is the validation-set size.

## Expected on-disk layout

The script reads from your project root (`--base`):

```
<base>/output/P2PSilico/Source_Only/logs/train-*.log
<base>/output/P2PSilico/ft_validation/test_<size>_<seed>.json
```

It identifies which log corresponds to which (size, seed) by parsing the
`--n_val_samples N --val_seed S` arguments on **line 1** of every `*.log`,
exactly as you described.

It does **not** use `output/P2PSilico/Source_Only/ft_snapshots/<size>/<seed>/output.json`,
since that file is only saved for the first model — the per-epoch validation
RMSE is recovered from the training log instead.

## How to run

```bash
python analyze_validset.py \
    --base /path/to/PARENet \
    --out  ./analysis
```

Optional flags:

```
--sizes 100 150 200 250 300 350 400        # default
--seeds 0 10 20 30 40 50                   # default
--epochs 150                               # expected number of epochs
--dup-strategy {first,last,mean}           # see "Distributed duplicates" below
```

## What the script computes

For every (size, seed) cell of the grid:

1. **Validation curve.** From the corresponding `*.log`, every line of the
   form `[CRIT] [Val] Epoch: E, ..., RMSE: x, RMSE_std: y, ...` is parsed.
   The script verifies that all 150 epochs are present and warns about any
   incomplete run; missing epochs are recorded as NaN and ignored downstream.
2. **Test RMSE.** From `test_<size>_<seed>.json`, every line has fields
   `BS`, `RMSE`, `RMSE_std` (per batch). The last batch is typically smaller,
   hence the BS-weighted aggregation:

   ```
   mean = sum_i BS_i * RMSE_i  / sum_i BS_i
   var  = sum_i BS_i * RMSE_std_i^2 / N      (within-batch contribution)
        + sum_i BS_i * (RMSE_i - mean)^2 / N (between-batch contribution)
   std  = sqrt(var)
   ```

   This is just the law of total variance; it gives the correct grand mean
   and grand std across all individual test samples.

For every validation-set size, across the seeds:

3. **Test stability.** Mean ± std (across seeds) of the per-seed test means.
   The size with the smallest cross-seed std is the one whose model selection
   is most stable.
4. **Validation-curve stability.** At every epoch, mean and std across seeds.
   Plots include both:
   - `val_curves.png`: all sizes overlaid, each with its `fill_between` band.
   - `val_curves_grid.png`: one panel per size, easier for direct band
     comparison; each panel title shows the mean band width.
   The numerical summary `val_band_summary.csv` contains, per size, the mean
   of the per-epoch std — the smaller this number, the tighter the band.

## Outputs

In the `--out` directory (`./analysis` by default):

| File                          | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| `test_summary.csv`            | per-size: mean / std of test RMSE across seeds                     |
| `test_means.png`              | bar plot of the same quantity                                      |
| `val_band_summary.csv`        | per-size: mean of per-epoch std across seeds                       |
| `val_curves.csv`              | per (size, epoch): mean and std of val RMSE across seeds           |
| `val_curves_per_seed.csv`     | per (size, seed, epoch): raw val RMSE — handy for re-plotting      |
| `val_curves.png`              | all sizes overlaid, mean curves with `fill_between` ± std bands    |
| `val_curves_grid.png`         | one subplot per size for easier band comparison                    |

The console prints two ranked tables at the end and a one-line suggestion
for each criterion (smallest test std and smallest val-band width). A robust
recommendation is the **smallest size that minimises both** (or comes close
on both) — that is the cheapest validation set that still selects models
reliably.

## Distributed duplicates

In your logs every epoch produces two `[Val]` lines, e.g.

```
[2026-05-07 17:50:07] [CRIT] [Val] Epoch: 1, ..., RMSE: 0.859, ...
[2026-05-07 17:50:24] [CRIT] [Val] Epoch: 1, ..., RMSE: 0.911, ...
```

This is the usual artefact of distributed training: each rank validates and
logs separately, sometimes on different shards of the val split. The script
defaults to `--dup-strategy first` (rank 0) for full consistency across runs
and epochs. `last` and `mean` are also available — pick whichever matches
how you usually report your val number, and keep it the same across all
runs in the comparison.

## Sanity checks the script performs

* All `test_*.json` files have the **same number of iterations** — printed
  as a single line if they match, warned if not (different counts mean the
  test set itself differed between runs, which would invalidate comparison).
* Every log has **150 epochs** of `[Val]` entries — runs with fewer are
  flagged in the warning block; their available epochs are still kept,
  marked NaN beyond.
* Logs that cannot be parsed (no `--n_val_samples`/`--val_seed` on line 1)
  are skipped silently.