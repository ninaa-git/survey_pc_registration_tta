import argparse
import itertools
import optuna
from optuna.samplers import QMCSampler, GridSampler


def define_sobol_search_space(trial):
    """Continuous + categorical search space for Sobol/TPE/random exploration."""
    trial.suggest_float("lr", 8e-5, 2e-3, log=True)
    trial.suggest_categorical("batch_size", [12])
    trial.suggest_float("lr_decay", 0.8, 0.99)
    trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    trial.suggest_categorical("lr_decay_steps", [1, 2, 3, 4, 5])


GRID_SPACE = {
    "lr":           [7e-4],
    "lr_decay":     [0.99],
    "batch_size":   [12],
    "weight_decay": [2e-5],
    "val_seed":     [20],
    "lr_decay_steps": [3],
}


def define_grid_search_space(trial):
    """Grid sampler needs every dimension to be suggested in the objective."""
    trial.suggest_categorical("lr",           GRID_SPACE["lr"])
    trial.suggest_categorical("lr_decay",     GRID_SPACE["lr_decay"])
    trial.suggest_categorical("batch_size",   GRID_SPACE["batch_size"])
    trial.suggest_categorical("weight_decay", GRID_SPACE["weight_decay"])
    trial.suggest_categorical("val_seed",     GRID_SPACE["val_seed"])
    trial.suggest_categorical("lr_decay_steps", GRID_SPACE["lr_decay_steps"])


def n_grid_combinations():
    """Total number of (lr, lr_decay, bs, wd, seed) combinations in the grid."""
    return len(list(itertools.product(*GRID_SPACE.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna-db", required=True)
    parser.add_argument("--optuna-study-name", required=True)
    parser.add_argument("--n-trials", type=int, default=32,
                        help="Ignored when sampler=grid (uses all combinations).")
    parser.add_argument(
        "--sampler", choices=["qmc", "tpe", "random", "grid"], default="qmc",
        help="qmc/tpe/random : continuous exploration. grid : exhaustive over GRID_SPACE.",
    )
    args = parser.parse_args()

    # ===== Sampler + search space selection =====
    if args.sampler == "grid":
        sampler = GridSampler(GRID_SPACE)
        define_space = define_grid_search_space
        n_total = n_grid_combinations()
        if args.n_trials != n_total:
            print(f"[info] grid sampler : ignoring --n-trials={args.n_trials}, "
                  f"using all {n_total} combinations.")
        n_trials = n_total
    else:
        if args.sampler == "qmc":
            sampler = QMCSampler(qmc_type="sobol", seed=0, scramble=True)
        elif args.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(seed=0)
        else:
            sampler = optuna.samplers.RandomSampler(seed=0)
        define_space = define_sobol_search_space
        n_trials = args.n_trials

    # ===== Study creation =====
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{args.optuna_db}?timeout=60",
        study_name=args.optuna_study_name,
        load_if_exists=True,
    )

    # ===== Generate N trials =====
    print(f"Generating {n_trials} trials with sampler={args.sampler}...")
    print("-" * 80)
    for i in range(n_trials):
        trial = study.ask()
        define_space(trial)
        print(f"Trial {trial.number}: {trial.params}")

    print("-" * 80)
    print(f"Done. DB: {args.optuna_db}")
    print(f"Study: {args.optuna_study_name}")
    print(f"Total trials in study: {len(study.trials)}")


if __name__ == "__main__":
    main()