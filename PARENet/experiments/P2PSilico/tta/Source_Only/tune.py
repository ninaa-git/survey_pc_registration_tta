import os
import sys
import wandb
import optuna
from optuna.trial import TrialState

from config import make_cfg
from trainval import Trainer
from common_utils.parser import make_hyperparameters_training_parser


def main():
    parser = make_hyperparameters_training_parser("Optuna Worker")
    args, _ = parser.parse_known_args()

    study = optuna.load_study(
        storage=f"sqlite:///{args.optuna_db}?timeout=60",
        study_name=args.optuna_study_name,
    )

    matching = [t for t in study.trials if t.number == args.trial_num]
    if not matching:
        print(f"ERROR: No trial with number={args.trial_num} in study "
              f"'{args.optuna_study_name}'. Did you run generate_trials.py?")
        sys.exit(1)
    trial = matching[0]

    if trial.state == TrialState.COMPLETE:
        print(f"Trial {trial.number} already COMPLETE with value={trial.value}, "
              f"skipping.")
        return

    params = trial.params
    print(f"Consuming trial #{trial.number} : {params}")

    cfg = make_cfg()
    cfg.optim.lr           = params["lr"]
    cfg.train.batch_size   = params["batch_size"]
    cfg.optim.lr_decay     = params["lr_decay"]
    cfg.optim.weight_decay = params["weight_decay"]
    cfg.optim.lr_decay_steps = params["lr_decay_steps"]

    val_seed = params.get("val_seed", None)
    if val_seed is not None:
        cfg.train.val_seed = val_seed

    cfg.ft_snapshot_dir = os.path.join(
        cfg.ft_snapshot_dir, f"tune_trial_{trial.number}"
    )

    seed_str = f"_s{val_seed}" if val_seed is not None else ""
    wandb_name = (
        f"tune_trial_{trial.number}"
        f"_bs{cfg.train.batch_size}"
        f"_lr{cfg.optim.lr:.2e}"
        f"_d{cfg.optim.lr_decay:.2f}"
        f"_ds{cfg.optim.lr_decay_steps}"
        f"{seed_str}"
    )
    sys.argv = sys.argv + ["--wandb_name", wandb_name]

    try:
        trainer = Trainer(cfg)
        trainer.run()
        rmse = trainer.best_metric_rmse
        study.tell(trial.number, rmse)
        print(f"Trial {trial.number} COMPLETE : RMSE = {rmse}")
    except Exception as e:
        print(f"Trial {trial.number} FAILED : {e}")
        try:
            study.tell(trial.number, state=TrialState.FAIL)
        except Exception as e2:
            print(f"Could not mark trial as FAIL: {e2}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()