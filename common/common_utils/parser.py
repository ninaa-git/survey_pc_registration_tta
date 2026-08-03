import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1', 'yes', 'y'):
        return True
    if v.lower() in ('false', 'f', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')

def make_launcher_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backbone", required=True, choices=["PARENet", "Lepard"])
    parser.add_argument("--dataset",  required=True, help="e.g. P2PSilico")
    parser.add_argument("--method",   required=True, help="e.g. LN_TTA, Source_Only, PEA_TTA")
    parser.add_argument("--wandb_project", type=str, default="PARE-Net")
    parser.add_argument("--wandb_name", type=str, default="default_run")
    return parser

def make_hyperparameters_training_parser(description: str) -> argparse.ArgumentParser:
    parser = make_launcher_parser(description="Training Registration Pipeline")
    parser.add_argument("--optuna-db", default=None,
                        help="Path to Optuna SQLite DB (required when running via tune_worker)")
    parser.add_argument("--optuna-study-name", default=None,
                        help="Optuna study name (required when running via tune_worker)")
    parser.add_argument("--trial-num", type=int, required=True,
                        help="0-indexed trial number to consume from the Optuna study.")
    return parser

def make_ft_training_parser(description: str) -> argparse.ArgumentParser:
    parser = make_launcher_parser(description="Fine-tuning Registration Pipeline")
    parser.add_argument("--corruption", required=True, help="e.g. clean, uniform, impulse_noise, local_density_dec, global_density_dec, cutout, occlusion, gaussian, background_noise")
    parser.add_argument("--severity", required=True, type=int, help="e.g. 1, 2, 3, 4, 5")
    return parser

def make_pointtta_training_parser():
    parser = make_launcher_parser(description="Point_TTA Training Registration Pipeline")
    parser.add_argument('--joint_snapshot', type=str, default=None) 
    parser.add_argument(
        '--aux_mode', type=str, default='rec',
        help=(
            'bi_aux   : Bi-Auxiliary (Auxsmr + Auxad). '
            'rec  : Reconstruction. '
        )
    )
    parser.add_argument('--meta_lr',    type=float, default=1.6e-4,
                        help='Backbone learning rate for meta TTA.')
    parser.add_argument('--train_phase', type=str, default='joint',
                        help='Train phase: joint or meta')
    parser.add_argument('--niter', type=int,   default=5, help='K in Algorithm 1: gradient steps per sample.')
    return parser

def make_parser():
    parser = make_launcher_parser(description="TTA Registration Pipeline")

    # --- Checkpoint ---
    parser.add_argument("--snapshot", default=None, help="path to .pth checkpoint")

    # --- Corruption ---
    parser.add_argument("--corruption", default="clean", help="corruption type, e.g. clean, jitter, crop")
    parser.add_argument("--severity",   default=None,    help="corruption severity level (1-5)")

    # --- Misc ---
    parser.add_argument("--seed",       type=int,   default=None)

    # --- Visualisation ---
    parser.add_argument('--viz_iter', type=int, nargs='*', default=[],
                    help='iterations to visualise, e.g. --viz_iter 10 50 2039; empty = disabled')
    parser.add_argument('--viz_max_clouds', type=int, default=1,
                    help='max number of point clouds to collect for t-SNE (default: 40)')
    parser.add_argument('--viz_sample', type=int, default=0,
                    help='0-based index of the first test pair to visualize '
                         '(change this to pick a different point cloud)')

    # --- Validation dataset P2PSilico  ---
    parser.add_argument("--n_val_ratio", type=float, default=0.2)
    parser.add_argument("--val_seed", type=int, default=10)

        # LN_TTA
    parser.add_argument('--layers_to_adapt', type=int, nargs='+', default=None,
                        help='Layer indices to adapt e.g. --layers_to_adapt 1 3. '
                             'None = adapt all layers.')
    parser.add_argument('--ln_momentum', type=float, default=0.03,
                        help='EMA momentum m in (0,1] for online target statistics. '
                             'μ(i) = (1-m)μ(i-1) + m μ_b  (default: 0.1)')

    # PEA_TTA
    parser.add_argument('--align_weight', type=float, default=1.0,
                        help='Blending weight w in [0,1] for WCT coarse-feature '
                             'alignment. 0 = disabled (default), 1 = full alignment.')
    parser.add_argument('--align_momentum', type=float, default=0.02,
                        help='EMA momentum m in (0,1] for online target statistics. '
                             'μ(i) = (1-m)μ(i-1) + m μ_b  (default: 0.1)')
    parser.add_argument('--align_stats_dir', type=str,
                        default='intermediate_features/pea_stats',
                        help='Directory containing *_pea_distances.pth files '
                             'produced by generate_stats.py.')

    # Point_TTA
    parser.add_argument('--meta_snapshot', type=str, default=None) 
    parser.add_argument(
        '--mode', type=str, default='standard',
        choices=['standard', 'pointtta'],
    )
    parser.add_argument('--lr',    type=float, default=1.6e-4,
                        help='Backbone learning rate for TTA.')
    parser.add_argument('--niter', type=int,   default=5, help='K: number of gradient steps per sample.')


    # Purge_Gate
    
    parser.add_argument('--force', action='store_true', 
                       help='Force regeneration even if cache exists')
    parser.add_argument('--feature', type=str, default='b_ref_feats_c_pad',
                       choices=['b_ref_feats_c', 'b_ref_feats_f', 'ref_feats_c_self_0','ref_feats_c', 'src_feats_c', 'ref_feats_c_norm', 'src_feats_c_norm', 'ref_feats_f_norm'],
                       help='Which feature to analyze')
    parser.add_argument('--metric', type=str, default='mahalanobis',
                       choices=['l2', 'mahalanobis', 'zscore'],
                       help='Divergence metric')
    parser.add_argument('--output_features', type=str, default=None,
                       help='Output path (default: intermediate_features/<dataset>_clean_c_intermediates.pth)')
    parser.add_argument('--purge_size', type=float, default=[0.0, 0.1, 0.5],
                       help='Severity, default = (L_pg = [0.0, 0.1, 0.5])')
    parser.add_argument('--stats_mode', default='stats', choices=['stats', 'stats_per_cloud'])


    return parser