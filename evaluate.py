from _launcher import launch
from common.common_utils.parser import make_launcher_parser

def main():
    args, extra_args = make_launcher_parser("Run a TTA generate_stats script").parse_known_args()
    launch(args.backbone, args.dataset, args.method, "generate_stats.py", extra_args)

if __name__ == "__main__":
    main()