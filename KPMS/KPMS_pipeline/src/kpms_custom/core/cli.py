import argparse
import sys
from kpms_custom.utils.logger_utils import setup_logger
from kpms_custom.core import runner
import warnings
import matplotlib

# Force HEADLESS mode before any other plot import
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")

def main():
    parser = argparse.ArgumentParser(description="KPMS Pipeline (Camellia Edition)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common Args
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")

    # Command: Train
    parser_train = subparsers.add_parser("train", parents=[parent_parser], help="Train models")
    parser_train.add_argument("--restarts", type=int, help="Number of restart runs (overrides config)")
    
    # Command: Scan
    parser_scan = subparsers.add_parser("scan", parents=[parent_parser], help="Run Kappa parameter scan")
    parser_scan.add_argument("--type", choices=['ar', 'full'], default='ar', help="Scan type")
    
    # Command: Evaluate
    parser_eval = subparsers.add_parser("evaluate", parents=[parent_parser], help="Compare models via EML")
    
    # Command: Analyze
    parser_analyze = subparsers.add_parser("analyze", parents=[parent_parser], help="Run Viz & Stats")
    parser_analyze.add_argument("--model", type=str, help="Specific model name (optional, defaults to latest)")
    
    # Command: Merge
    parser_merge = subparsers.add_parser("merge", parents=[parent_parser], help="Run Motif Merging")
    parser_merge.add_argument("--model", type=str, help="Specific model name")

    args = parser.parse_args()
    
    # Setup Logger
    setup_logger()
    
    if args.command == "train":
        runner.run_training(args.config, restarts=args.restarts)
    elif args.command == "scan":
        runner.run_scan(args.config, scan_type=args.type)
    elif args.command == "evaluate":
        runner.run_evaluation(args.config)
    elif args.command == "analyze":
        runner.run_analysis(args.config, model_name=args.model)
    elif args.command == "merge":
        runner.run_merging(args.config, model_name=args.model)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
