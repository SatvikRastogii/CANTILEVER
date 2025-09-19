"""
Main entry point for the Image Captioning System.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Image Captioning System")
    parser.add_argument("--mode", choices=["api", "web", "train", "evaluate"], required=True,
                       help="Mode to run the system in")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--data-config", type=str, help="Data configuration file path")
    parser.add_argument("--model-path", type=str, help="Model path for evaluation")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        from src.api.main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    elif args.mode == "web":
        from src.web.main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8080)
    
    elif args.mode == "train":
        if not args.config or not args.data_config:
            print("Error: --config and --data-config are required for training")
            sys.exit(1)
        
        # Import and run training
        from train import main as train_main
        sys.argv = ["train.py", "--config", args.config, "--data-config", args.data_config]
        if args.output_dir:
            sys.argv.extend(["--output-dir", args.output_dir])
        if args.device:
            sys.argv.extend(["--device", args.device])
        train_main()
    
    elif args.mode == "evaluate":
        if not args.model_path or not args.data_config:
            print("Error: --model-path and --data-config are required for evaluation")
            sys.exit(1)
        
        # Import and run evaluation
        from evaluate import main as eval_main
        sys.argv = ["evaluate.py", "--model-path", args.model_path, "--data-config", args.data_config]
        if args.output_dir:
            sys.argv.extend(["--output-dir", args.output_dir])
        if args.device:
            sys.argv.extend(["--device", args.device])
        eval_main()


if __name__ == "__main__":
    main()
