#!/usr/bin/env python3
import argparse
import json
import os
import time

import pytorch_lightning as pl
import scanpy as sc
import torch
import yaml

from gfm import GFM
from gfm.helpers import compute_metrics


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments with config file support"""
    parser = argparse.ArgumentParser(
        description="Train GFM model with config file or CLI arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Data arguments
    parser.add_argument("--adata-path", type=str, help="Path to AnnData h5ad file")
    parser.add_argument(
        "--split-df-path", type=str, help="Path to train/test split DataFrame pickle file"
    )
    parser.add_argument(
        "--split-dict-path",
        type=str,
        help="Path to train/test split dictionary pickle file (alternative to split_df_path)",
    )
    parser.add_argument("--output-dir", type=str, help="Working directory for outputs")
    parser.add_argument(
        "--graph-dir", type=str, help="Directory containing graph files (GO, PPI, etc.)"
    )

    # Model arguments (no defaults - will use config or fallback values)
    parser.add_argument("--latent-dim", type=int, help="Latent dimension size")
    parser.add_argument("--vae-name", type=str, help="VAE model filename")
    parser.add_argument("--model-name", type=str, help="GFM model filename")
    parser.add_argument(
        "--vae-save-path",
        type=str,
        help="Full path to save VAE model (overrides output_dir + vae_name)",
    )
    parser.add_argument("--graph-type", type=str, help="Type of graph to use for GNN")
    parser.add_argument("--pert-encoding", type=str, help="Type of GNN encoding")
    parser.add_argument(
        "--pert-adata-path",
        type=str,
        help="Path to AnnData h5ad file for perturbation data (required for certain graph types)",
    )

    # Training arguments (no defaults - will use config or fallback values)
    parser.add_argument("--max-epochs", type=int, help="Maximum training epochs")
    parser.add_argument("--early-stopping-patience", type=int, help="Early stopping patience")
    parser.add_argument("--eval-freq", type=int, help="Evaluation frequency (epochs)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--vae-batch-size", type=int, help="Batch size for VAE pretraining")

    # Flags
    parser.add_argument("--no-fm", action="store_true", help="Disable flow matching")
    parser.add_argument("--randomize-graph", action="store_true", help="Randomize graph structure")
    parser.add_argument("--skip-vae-pretrain", action="store_true", help="Skip VAE pretraining")
    parser.add_argument(
        "--skip-model-training", action="store_true", help="Skip GFM model training"
    )
    parser.add_argument(
        "--skip-prediction", action="store_true", help="Skip generating predictions"
    )
    parser.add_argument("--skip-metrics", action="store_true", help="Skip computing metrics")
    parser.add_argument("--use-contrastive", action="store_true", help="Use contrastive loss")
    parser.add_argument(
        "--use-condition-classifier", action="store_true", help="Use condition classifier loss"
    )
    parser.add_argument(
        "--use-null-embedding", action="store_true", help="Use null embedding in model"
    )
    parser.add_argument(
        "--use-scvi-vae",
        action="store_true",
        help="Use scvi-tools VAE instead of custom VAE implementation",
    )
    parser.add_argument(
        "--save-times",
        action="store_true",
        help="Save training and inference times to timing.json in working directory",
    )
    parser.add_argument(
        "--aggregation-method",
        type=str,
        choices=["sum", "deepset"],
        help="Aggregation method for GNN (default: sum)",
    )

    # Metrics
    parser.add_argument("--top-n", type=int, help="Top N genes for metrics")

    return parser.parse_args()


def merge_config_and_args(args):
    """Merge config file with CLI arguments (CLI takes precedence)"""
    # Default values if not in config
    defaults = {
        "adata_path": None,
        "latent_dim": 50,
        "vae_name": "vae.pt",
        "model_name": "gfm.pt",
        "output_dir": None,
        "split_df_path": None,
        "split_dict_path": None,
        "vae_save_path": None,
        "graph_dir": None,
        "graph_type": "go",
        "pert_encoding": "gat",
        "pert_adata_path": None,
        "max_epochs": 500,
        "early_stopping_patience": 4,
        "eval_freq": 50,
        "lr": 5e-4,
        "vae_batch_size": 256,
        "no_fm": False,
        "randomize_graph": False,
        "skip_vae_pretrain": False,
        "skip_model_training": False,
        "skip_prediction": False,
        "skip_metrics": False,
        "use_contrastive": False,
        "use_condition_classifier": False,
        "use_null_embedding": False,
        "use_scvi_vae": False,
        "top_n": 20,
        "save_times": False,
        "aggregation_method": "sum",
    }

    if args.config:
        # Load config file
        config = load_config(args.config)

        # Start with defaults
        final_config = defaults.copy()

        # Update with config file values
        final_config.update(config)

        # Override with any CLI arguments that were explicitly provided
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                # For boolean flags, only override if True (flag was set)
                if isinstance(value, bool):
                    if value:
                        final_config[key] = value
                else:
                    # Non-boolean: always override if provided
                    final_config[key] = value

        return argparse.Namespace(**final_config)
    else:
        # No config file: use CLI args + defaults
        final_config = defaults.copy()
        for key, value in vars(args).items():
            if value is not None:
                final_config[key] = value
        return argparse.Namespace(**final_config)


def main():
    pl.seed_everything(42)

    args = parse_args()
    args = merge_config_and_args(args)

    # Validate required arguments
    if (
        not args.adata_path
        or (not args.split_df_path and not args.split_dict_path)
        or not args.output_dir
    ):
        raise ValueError(
            "Required: --adata-path, --split-df-path, --working-dir (or provide --config)"
        )

    # Setup device
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("\n" + "=" * 60)
    print("Configuration:")
    print("=" * 60)
    for key, value in sorted(vars(args).items()):
        print(f"  {key:30s}: {value}")
    print("=" * 60 + "\n")

    # Load data
    print(f"Loading data from {args.adata_path}...")
    adata = sc.read_h5ad(args.adata_path, backed="r")

    train_start = time.time()
    # Initialize GFM
    gfm = GFM(
        adata,
        latent_dim=args.latent_dim,
        split_df_path=args.split_df_path,
        split_dict_path=args.split_dict_path,
        output_dir=args.output_dir,
        vae_name=args.vae_name,
        vae_save_path=args.vae_save_path,
        model_name=args.model_name,
        device=device,
        use_contrastive=args.use_contrastive,
        use_condition_classifier=args.use_condition_classifier,
        no_fm=args.no_fm,
        use_null_embedding=args.use_null_embedding,
        use_scvi_vae=args.use_scvi_vae,
        aggregation_method=args.aggregation_method,
    )

    # Initialize flow matching
    gfm.initialize_fm(
        randomize_graph=args.randomize_graph,
        graph_type=args.graph_type,
        pert_encoding=args.pert_encoding,
        graph_dir=args.graph_dir,
        pert_adata_path=args.pert_adata_path,
    )

    # Optional VAE pretraining
    if not args.skip_vae_pretrain:
        print("Pretraining VAE...")
        gfm.pretrain_vae(lr=args.lr, batch_size=args.vae_batch_size)

    # Prepare and train
    if not args.skip_model_training:
        print("Preparing training...")
        gfm.prepare_training(lr=args.lr)

        print("Training model...")
        gfm.train_model(
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience,
            eval_freq=args.eval_freq,
        )
    else:
        gfm.load_model()  # Load existing model if skipping training

    train_time = time.time() - train_start
    print(f"Training time: {train_time:.2f}s")

    # Generate predictions
    infer_time = None
    if not args.skip_prediction:
        print("Generating predictions...")
        trimmed_name = os.path.splitext(gfm.model_name)[0]
        if gfm.split.split_df is None:
            raise ValueError("SplitHandler did not load split_df.")
        pred_df = gfm.split.split_df[gfm.split.split_df["split"] == "test"].copy()
        infer_start = time.time()
        adata_pred = gfm.predict(pred_df)
        infer_time = time.time() - infer_start
        print(f"Inference time: {infer_time:.2f}s")
        pred_path = f"{gfm.output_dir}/adata_pred_{trimmed_name}.h5ad"
        adata_pred.write(pred_path)
        print(f"Predictions saved to {pred_path}")

    # Compute metrics
    if not args.skip_metrics:
        if args.skip_prediction:
            trimmed_name = os.path.splitext(gfm.model_name)[0]
            if gfm.split.split_df is None:
                raise ValueError("SplitHandler did not load split_df.")
            pred_df = gfm.split.split_df[gfm.split.split_df["split"] == "test"].copy()
            adata_pred = sc.read_h5ad(f"{gfm.output_dir}/adata_pred_{trimmed_name}.h5ad")
        print("Computing metrics...")
        results_path = f"{gfm.output_dir}/results_test_{trimmed_name}.csv"
        gfm.split.add_split_to_adata(adata)  # Ensure split info is in adata.obs

        _ = compute_metrics(
            adata,
            adata_pred,
            pred_df,
            covariate_columns=gfm.split.covariate_columns,
            top_n=20,
            n_jobs=32,
            save_path=results_path,
        )

        print(f"Results saved to {results_path}")

    if args.save_times:
        timing = {"train_time_s": train_time, "inference_time_s": infer_time}
        with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)
        print(f"Timing saved to {os.path.join(args.output_dir, 'timing.json')}")
    print("Done!")


if __name__ == "__main__":
    main()
