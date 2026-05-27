"""
Fast hyperparameter tuning for GNN-ReGVD using Optuna with median pruning.

Usage:
    python tune.py --n_trials 30 --subset 0.3 --epoch 10 --config ../configs/default.yaml

This mirrors the "sloth" approach from LLMxCPG (Unsloth fast fine-tuning):
  - MedianPruner: early-stops unpromising trials (like Unsloth's efficient training)
  - Subset training: trains on a fraction of data for quick evaluation
  - Fewer epochs: fast iteration to find promising regions
"""

import argparse
import logging
import os
import sys
import json
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml

import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import (
    set_seed, TextDataset, train, evaluate, MODEL_CLASSES,
)
from model import GNNReGVD, DevignModel
from config import load_config, save_config, get_tuning_search_space

logger = logging.getLogger(__name__)

# Global vars for trial context
_trial_context = {}


def _objective(trial: optuna.Trial, base_args: argparse.Namespace, search_space: dict) -> float:
    """Optuna objective: train with sampled hyperparams, return validation accuracy."""
    args = argparse.Namespace(**vars(base_args))

    # --- Sample hyperparameters ---
    sp = search_space

    if "learning_rate" in sp:
        args.learning_rate = trial.suggest_float("learning_rate", **sp["learning_rate"])
    if "weight_decay" in sp:
        args.weight_decay = trial.suggest_float("weight_decay", **sp["weight_decay"])
    if "hidden_size" in sp:
        if "choices" in sp["hidden_size"]:
            args.hidden_size = trial.suggest_categorical("hidden_size", sp["hidden_size"]["choices"])
        else:
            args.hidden_size = trial.suggest_int("hidden_size", **sp["hidden_size"])
    if "num_GNN_layers" in sp:
        args.num_GNN_layers = trial.suggest_int("num_GNN_layers", **sp["num_GNN_layers"])
    if "window_size" in sp:
        if "choices" in sp["window_size"]:
            args.window_size = trial.suggest_categorical("window_size", sp["window_size"]["choices"])
        else:
            args.window_size = trial.suggest_int("window_size", **sp["window_size"])
    if "dropout" in sp:
        if "choices" in sp["dropout"]:
            dropout = trial.suggest_categorical("dropout", sp["dropout"]["choices"])
        else:
            dropout = trial.suggest_float("dropout", **sp["dropout"])
        # Patch dropout into config
        args._dropout = dropout
    if "gradient_accumulation_steps" in sp:
        args.gradient_accumulation_steps = trial.suggest_categorical(
            "gradient_accumulation_steps", sp["gradient_accumulation_steps"]["choices"]
        )
    if "train_batch_size" in sp:
        args.train_batch_size = trial.suggest_categorical(
            "train_batch_size", sp["train_batch_size"]["choices"]
        )
    if "gnn" in sp:
        args.gnn = trial.suggest_categorical("gnn", sp["gnn"]["choices"])
    if "att_op" in sp:
        args.att_op = trial.suggest_categorical("att_op", sp["att_op"]["choices"])
    if "format" in sp:
        args.format = trial.suggest_categorical("format", sp["format"]["choices"])
    if "max_grad_norm" in sp:
        args.max_grad_norm = trial.suggest_float("max_grad_norm", **sp["max_grad_norm"])
    if "adam_epsilon" in sp:
        args.adam_epsilon = trial.suggest_float("adam_epsilon", **sp["adam_epsilon"])

    # Derived args
    args.per_gpu_train_batch_size = args.train_batch_size // max(args.n_gpu, 1)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(args.n_gpu, 1)

    logger.info("=== Trial %d ===", trial.number)
    logger.info("Params: lr=%.2e, hidden=%d, layers=%d, window=%d, gnn=%s, att=%s, fmt=%s, bs=%d",
                args.learning_rate, args.hidden_size, args.num_GNN_layers,
                args.window_size, args.gnn, args.att_op, args.format, args.train_batch_size)

    set_seed(args.seed)

    # Load model components
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.num_labels = 1

    # Apply trial dropout if sampled
    if hasattr(args, "_dropout"):
        config.hidden_dropout_prob = args._dropout
        config.attention_probs_dropout_prob = args._dropout

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.model == "devign":
        model = DevignModel(model, config, tokenizer, args)
    else:
        model = GNNReGVD(model, config, tokenizer, args)

    model.to(args.device)

    # Load data
    train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)

    # Train (evaluate_during_training is set by caller)
    args.evaluate_during_training = True
    args.do_train = True

    # Use per-trial output dir
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    original_output = args.output_dir
    args.output_dir = trial_dir
    os.makedirs(trial_dir, exist_ok=True)

    best_acc = 0.0
    try:
        logger.info("  Training...")
        train(args, train_dataset, model, tokenizer)

        # Evaluate best checkpoint
        checkpoint = os.path.join(trial_dir, "checkpoint-best-acc", "model.bin")
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location=args.device))
            model.to(args.device)
            result = evaluate(args, model, tokenizer)
            best_acc = result.get("eval_acc", 0.0)
            logger.info("  Trial %d best_acc = %.4f", trial.number, best_acc)

            # Report intermediate value for pruning
            trial.report(best_acc, step=0)
    finally:
        args.output_dir = original_output

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Fast hyperparameter tuning with Optuna")

    # Tuning control
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--subset", type=float, default=0.3,
                        help="Fraction of training data per trial (for speed)")
    parser.add_argument("--epoch", type=int, default=10,
                        help="Epochs per trial (fewer = faster iteration)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Parallel trials (set >1 only if you have multiple GPUs)")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name for resuming")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///tuning.db) for persistence")

    # Model / data
    parser.add_argument("--config", type=str, default=None,
                        help="Base YAML config (can include a 'search_space' key)")
    parser.add_argument("--search_space", type=str, default=None,
                        help="Separate YAML file defining the search space")
    parser.add_argument("--train_data_file", default="../dataset/train.jsonl", type=str)
    parser.add_argument("--eval_data_file", default="../dataset/valid.jsonl", type=str)
    parser.add_argument("--output_dir", default="./tuning_results", type=str)
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str)
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str)
    parser.add_argument("--model", default="GNNs", type=str)
    parser.add_argument("--gnn", default="ReGCN", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", default=-1, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--feature_dim_size", default=768, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_GNN_layers", default=2, type=int)
    parser.add_argument("--window_size", default=3, type=int)
    parser.add_argument("--att_op", default="mul", type=str)
    parser.add_argument("--format", default="uni", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--remove_residual", action="store_true")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup device
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    args.training_percent = args.subset

    logger.info("=" * 60)
    logger.info("  GNN-ReGVD Fast Hyperparameter Tuning (Optuna + MedianPruner)")
    logger.info("  Trials: %d | Subset: %.0f%% | Epochs/trial: %d", args.n_trials, args.subset * 100, args.epoch)
    logger.info("  Device: %s | GPUs: %d", device, args.n_gpu)
    logger.info("=" * 60)

    # Load search space
    search_space = None
    if args.search_space and os.path.exists(args.search_space):
        search_space = yaml.safe_load(open(args.search_space)) or {}
    elif args.config and os.path.exists(args.config):
        cfg = yaml.safe_load(open(args.config)) or {}
        search_space = cfg.get("search_space", None)

    if search_space is None:
        from config import DEFAULT_SEARCH_SPACE
        search_space = DEFAULT_SEARCH_SPACE
        logger.info("Using default search space")

    logger.info("Search space: %s", json.dumps(search_space, indent=2))

    # Create Optuna study with MedianPruner (like Unsloth's fast-pruning philosophy)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)

    study_kwargs = dict(
        direction="maximize",
        pruner=pruner,
    )
    if args.storage:
        study_kwargs["storage"] = args.storage
    if args.study_name:
        study_kwargs["study_name"] = args.study_name
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)

    logger.info("Study name: %s", study.study_name)

    # Run optimization
    def objective(trial):
        return _objective(trial, args, search_space)

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True)

    # --- Summarize results ---
    logger.info("=" * 60)
    logger.info("  TUNING COMPLETE")
    logger.info("=" * 60)

    pruned_trials = study.get_trials(states=[TrialState.PRUNED])
    complete_trials = study.get_trials(states=[TrialState.COMPLETE])

    logger.info("  Completed: %d | Pruned: %d", len(complete_trials), len(pruned_trials))
    logger.info("  Best trial: #%d | Accuracy: %.4f", study.best_trial.number, study.best_trial.value)
    logger.info("  Best params:")
    for k, v in study.best_params.items():
        logger.info("    %s: %s", k, v)

    # Save best config as YAML
    best_config = {
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name": args.tokenizer_name,
        "model": args.model,
        "block_size": args.block_size,
        **study.best_params,
        "feature_dim_size": args.feature_dim_size,
        "num_classes": args.num_classes,
    }
    best_config_path = os.path.join(args.output_dir, "best_config.yaml")
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(best_config, best_config_path)
    logger.info("  Best config saved to: %s", best_config_path)

    # Save full trial history
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(args.output_dir, "tuning_history.csv")
    trials_df.to_csv(trials_csv, index=False)
    logger.info("  Trial history saved to: %s", trials_csv)

    # Clean up trial directories (keep only best)
    for trial in study.trials:
        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
        if trial.number != study.best_trial.number and os.path.exists(trial_dir):
            shutil.rmtree(trial_dir, ignore_errors=True)

    logger.info("  Done. Run with: python run.py --config %s --do_train --do_test", best_config_path)

    return study.best_params, study.best_value


if __name__ == "__main__":
    main()
