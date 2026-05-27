"""
Configuration management for GNN-ReGVD.
Supports YAML config files with CLI override, plus Optuna search space definitions.
"""

import argparse
import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


# Default search spaces for Optuna tuning
DEFAULT_SEARCH_SPACE = {
    "learning_rate": {"low": 1e-6, "high": 1e-3, "log": True},
    "weight_decay": {"low": 0.0, "high": 0.1, "log": False},
    "hidden_size": {"choices": [64, 128, 256, 384, 512]},
    "num_GNN_layers": {"low": 1, "high": 4, "step": 1},
    "window_size": {"low": 2, "high": 10, "step": 1},
    "dropout": {"low": 0.0, "high": 0.5, "log": False},
    "gradient_accumulation_steps": {"choices": [1, 2, 4, 8]},
    "train_batch_size": {"choices": [4, 8, 16, 32]},
    "gnn": {"choices": ["ReGCN", "ReGGNN"]},
    "att_op": {"choices": ["mul", "sum", "concat"]},
    "format": {"choices": ["uni", "idx"]},
    "max_grad_norm": {"low": 0.5, "high": 5.0, "log": False},
    "adam_epsilon": {"low": 1e-9, "high": 1e-6, "log": True},
}


def load_config(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_config(config: Dict[str, Any], yaml_path: str):
    """Save configuration to a YAML file."""
    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override dict into base dict. Override keys take precedence."""
    merged = base.copy()
    for k, v in override.items():
        if v is not None:
            merged[k] = v
    return merged


def config_to_args(config: Dict[str, Any], parser: argparse.ArgumentParser = None) -> argparse.Namespace:
    """Convert a config dict to an argparse.Namespace. If parser is given, uses its defaults for missing keys."""
    if parser is not None:
        defaults = vars(parser.parse_args([]))
    else:
        defaults = {}
    merged = {**defaults, **config}
    return argparse.Namespace(**merged)


def args_to_config(args: argparse.Namespace, keys: List[str] = None) -> Dict[str, Any]:
    """Extract relevant keys from args Namespace into a config dict."""
    d = vars(args)
    if keys is not None:
        return {k: d[k] for k in keys if k in d}
    return dict(d)


def load_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse CLI args, optionally loading from a YAML config file first.
    Use --config to specify a YAML file. CLI args override YAML values.
    """
    # Add --config argument if not already present
    if "--config" not in parser.format_help():
        parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    cli_args = parser.parse_args()

    if cli_args.config and os.path.exists(cli_args.config):
        yaml_config = load_config(cli_args.config)
        # Start with parser defaults, overlay YAML, then overlay CLI
        defaults = vars(parser.parse_args([]))
        # Remove the config key itself from YAML
        yaml_config.pop("config", None)
        merged = {**defaults, **yaml_config}
        # CLI explicit values override (only non-default values)
        for k, v in vars(cli_args).items():
            if v != defaults.get(k) and v is not None:
                merged[k] = v
        return argparse.Namespace(**merged)

    return cli_args


def get_tuning_search_space(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load search space from a YAML config or use defaults."""
    if config_path and os.path.exists(config_path):
        cfg = load_config(config_path)
        if "search_space" in cfg:
            return cfg["search_space"]
    return DEFAULT_SEARCH_SPACE
