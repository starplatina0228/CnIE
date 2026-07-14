"""utils.py  —  설정 로더."""
import json
import os
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed":            42,
    "sim_time":        36000,
    "lambda_default":  500 / 3600,
    "lambda_grid":     [300/3600, 400/3600, 500/3600, 600/3600, 700/3600],
    "robot_counts":    [2, 3, 4],
    "robot_capacity":  5,
    "partial_k":       3,
    "total_episodes":  1000,
    "n_eval":          10,
}


def load_config(path: str = "config.json") -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()