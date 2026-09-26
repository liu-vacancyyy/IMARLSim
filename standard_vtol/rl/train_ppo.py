"""Train standard_vtol with the standard PPO implementation from rsl_rl."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "envs"))

from rsl_rl.runners import OnPolicyRunner

from rsl_vec_env import StandardVTOLVecEnv


def build_config(args):
    return {
        "seed": args.seed,
        "runner_class_name": "OnPolicyRunner",
        "run_name": "standard_vtol_fixed_target",
        "num_steps_per_env": args.steps_per_env,
        "save_interval": args.save_interval,
        "check_for_nan": True,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 3.0e-4,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
                "learn_std": True,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
            "obs_normalization": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps-per-env", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", default="runs/standard_vtol_fixed_target")
    args = parser.parse_args()

    env = StandardVTOLVecEnv(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
    )
    cfg = build_config(args)
    runner = OnPolicyRunner(env, cfg, log_dir=args.log_dir, device=args.device)
    try:
        runner.learn(args.iterations)
    finally:
        env.close()


if __name__ == "__main__":
    main()
