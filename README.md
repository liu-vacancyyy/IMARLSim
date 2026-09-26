# IMARLSim: public `standard_vtol` mission

This repository contains only the single-agent, fixed-target point-to-point mission for the PX4/Gazebo Classic `standard_vtol`. Every episode starts from the same grounded initial state, follows the fixed route to `(north=300 m, east=0 m)`, and lands at the target. The public configuration disables target randomization, curriculum starts, sensor noise, wind, mass/inertia randomization, and all other domain randomization. No trained weights are included.

## Environment Setup

Use Python 3.10 or newer with PyTorch 2.6+ and a matching CUDA build for GPU training:

```bash
cd /home/a/IMARLSim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The training entry point uses torch-vectorized dynamics and does not require Gazebo to be running. The world/SDF files under `standard_vtol/assets` are provided for optional Gazebo Classic visualization. Add the model directory to the Gazebo search path when needed:

```bash
export GAZEBO_MODEL_PATH="$PWD/standard_vtol/assets/standard_vtol_single/models:${GAZEBO_MODEL_PATH:-}"
```

The training configuration is `standard_vtol/configs/standard_vtol_fixed_target.yaml`. It defines `num_observation=45`, `num_critic_observation=64`, and `num_actions=8`; the dynamics and task run in batches on the selected torch device. The `rsl_rl` actor receives 45 observations and the critic receives 64 privileged observations.

## Start Standard PPO Training

```bash
cd /home/a/IMARLSim
source .venv/bin/activate
python standard_vtol/rl/train_ppo.py \
  --device cuda:0 \
  --num-envs 1024 \
  --steps-per-env 256 \
  --iterations 1000
```

Without a GPU, set `--device cpu` and reduce `--num-envs`, for example to `32`. Training logs and checkpoints created by `rsl_rl` are written to `runs/`; generated files are excluded by `.gitignore` and are not part of the source release.

The training entry point instantiates only `rsl_rl.runners.OnPolicyRunner` and `rsl_rl.algorithms.PPO`. RND, symmetry augmentation, recurrent policies, adversarial training, and other RL algorithms are disabled.
