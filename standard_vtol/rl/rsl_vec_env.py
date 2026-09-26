"""rsl_rl VecEnv adapter for the public standard_vtol simulator."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "envs"))

from rsl_rl.env import VecEnv
from envs.base_env import ControlEnv


class StandardVTOLVecEnv(VecEnv):
    """Expose the GPU-batched simulator through rsl_rl's current API."""

    def __init__(self, num_envs=1024, config="standard_vtol_fixed_target", device="cuda:0", seed=1):
        self.env = ControlEnv(
            num_envs=num_envs,
            config=config,
            model="GAZEBO",
            random_seed=seed,
            device=device,
        )
        self.num_envs = self.env.n
        self.num_actions = self.env.num_actions
        self.device = self.env.device
        self.cfg = self.env.config
        self.max_episode_length = int(self.env.task.max_steps)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reset()

    def _observations(self):
        actor = self.env.obs().to(self.device)
        critic = self.env.critic_obs().to(self.device)
        return TensorDict(
            {"actor": actor, "critic": critic},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self):
        return self._observations()

    def reset(self):
        self.episode_length_buf.zero_()
        self.env.is_done.fill_(True)
        self.env.bad_done.fill_(True)
        self.env.exceed_time_limit.fill_(True)
        self.env.reset()
        return self._observations()

    def step(self, actions: torch.Tensor):
        actions = actions.reshape(self.num_envs, self.num_actions).to(self.device)
        _, rewards, done, bad_done, timeout, info = self.env.step(actions)
        terminal = done | bad_done | timeout
        self.episode_length_buf += 1

        # rsl_rl expects the observation following a terminal transition. The
        # base simulator keeps reset slots pending until the next call, so
        # perform that reset here and return fresh observations immediately.
        if torch.any(terminal):
            reset_obs = self.env.reset()
            current_obs = self._observations()
            current_obs = TensorDict(
                {
                    key: torch.where(terminal.unsqueeze(-1), reset_obs[key], current_obs[key])
                    for key in reset_obs.keys()
                },
                batch_size=[self.num_envs],
                device=self.device,
            )
            self.episode_length_buf[terminal] = 0
        else:
            current_obs = self._observations()

        extras = {"time_outs": timeout}
        if isinstance(info, dict):
            extras["log"] = {
                f"/standard_vtol/{key}": value
                for key, value in info.items()
                if torch.is_tensor(value) and value.numel() > 0
            }
            extras["log"] = {
                key: value.to(dtype=torch.float32)
                for key, value in extras["log"].items()
            }
        return current_obs, rewards, terminal, extras

    def close(self):
        self.env.close()
