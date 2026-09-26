"""Minimal vectorized environment used by the public standard_vtol release.

The simulator is batched on a single torch device.  This class intentionally
contains no wind, sensor, mass, inertia, curriculum, or target randomization;
those options are disabled in the public configuration and remain out of the
release surface.
"""

import random

import gym
import numpy as np
import torch

from utils.utils import parse_config
from models.model_base import BaseModel
from tasks.task_base import BaseTask


class BaseEnv(gym.Env):
    def __init__(self, num_envs, config, model="GAZEBO", random_seed=1, device="cpu"):
        super().__init__()
        self.config = parse_config(config)
        self.num_envs = int(num_envs)
        self.num_agents = int(getattr(self.config, "num_agents", 1))
        if self.num_agents != 1:
            raise ValueError("standard_vtol exposes one agent per environment")
        self.n = self.num_envs
        self.device = torch.device(device)
        self.load(random_seed, config, model)
        self.step_count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.is_done = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.bad_done = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.exceed_time_limit = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._action_preprocessed = False

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    def load(self, random_seed, config, model):
        raise NotImplementedError

    @property
    def observation_space(self):
        return self.task.observation_space

    @property
    def critic_observation_space(self):
        return self.task.critic_observation_space

    @property
    def action_space(self):
        return self.task.action_space

    @property
    def num_observation(self):
        return self.task.num_observation

    @property
    def num_critic_observation(self):
        return self.task.num_critic_observation

    @property
    def num_actions(self):
        return self.task.num_actions

    def obs(self):
        return self.task.get_obs(self)

    def critic_obs(self):
        return self.task.get_critic_obs(self)

    def info(self):
        return {}

    def reset(self, update_observation=True):
        reset = self.is_done | self.bad_done | self.exceed_time_limit
        self.model.reset(self)
        self.task.reset(self)
        self.step_count[reset] = 0
        self.is_done.zero_()
        self.bad_done.zero_()
        self.exceed_time_limit.zero_()
        self._action_preprocessed = False
        if update_observation and hasattr(self.task, "update_before_observation"):
            self.task.update_before_observation(self)
        return self.obs()

    def step(self, action):
        # The model/task own reset semantics; a completed slot is reset before
        # its next action is integrated, matching the original vector backend.
        self.reset(update_observation=False)
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if self._action_preprocessed:
            self._action_preprocessed = False
        elif hasattr(self.task, "maybe_override_action"):
            action = self.task.maybe_override_action(self, action)
        self.model.update(action)
        self.step_count += 1
        if hasattr(self.task, "update_before_observation"):
            self.task.update_before_observation(self)
        obs = self.obs()
        info = self.info()
        done, bad_done, timeout, info = self.task.get_termination(self, info)
        self.is_done |= done
        self.bad_done |= bad_done
        self.exceed_time_limit |= timeout
        reward = self.task.get_reward(self)
        self.task.step(self)
        return obs, reward, done, bad_done, timeout, info

    def close(self):
        return None


class ControlEnv(BaseEnv):
    """The only public environment: standard_vtol fixed-target mission."""

    def load(self, random_seed, config, model):
        from models.gazebo_model import GazeboModel
        from tasks.vtol_mission_task import VTOLMissionTask

        if str(model).upper() not in ("GAZEBO", "GAZEBO_VTOL"):
            raise ValueError("standard_vtol requires model='GAZEBO'")
        self.model = GazeboModel(self.config, self.n, self.device, random_seed)
        if getattr(self.config, "task_name", None) != "vtol_mission":
            raise ValueError("the public config must use task_name='vtol_mission'")
        self.task = VTOLMissionTask(self.config, self.n, self.device, random_seed)
