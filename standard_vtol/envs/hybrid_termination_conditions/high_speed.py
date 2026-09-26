import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch

class HighSpeed(BaseTerminationCondition):
    """
    HighSpeed
    End up the simulation if speed are too high.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_velocity = float(getattr(config, 'max_velocity', 10))
        self.transition_max_velocity = float(getattr(
            config, 'mission_transition_max_velocity', self.max_velocity
        ))
        self.backtransition_max_velocity = float(getattr(
            config, 'mission_backtransition_max_velocity', self.max_velocity
        ))


    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if speed are too high.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        velocity = env.model.get_TAS().abs()
        limit = torch.full_like(velocity, self.max_velocity)
        if hasattr(task, 'phase'):
            transition = task.phase == getattr(task, 'TRANSITION', -1)
            backtransition = task.phase == getattr(task, 'BACK_TRANSITION', -1)
            limit = torch.where(
                transition,
                torch.full_like(limit, self.transition_max_velocity),
                limit,
            )
            limit = torch.where(
                backtransition,
                torch.full_like(limit, self.backtransition_max_velocity),
                limit,
            )
        bad_done = velocity >= limit
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if info is None:
            info = {}
        info['high_speed'] = bad_done
        info['high_speed_limit'] = limit
        if getattr(self.config, 'termination_verbose', True) and torch.any(bad_done):
            self.log(f'speed is too high!')
            print(torch.sum(bad_done), 'speed is too high!')
        return bad_done, done, exceed_time_limit, info
