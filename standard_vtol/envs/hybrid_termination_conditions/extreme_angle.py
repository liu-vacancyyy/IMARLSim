import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch


class ExtremeAngle(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_pitch = getattr(config, 'max_pitch', 25)
        self.max_roll = getattr(config, 'max_roll', 30)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        roll, pitch, yaw = env.model.get_posture()
        roll = roll * 180 / torch.pi
        pitch = pitch * 180 / torch.pi
        yaw = yaw * 180 / torch.pi
        bad_done = (torch.abs(roll) > self.max_roll) | (torch.abs(pitch) > self.max_pitch)
        if info is None:
            info = {}
        info['extreme_angle'] = bad_done
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if getattr(self.config, 'termination_verbose', True) and torch.any(bad_done):
            self.log(f'extreme angle!')
            print(torch.sum(bad_done), 'extreme angle!')
        return bad_done, done, exceed_time_limit, info
