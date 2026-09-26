import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch


class ExtremeOmega(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_omega = float(getattr(config, 'max_omega', 2))
        self.max_omega_norm = getattr(config, 'max_omega_norm', None)
        if self.max_omega_norm is not None:
            self.max_omega_norm = float(self.max_omega_norm)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        if self.max_omega_norm is None:
            omega1, omega2, omega3 = env.model.get_euler_angular_velocity()
            bad_done = (
                (torch.abs(omega1) > self.max_omega)
                | (torch.abs(omega2) > self.max_omega)
                | (torch.abs(omega3) > self.max_omega)
            )
            message = 'extreme omega!'
        else:
            p, q, r = env.model.get_angular_velocity()
            omega_norm = torch.sqrt(p * p + q * q + r * r)
            bad_done = omega_norm > self.max_omega_norm
            message = 'angular velocity norm is too high!'
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if info is None:
            info = {}
        info['extreme_omega'] = bad_done
        if getattr(self.config, 'termination_verbose', True) and torch.any(bad_done):
            self.log(message)
            print(torch.sum(bad_done), message)
        return bad_done, done, exceed_time_limit, info
