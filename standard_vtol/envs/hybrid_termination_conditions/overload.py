import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from termination_condition_base import BaseTerminationCondition


class Overload(BaseTerminationCondition):
    """
    Overload
    End up the simulation if acceleration are too high.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.acceleration_limit = float(getattr(config, 'acceleration_limit', 5.0))
        self.persist_steps = max(1, int(getattr(
            config, 'overload_bad_done_persist_steps', 1
        )))
        self.violation_count = None

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if acceleration are too high.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        violation = self._judge_overload(env)
        self._ensure_state(env)
        episode_start = env.step_count <= 1
        self.violation_count = torch.where(
            episode_start,
            torch.zeros_like(self.violation_count),
            self.violation_count,
        )
        self.violation_count = torch.where(
            violation,
            self.violation_count + 1,
            torch.zeros_like(self.violation_count),
        )
        bad_done = self.violation_count >= self.persist_steps
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if info is None:
            info = {}
        info['overload'] = bad_done
        if getattr(self.config, 'termination_verbose', True) and torch.any(bad_done):
            self.log(f'acceleration is too high!')
            print(torch.sum(bad_done), 'acceleration is too high!')
        return bad_done, done, exceed_time_limit, info

    def _ensure_state(self, env):
        if (
            self.violation_count is None
            or self.violation_count.numel() != env.n
            or self.violation_count.device != env.device
        ):
            self.violation_count = torch.zeros(
                env.n, dtype=torch.long, device=env.device
            )

    def _judge_overload(self, env):
        ax, ay, az = env.model.get_acceleration()
        acceleration = ax ** 2 + ay ** 2 + az ** 2
        acceleration = torch.sqrt(acceleration)
        limit = self.acceleration_limit
        # Reverse transition deliberately blends rotor and wing forces and can
        # create a short high-g transient while the aircraft reconfigures. Use
        # a bounded, phase-specific envelope rather than terminating a valid
        # transition at the nominal cruise limit.
        if hasattr(env, 'task'):
            backtransition = env.task.phase == getattr(env.task, 'BACK_TRANSITION', -1)
            transition = env.task.phase == getattr(env.task, 'TRANSITION', -1)
            back_limit = float(getattr(
                self.config, 'mission_backtransition_acceleration_limit', limit
            ))
            transition_limit = float(getattr(
                self.config, 'mission_transition_acceleration_limit', limit
            ))
            limit_tensor = torch.full_like(acceleration, limit)
            limit_tensor = torch.where(
                transition, torch.full_like(acceleration, transition_limit), limit_tensor
            )
            limit_tensor = torch.where(
                backtransition, torch.full_like(acceleration, back_limit), limit_tensor
            )
        else:
            limit_tensor = torch.full_like(acceleration, limit)
        flag_overload = acceleration > limit_tensor
        return flag_overload
