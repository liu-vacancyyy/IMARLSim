import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition


class GroundCollision(BaseTerminationCondition):
    """Allow normal ground contact and terminate only damaging contacts."""

    def __init__(self, config):
        super().__init__(config)
        self.max_touchdown_speed = float(
            getattr(config, 'ground_crash_max_touchdown_speed', 2.5)
        )
        self.max_penetration = float(
            getattr(config, 'ground_crash_max_penetration', 0.05)
        )
        self.max_roll = math.radians(float(
            getattr(config, 'ground_crash_max_roll_deg', 45.0)
        ))
        self.max_pitch = math.radians(float(
            getattr(config, 'ground_crash_max_pitch_deg', 45.0)
        ))
        self.force_factor = float(
            getattr(config, 'ground_crash_force_factor', 20.0)
        )

    def get_termination(self, task, env, info=None):
        if info is None:
            info = {}
        if not hasattr(env.model, 'get_ground_contact_state'):
            bad_done = torch.zeros(env.n, dtype=torch.bool, device=env.device)
            return bad_done, bad_done.clone(), bad_done.clone(), info

        contact = env.model.get_ground_contact_state()
        roll, pitch, _ = env.model.get_posture()
        hard_touchdown = (
            contact['just_touchdown']
            & (contact['touchdown_vertical_speed'] > self.max_touchdown_speed)
        )
        excessive_penetration = contact['max_penetration'] > self.max_penetration
        tipover = contact['on_ground'] & (
            (roll.abs() > self.max_roll) | (pitch.abs() > self.max_pitch)
        )

        if hasattr(env.model, 'mass_curr'):
            force_limit = (
                env.model.mass_curr
                * env.model.dynamics.g
                * self.force_factor
            )
            excessive_force = contact['normal_force'] > force_limit
        else:
            excessive_force = torch.zeros_like(contact['on_ground'])

        bad_done = hard_touchdown | excessive_penetration | tipover | excessive_force
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)

        info['ground_contact'] = contact
        info['ground_crash_hard_touchdown'] = hard_touchdown
        info['ground_crash_penetration'] = excessive_penetration
        info['ground_crash_tipover'] = tipover
        info['ground_crash_force'] = excessive_force
        if getattr(self.config, 'termination_verbose', True) and torch.any(bad_done):
            self.log('damaging ground contact')
        return bad_done, done, exceed_time_limit, info
