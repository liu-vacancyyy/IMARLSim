import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from hybrid_termination_conditions.termination_condition_base import (
    BaseTerminationCondition,
)


class VTOLMissionBoundary(BaseTerminationCondition):
    """Terminate unsafe route departures and premature ground contact."""

    def __init__(self, config):
        super().__init__(config)
        self.max_cross_track = float(
            getattr(config, 'mission_max_cross_track', 50.0)
        )
        self.route_margin = float(
            getattr(config, 'mission_route_margin', 30.0)
        )
        self.max_altitude = float(
            getattr(config, 'mission_max_altitude', 50.0)
        )
        self.min_cruise_altitude = float(
            getattr(config, 'mission_min_safe_cruise_altitude', 4.0)
        )

    def get_termination(self, task, env, info=None):
        if info is None:
            info = {}
        npos, epos, altitude = env.model.get_position()
        contact = env.model.get_ground_contact_state()
        along_track = (
            (npos - task.start_n) * task.route_unit_n_batch
            + (epos - task.start_e) * task.route_unit_e_batch
        )
        cross_track = (
            -(npos - task.start_n) * task.route_unit_e_batch
            + (epos - task.start_e) * task.route_unit_n_batch
        )

        # Once reverse transition has brought the aircraft inside the landing
        # capture circle, a small amount of along-track overshoot is expected:
        # the vehicle must bleed forward speed before it can descend vertically.
        # Keep lateral deviation and all pre-capture route bounds strict.
        distance_to_landing = torch.sqrt(
            (npos - task.goal_n) ** 2 + (epos - task.goal_e) ** 2
        )
        landing_phase = (
            (task.phase == getattr(task, 'BACK_TRANSITION', -1))
            | (task.phase == getattr(task, 'VERTICAL_LANDING', -1))
        )
        capture_corridor = landing_phase & (
            distance_to_landing <= task.descent_capture_radius
        )
        along_track_before_start = along_track < -self.route_margin
        along_track_after_end = (
            along_track > task.route_length_batch + self.route_margin
        )
        along_track_overshoot = along_track_after_end & ~capture_corridor

        off_route = (
            (cross_track.abs() > self.max_cross_track)
            | along_track_before_start
            | along_track_overshoot
        )
        altitude_violation = altitude > self.max_altitude
        cruise_phase = (
            (task.phase == task.TRANSITION)
            | (task.phase == task.FIXED_WING)
            | (task.phase == task.BACK_TRANSITION)
        )
        too_low = cruise_phase & (altitude < self.min_cruise_altitude)
        hover_terminal_contact = (
            getattr(task, 'terminal_mode', 'landing') == 'hover'
        ) & (task.phase == task.VERTICAL_LANDING)
        premature_contact = (
            contact['on_ground']
            & (
                (
                    (task.phase >= task.ROTOR_CLIMB)
                    & (task.phase <= task.BACK_TRANSITION)
                )
                | hover_terminal_contact
            )
        )
        finite = torch.isfinite(env.model.s).all(dim=1)
        nonfinite = ~finite

        bad_done = (
            off_route
            | altitude_violation
            | too_low
            | premature_contact
            | nonfinite
        )
        done = torch.zeros_like(bad_done)
        timeout = torch.zeros_like(bad_done)
        info['mission_off_route'] = off_route
        info['mission_capture_corridor'] = capture_corridor
        info['mission_along_track'] = along_track
        info['mission_cross_track'] = cross_track
        info['mission_landing_distance'] = distance_to_landing
        vel_n, vel_e, vel_up = env.model.get_world_velocity()
        horizontal_speed = torch.sqrt(
            (vel_n * vel_n + vel_e * vel_e).clamp_min(0.0)
        )
        landing_dn = task.goal_n - npos
        landing_de = task.goal_e - epos
        closing_speed = (
            vel_n * landing_dn + vel_e * landing_de
        ) / distance_to_landing.clamp_min(1.0e-6)
        roll, pitch, _ = env.model.get_posture()
        p, q, _ = env.model.get_angular_velocity()
        info['mission_altitude'] = altitude
        info['mission_horizontal_speed'] = horizontal_speed
        info['mission_closing_speed'] = closing_speed
        info['mission_vertical_speed'] = vel_up
        info['mission_roll'] = roll
        info['mission_pitch'] = pitch
        info['mission_roll_rate'] = p
        info['mission_pitch_rate'] = q
        info['mission_along_track_before_start'] = along_track_before_start
        info['mission_along_track_overshoot'] = along_track_overshoot
        info['mission_cross_track_violation'] = (
            cross_track.abs() > self.max_cross_track
        )
        info['mission_altitude_violation'] = altitude_violation | too_low
        info['mission_premature_contact'] = premature_contact
        info['mission_nonfinite'] = nonfinite
        if getattr(self.config, 'termination_verbose', True) and torch.any(bad_done):
            self.log('VTOL mission safety boundary violated')
        return bad_done, done, timeout, info


class VTOLMissionSuccess(BaseTerminationCondition):
    """Complete only after a settled, accurate, low-energy touchdown."""

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, info=None):
        if info is None:
            info = {}
        success = task.mission_success(env)
        bad_done = torch.zeros_like(success)
        timeout = torch.zeros_like(success)
        info['mission_success'] = success
        info['mission_phase'] = task.phase
        info['mission_phase_advanced'] = task.phase_advanced
        info['mission_start_phase'] = task.start_phase
        npos, epos, _ = env.model.get_position()
        info['mission_landing_error'] = torch.sqrt(
            (npos - task.goal_n) ** 2 + (epos - task.goal_e) ** 2
        )
        info['mission_target_n'] = task.goal_n.clone()
        info['mission_target_e'] = task.goal_e.clone()
        info['mission_elapsed_steps'] = env.step_count.clone()
        info['mission_touchdown_speed'] = (
            env.model.last_touchdown_vertical_speed.clone()
        )
        if getattr(task, 'terminal_mode', 'landing') == 'hover':
            info['mission_hover_stable_steps'] = task.hover_stable_count.clone()
        if getattr(self.config, 'termination_verbose', True) and torch.any(success):
            self.log('VTOL mission completed with a safe landing')
        return bad_done, success, timeout, info


class VTOLMissionTimeout(BaseTerminationCondition):
    """Mark the maximum mission duration as a truncation, not a crash."""

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = int(getattr(config, 'max_steps', 3000))

    def get_termination(self, task, env, info=None):
        if info is None:
            info = {}
        timeout = env.step_count >= self.max_steps
        zeros = torch.zeros_like(timeout)
        info['mission_timeout'] = timeout
        return zeros, zeros.clone(), timeout, info
