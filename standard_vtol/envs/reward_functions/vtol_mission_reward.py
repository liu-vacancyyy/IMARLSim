import os
import sys
import math

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class VTOLMissionReward(BaseRewardFunction):
    """Dense progress, mode-efficiency, and safety reward for a VTOL mission."""

    def __init__(self, config):
        super().__init__(config)
        self.w_progress = float(getattr(config, 'mission_w_progress', 4.0))
        self.w_altitude = float(getattr(config, 'mission_w_altitude', 1.0))
        self.w_speed = float(getattr(config, 'mission_w_speed', 0.8))
        self.w_heading = float(getattr(config, 'mission_w_heading', 0.8))
        self.w_waypoint = float(getattr(config, 'mission_w_waypoint', 0.5))
        self.w_phase = float(getattr(config, 'mission_w_phase_bonus', 30.0))
        self.w_time = float(getattr(config, 'mission_w_time', 0.03))
        self.w_attitude = float(getattr(config, 'mission_w_attitude_safety', 0.8))
        self.w_rate = float(getattr(config, 'mission_w_rate_safety', 0.25))
        self.w_aero = float(getattr(config, 'mission_w_aero_safety', 0.5))
        self.w_sink = float(getattr(config, 'mission_w_sink_safety', 1.5))
        self.w_smooth = float(getattr(config, 'mission_w_action_smooth', 0.15))
        self.w_mode = float(getattr(config, 'mission_w_mode_efficiency', 0.2))
        self.w_backtransition_braking = float(getattr(
            config, 'mission_w_backtransition_braking', 0.75
        ))
        self.w_backtransition_overspeed = float(getattr(
            config, 'mission_w_backtransition_overspeed', 0.40
        ))
        self.w_backtransition_receding = float(getattr(
            config, 'mission_w_backtransition_receding', 0.40
        ))
        self.w_constraint = float(
            getattr(config, 'mission_w_constraint_safety', 5.0)
        )
        self.w_ground_wait = float(
            getattr(config, 'mission_w_ground_wait', 1.0)
        )
        self.w_hover_stable = float(
            getattr(config, 'mission_w_hover_stable', 0.0)
        )
        self.constraint_warning = float(
            getattr(config, 'mission_constraint_warning_fraction', 0.8)
        )
        self.constraint_warning = min(max(self.constraint_warning, 0.0), 1.0)

        self.altitude_sigma = max(float(
            getattr(config, 'mission_reward_altitude_sigma', 4.0)
        ), 1e-6)
        self.speed_sigma = max(float(
            getattr(config, 'mission_reward_speed_sigma', 3.0)
        ), 1e-6)
        self.backtransition_speed_sigma = max(float(
            getattr(config, 'mission_reward_backtransition_speed_sigma', 2.0)
        ), 1e-6)
        self.backtransition_potential_clip = max(float(
            getattr(config, 'mission_reward_backtransition_potential_clip', 4.0)
        ), 1.0)
        self.reward_shaping_gamma = min(max(float(getattr(
            config, 'mission_reward_shaping_gamma', 1.0
        )), 0.0), 1.0)
        self.vertical_speed_sigma = max(float(
            getattr(config, 'mission_reward_vertical_speed_sigma', 0.75)
        ), 1e-6)
        self.heading_sigma = max(float(
            getattr(config, 'mission_reward_heading_sigma', 0.35)
        ), 1e-6)
        self.waypoint_sigma = max(float(
            getattr(config, 'mission_reward_waypoint_sigma', 12.0)
        ), 1e-6)
        self.safe_tilt = math.radians(float(
            getattr(config, 'mission_reward_safe_tilt_deg', 25.0)
        ))
        self.safe_rate = float(
            getattr(config, 'mission_reward_safe_rate', 1.5)
        )
        self.safe_alpha = math.radians(float(
            getattr(config, 'mission_reward_safe_alpha_deg', 18.0)
        ))
        self.safe_beta = math.radians(float(
            getattr(config, 'mission_reward_safe_beta_deg', 12.0)
        ))
        self.safe_sink = float(
            getattr(config, 'mission_reward_safe_sink_speed', 1.0)
        )
        self.progress_clip = float(
            getattr(config, 'mission_reward_progress_clip', 2.0)
        )
        self.ground_wait_grace_steps = max(0, int(getattr(
            config, 'mission_ground_wait_grace_steps', 10
        )))
        self.max_velocity = max(float(
            getattr(config, 'max_velocity', 25.0)
        ), 1e-6)
        self.acceleration_limit = max(float(
            getattr(config, 'acceleration_limit', 35.0)
        ), 1e-6)
        self.max_cross_track = max(float(
            getattr(config, 'mission_max_cross_track', 50.0)
        ), 1e-6)
        self.max_altitude = max(float(
            getattr(config, 'mission_max_altitude', 50.0)
        ), 1e-6)
        self.min_cruise_altitude = max(float(
            getattr(config, 'mission_min_safe_cruise_altitude', 4.0)
        ), 1e-6)
        self.max_touchdown_speed = max(float(
            getattr(config, 'ground_crash_max_touchdown_speed', 1.5)
        ), 1e-6)
        self.max_penetration = max(float(
            getattr(config, 'ground_crash_max_penetration', 0.05)
        ), 1e-6)
        self.ground_force_factor = max(float(
            getattr(config, 'ground_crash_force_factor', 20.0)
        ), 1e-6)

    @staticmethod
    def _excess_square(value, limit):
        return torch.relu(value.abs() - limit) ** 2

    def get_reward(self, task, env):
        npos, epos, altitude = env.model.get_position()
        roll, pitch, heading = env.model.get_posture()
        speed = env.model.get_TAS()
        climb_rate = env.model.get_climb_rate()
        p, q, r = env.model.get_angular_velocity()
        alpha = env.model.get_AOA()
        beta = env.model.get_AOS()

        waypoint_distance = torch.sqrt(
            (task.target_npos - npos) ** 2
            + (task.target_epos - epos) ** 2
            + (task.target_altitude - altitude) ** 2
        )
        raw_progress = task.previous_waypoint_distance - waypoint_distance
        progress = torch.where(
            task.phase_advanced,
            torch.zeros_like(raw_progress),
            raw_progress.clamp(-self.progress_clip, self.progress_clip),
        )

        altitude_error = task.target_altitude - altitude
        speed_error = task.target_speed - speed
        heading_error = wrap_PI(task.target_heading - heading)
        # Tracking terms are zero-centered: being exactly on target is neutral,
        # rather than a positive reward that can be farmed by waiting.
        altitude_reward = torch.exp(
            -(altitude_error / self.altitude_sigma) ** 2
        ) - 1.0
        speed_reward = torch.exp(-(speed_error / self.speed_sigma) ** 2) - 1.0
        vertical_climb = (
            (task.phase == task.TAKEOFF)
            | (task.phase == task.ROTOR_CLIMB)
        )
        climb_error = task.rotor_climb_speed - climb_rate
        climb_reward = torch.exp(
            -(climb_error / self.vertical_speed_sigma) ** 2
        ) - 1.0
        speed_reward = torch.where(vertical_climb, climb_reward, speed_reward)
        heading_reward = torch.exp(
            -(heading_error / self.heading_sigma) ** 2
        ) - 1.0
        waypoint_reward = torch.exp(
            -(waypoint_distance / self.waypoint_sigma) ** 2
        ) - 1.0

        # In reverse transition, total airspeed is not the capture variable:
        # vertical motion can be large while the horizontal ground speed is
        # already safe. Track a distance-dependent horizontal-speed profile
        # and use its potential difference as shaping, so holding position at
        # the target speed yields zero reward instead of a loitering bonus.
        vel_n, vel_e, _ = env.model.get_world_velocity()
        horizontal_speed = torch.sqrt(
            (vel_n * vel_n + vel_e * vel_e).clamp_min(0.0)
        )
        distance_to_landing = torch.sqrt(
            (npos - task.goal_n) ** 2 + (epos - task.goal_e) ** 2
        )
        distance_fraction = (
            (distance_to_landing - task.descent_capture_radius)
            / (task.approach_distance - task.descent_capture_radius)
        ).clamp(0.0, 1.0)
        capture_target_speed = min(
            max(task.approach_speed, 0.0), task.backtransition_speed
        )
        desired_backtransition_speed = (
            capture_target_speed
            + distance_fraction * (task.cruise_speed - capture_target_speed)
        )
        backtransition = task.phase == task.BACK_TRANSITION
        backtransition_speed_error = (
            horizontal_speed - desired_backtransition_speed
        ) / self.backtransition_speed_sigma
        backtransition_speed_reward = torch.exp(
            -backtransition_speed_error ** 2
        ) - 1.0
        speed_reward = torch.where(
            backtransition,
            backtransition_speed_reward,
            speed_reward,
        )

        # Compute the previous-step potential from the simulator state saved by
        # GazeboModel.  This is a potential-based term: it pays for reducing
        # speed-profile error once and pays nothing when the aircraft stalls
        # at the same state.
        previous_kinematics = env.model.dynamics.kinematic_derivatives(
            env.model.recent_s
        )
        previous_horizontal_speed = torch.sqrt(
            (previous_kinematics[:, 0] ** 2
             + previous_kinematics[:, 1] ** 2).clamp_min(0.0)
        )
        previous_distance = torch.sqrt(
            (env.model.recent_s[:, 0] - task.goal_n) ** 2
            + (env.model.recent_s[:, 1] - task.goal_e) ** 2
        )
        previous_fraction = (
            (previous_distance - task.descent_capture_radius)
            / (task.approach_distance - task.descent_capture_radius)
        ).clamp(0.0, 1.0)
        previous_desired_speed = (
            capture_target_speed
            + previous_fraction
            * (task.cruise_speed - capture_target_speed)
        )
        current_potential = -torch.clamp(
            backtransition_speed_error ** 2,
            max=self.backtransition_potential_clip,
        )
        previous_error = (
            previous_horizontal_speed - previous_desired_speed
        ) / self.backtransition_speed_sigma
        previous_potential = -torch.clamp(
            previous_error ** 2,
            max=self.backtransition_potential_clip,
        )
        braking_shaping = self.reward_shaping_gamma * current_potential - previous_potential
        braking_shaping = torch.where(
            backtransition,
            braking_shaping,
            torch.zeros_like(braking_shaping),
        )

        landing_dn = task.goal_n - npos
        landing_de = task.goal_e - epos
        closing_speed = (
            vel_n * landing_dn + vel_e * landing_de
        ) / distance_to_landing.clamp_min(1e-6)
        receding_limit = max(float(getattr(
            task, 'backtransition_max_receding_speed', 1.0
        )), 0.0)
        overspeed_error = torch.relu(
            horizontal_speed - desired_backtransition_speed
        ) / self.backtransition_speed_sigma
        receding_error = torch.relu(
            -closing_speed - receding_limit
        ) / self.backtransition_speed_sigma
        overspeed_cost = torch.clamp(
            overspeed_error, max=3.0
        ) ** 2
        receding_cost = torch.clamp(
            receding_error, max=3.0
        ) ** 2
        overspeed_cost = torch.where(
            backtransition, overspeed_cost, torch.zeros_like(overspeed_cost)
        )
        receding_cost = torch.where(
            backtransition, receding_cost, torch.zeros_like(receding_cost)
        )

        tilt_cost = (
            self._excess_square(roll, self.safe_tilt)
            + self._excess_square(pitch, self.safe_tilt)
        )
        rate_cost = (
            self._excess_square(p, self.safe_rate)
            + self._excess_square(q, self.safe_rate)
            + self._excess_square(r, self.safe_rate)
        )
        raw_aero_cost = (
            self._excess_square(alpha, self.safe_alpha)
            + self._excess_square(beta, self.safe_beta)
        )
        aero_cost = torch.where(
            task.aero_envelope_active(env),
            raw_aero_cost,
            torch.zeros_like(raw_aero_cost),
        )
        final_approach = task.phase == task.VERTICAL_LANDING
        sink_cost = torch.where(
            final_approach,
            torch.relu(-climb_rate - self.safe_sink) ** 2,
            torch.zeros_like(climb_rate),
        )

        motor_scale = env.model.motor_omega_max.reshape(1, 5)
        motor = env.model.u[:, 0:5] / motor_scale
        recent_motor = env.model.recent_u[:, 0:5] / motor_scale
        surface = env.model.u[:, 5:8] / env.model.surface_limit.reshape(1, 3)
        recent_surface = (
            env.model.recent_u[:, 5:8]
            / env.model.surface_limit.reshape(1, 3)
        )
        smooth_cost = torch.sum((motor - recent_motor) ** 2, dim=1)
        smooth_cost += torch.sum((surface - recent_surface) ** 2, dim=1)

        vertical_mode = (
            (task.phase == task.TAKEOFF)
            | (task.phase == task.ROTOR_CLIMB)
            | (task.phase == task.VERTICAL_LANDING)
        )
        fixed_mode = task.phase == task.FIXED_WING
        mode_cost = torch.where(
            vertical_mode,
            motor[:, 4] ** 2,
            torch.zeros_like(speed),
        )
        mode_cost += torch.where(
            fixed_mode,
            torch.mean(motor[:, 0:4] ** 2, dim=1),
            torch.zeros_like(speed),
        )

        # Convert the former constrained-PPO safety envelope into a smooth
        # reward barrier. Terminal violations still receive the event penalty.
        warning = self.constraint_warning
        speed_constraint = torch.relu(speed / self.max_velocity - warning) ** 2
        body_acceleration = (
            env.model.s[:, 6:9] - env.model.recent_s[:, 6:9]
        ) / max(float(getattr(env.model, 'dt', 0.02)), 1e-6)
        acceleration = torch.linalg.vector_norm(body_acceleration, dim=1)
        acceleration_constraint = torch.relu(
            acceleration / self.acceleration_limit - warning
        ) ** 2
        cross_track = (
            -(npos - task.start_n) * task.route_unit_e_batch
            + (epos - task.start_e) * task.route_unit_n_batch
        )
        route_constraint = torch.relu(
            cross_track.abs() / self.max_cross_track - warning
        ) ** 2
        altitude_constraint = torch.relu(
            altitude / self.max_altitude - warning
        ) ** 2
        cruise_phase = (
            (task.phase == task.TRANSITION)
            | (task.phase == task.FIXED_WING)
            | (task.phase == task.BACK_TRANSITION)
        )
        low_altitude_constraint = torch.where(
            cruise_phase,
            torch.relu(
                (self.min_cruise_altitude - altitude)
                / self.min_cruise_altitude
            ) ** 2,
            torch.zeros_like(altitude),
        )

        contact = env.model.get_ground_contact_state()
        ground_wait_cost = (
            (task.phase == task.TAKEOFF)
            & contact['on_ground']
            & (env.step_count > self.ground_wait_grace_steps)
        ).to(speed.dtype)
        weight = (env.model.mass_curr * env.model.dynamics.g).clamp_min(1e-6)
        touchdown_constraint = torch.where(
            contact['just_touchdown'],
            torch.relu(
                contact['touchdown_vertical_speed'] / self.max_touchdown_speed
                - warning
            ) ** 2,
            torch.zeros_like(speed),
        )
        penetration_constraint = torch.relu(
            contact['max_penetration'] / self.max_penetration - warning
        ) ** 2
        force_constraint = torch.relu(
            contact['normal_force'] / (weight * self.ground_force_factor)
            - warning
        ) ** 2
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
        ).to(speed.dtype)
        constraint_cost = (
            speed_constraint
            + acceleration_constraint
            + route_constraint
            + altitude_constraint
            + low_altitude_constraint
            + touchdown_constraint
            + penetration_constraint
            + force_constraint
            + premature_contact
        )
        task.constraint_penalty = constraint_cost.detach()
        hover_stable_reward = torch.zeros_like(speed)
        if getattr(task, 'terminal_mode', 'landing') == 'hover':
            hover_stable_reward = task.hover_stable_candidate(env).to(speed.dtype)

        return (
            self.w_progress * progress
            + self.w_altitude * altitude_reward
            + self.w_speed * speed_reward
            + self.w_heading * heading_reward
            + self.w_waypoint * waypoint_reward
            + self.w_phase * task.phase_advanced.to(speed.dtype)
            - self.w_time
            - self.w_attitude * tilt_cost
            - self.w_rate * rate_cost
            - self.w_aero * aero_cost
            - self.w_sink * sink_cost
            - self.w_smooth * smooth_cost
            - self.w_mode * mode_cost
            + self.w_backtransition_braking * braking_shaping
            - self.w_backtransition_overspeed * overspeed_cost
            - self.w_backtransition_receding * receding_cost
            - self.w_constraint * constraint_cost
            - self.w_ground_wait * ground_wait_cost
            + self.w_hover_stable * hover_stable_reward
        )


class VTOLMissionEventReward(BaseRewardFunction):
    """Terminal rewards that distinguish safe landing, crash, and timeout."""

    def __init__(self, config):
        super().__init__(config)
        self.success_bonus = float(
            getattr(config, 'mission_success_bonus', 500.0)
        )
        self.failure_penalty = float(
            getattr(config, 'mission_failure_penalty', 250.0)
        )
        self.timeout_penalty = float(
            getattr(config, 'mission_timeout_penalty', 100.0)
        )

    def get_reward(self, task, env):
        failure_mask = env.bad_done.bool()
        success_mask = task.mission_success(env) & ~failure_mask
        timeout_mask = (
            env.exceed_time_limit.bool() & ~success_mask & ~failure_mask
        )
        success = success_mask.to(env.model.s.dtype)
        failure = failure_mask.to(env.model.s.dtype)
        timeout = timeout_mask.to(env.model.s.dtype)
        return (
            self.success_bonus * success
            - self.failure_penalty * failure
            - self.timeout_penalty * timeout
        )
