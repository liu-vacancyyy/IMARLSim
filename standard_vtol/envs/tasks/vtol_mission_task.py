import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.extreme_state import ExtremeState
from hybrid_termination_conditions.ground_collision import GroundCollision
from hybrid_termination_conditions.high_speed import HighSpeed
from hybrid_termination_conditions.overload import Overload
from reward_functions.vtol_mission_reward import (
    VTOLMissionEventReward,
    VTOLMissionReward,
)
from termination_conditions.vtol_mission_termination import (
    VTOLMissionBoundary,
    VTOLMissionSuccess,
    VTOLMissionTimeout,
)
from utils.utils import wrap_PI


class VTOLMissionTask(BaseTask):
    """GPU-vectorized standard_vtol takeoff, cruise, and landing mission."""

    TAKEOFF = 0
    ROTOR_CLIMB = 1
    TRANSITION = 2
    FIXED_WING = 3
    BACK_TRANSITION = 4
    VERTICAL_LANDING = 5
    VERTICAL_HOVER = 5
    PHASE_COUNT = 6
    OBSERVATION_SIZE = 45
    CRITIC_OBSERVATION_SIZE = 64
    PHASE_NAMES = (
        'takeoff',
        'rotor_climb',
        'transition',
        'fixed_wing',
        'back_transition',
        'vertical_landing',
    )

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'vtol_mission'
        self.dt = float(getattr(config, 'dt', 0.02))
        self.max_steps = int(getattr(config, 'max_steps', 3000))

        self.start_n = float(getattr(config, 'mission_start_n', 0.0))
        self.start_e = float(getattr(config, 'mission_start_e', 0.0))
        self.landing_n = float(getattr(config, 'mission_landing_n', 200.0))
        self.landing_e = float(getattr(config, 'mission_landing_e', 0.0))
        self.randomize_target = bool(getattr(
            config, 'mission_randomize_target', False
        ))
        self.landing_altitude = float(
            getattr(config, 'mission_landing_altitude', 0.095)
        )
        self.terminal_mode = str(getattr(
            config, 'mission_terminal_mode', 'landing'
        )).lower()
        if self.terminal_mode not in ('landing', 'hover'):
            raise ValueError("mission_terminal_mode must be 'landing' or 'hover'")
        if self.terminal_mode == 'hover':
            self.PHASE_NAMES = self.PHASE_NAMES[:-1] + ('vertical_hover',)
        route_n = self.landing_n - self.start_n
        route_e = self.landing_e - self.start_e
        self.route_length = math.hypot(route_n, route_e)
        if self.route_length < 20.0:
            raise ValueError('VTOL mission route must be at least 20 m long')
        self.route_unit_n = route_n / self.route_length
        self.route_unit_e = route_e / self.route_length
        self.route_heading = math.atan2(route_e, route_n)
        self.target_distance_min = float(getattr(
            config, 'mission_target_distance_min', self.route_length
        ))
        self.target_distance_max = float(getattr(
            config, 'mission_target_distance_max', self.route_length
        ))
        self.target_bearing_min = math.radians(float(getattr(
            config, 'mission_target_bearing_min_deg', 0.0
        )))
        self.target_bearing_max = math.radians(float(getattr(
            config, 'mission_target_bearing_max_deg', 0.0
        )))
        if self.randomize_target:
            if not 20.0 <= self.target_distance_min <= self.target_distance_max:
                raise ValueError(
                    'random VTOL target distances must satisfy '
                    '20 <= min <= max'
                )
            if self.target_bearing_max <= self.target_bearing_min:
                raise ValueError(
                    'mission_target_bearing_max_deg must exceed the minimum'
                )

        self.cruise_altitude = float(
            getattr(config, 'mission_cruise_altitude', 25.0)
        )
        self.landing_hover_altitude = float(
            getattr(config, 'mission_landing_hover_altitude', 8.0)
        )
        self.takeoff_clearance = float(
            getattr(config, 'mission_takeoff_clearance', 1.0)
        )
        self.approach_distance = float(
            getattr(config, 'mission_approach_distance', 60.0)
        )
        self.descent_capture_radius = float(
            getattr(config, 'mission_descent_capture_radius', 15.0)
        )
        shortest_route = (
            self.target_distance_min if self.randomize_target else self.route_length
        )
        if not 5.0 < self.approach_distance < shortest_route:
            raise ValueError('mission_approach_distance must be inside the route')
        if not 0.0 < self.descent_capture_radius < self.approach_distance:
            raise ValueError(
                'mission_descent_capture_radius must be positive and smaller '
                'than mission_approach_distance'
            )

        self.transition_speed = float(
            getattr(config, 'mission_transition_complete_speed', 11.0)
        )
        self.backtransition_speed = float(
            getattr(config, 'mission_backtransition_complete_speed', 6.0)
        )
        if self.backtransition_speed <= 0.0:
            raise ValueError('mission_backtransition_complete_speed must be positive')
        self.backtransition_capture_hold_steps = max(1, int(getattr(
            config, 'mission_backtransition_capture_hold_steps', 5
        )))
        self.backtransition_max_receding_speed = max(0.0, float(getattr(
            config, 'mission_backtransition_max_receding_speed', 1.0
        )))
        self.backtransition_max_vertical_speed = max(0.0, float(getattr(
            config,
            'mission_backtransition_max_vertical_speed',
            getattr(config, 'mission_backtransition_max_sink_speed', 1.5),
        )))
        # Backward-compatible alias used by existing diagnostics/tests.
        self.backtransition_max_sink_speed = self.backtransition_max_vertical_speed
        self.descent_capture_altitude = float(getattr(
            config,
            'mission_descent_capture_altitude',
            self.landing_hover_altitude + 2.0,
        ))
        if self.descent_capture_altitude <= self.landing_altitude:
            raise ValueError(
                'mission_descent_capture_altitude must exceed landing altitude'
            )
        self.gate_backtransition_pusher = bool(getattr(
            config, 'mission_gate_backtransition_pusher', True
        ))
        self.backtransition_pusher_max_fraction = min(max(float(getattr(
            config, 'mission_backtransition_pusher_max_fraction', 0.15
        )), 0.0), 1.0)
        self.backtransition_surface_action_scale = min(max(float(getattr(
            config, 'mission_backtransition_surface_action_scale', 0.35
        )), 0.0), 1.0)
        self.backtransition_aero_scale = min(max(float(getattr(
            config, 'mission_backtransition_aero_scale', 0.20)
        ), 0.0), 1.0)
        self.rotor_climb_aero_scale = min(max(float(getattr(
            config, 'mission_rotor_climb_aero_scale', 0.0)
        ), 0.0), 1.0)
        self.transition_aero_scale = min(max(float(getattr(
            config, 'mission_transition_aero_scale', 0.25)
        ), 0.0), 1.0)
        self.descent_capture_max_tilt = math.radians(float(getattr(
            config, 'mission_descent_capture_max_tilt_deg', 25.0
        )))
        # Do not hand an unstable attitude to the vertical controller.  Roll
        # and pitch rates are the axes directly actuated by the lift-rotor
        # differential; yaw rate is intentionally left out of this gate
        # because the standard_vtol yaw response is slower and is damped after
        # capture.
        self.descent_capture_max_roll_rate = max(float(getattr(
            config, 'mission_descent_capture_max_roll_rate', 0.75
        )), 0.0)
        self.descent_capture_max_pitch_rate = max(float(getattr(
            config, 'mission_descent_capture_max_pitch_rate', 0.75
        )), 0.0)
        self.cruise_speed = float(
            getattr(config, 'mission_cruise_speed', 14.0)
        )
        self.min_cruise_altitude = float(getattr(
            config, 'mission_min_safe_cruise_altitude', 4.0
        ))
        self.max_cross_track = max(float(getattr(
            config, 'mission_max_cross_track', 50.0
        )), 1.0)
        self.approach_speed = float(
            getattr(config, 'mission_approach_speed', 5.0)
        )
        self.rotor_climb_speed = float(
            getattr(config, 'mission_rotor_climb_speed', 2.0)
        )
        self.final_heading = float(
            getattr(config, 'mission_landing_heading', self.route_heading)
        )
        self.final_heading_follows_route = bool(getattr(
            config, 'mission_final_heading_follows_route', False
        ))
        self.goal_n = torch.full(
            (self.n,), self.landing_n, dtype=torch.float32, device=self.device
        )
        self.goal_e = torch.full(
            (self.n,), self.landing_e, dtype=torch.float32, device=self.device
        )
        self.route_length_batch = torch.full(
            (self.n,), self.route_length, dtype=torch.float32, device=self.device
        )
        self.route_unit_n_batch = torch.full(
            (self.n,), self.route_unit_n, dtype=torch.float32, device=self.device
        )
        self.route_unit_e_batch = torch.full(
            (self.n,), self.route_unit_e, dtype=torch.float32, device=self.device
        )
        self.route_heading_batch = torch.full(
            (self.n,), self.route_heading, dtype=torch.float32, device=self.device
        )
        self.final_heading_batch = torch.full(
            (self.n,), self.final_heading, dtype=torch.float32, device=self.device
        )

        dwell = getattr(config, 'mission_phase_min_steps', [10, 25, 40, 40, 25, 0])
        if len(dwell) != self.PHASE_COUNT:
            raise ValueError('mission_phase_min_steps must contain six values')
        self.phase_min_steps = torch.tensor(
            [int(value) for value in dwell], dtype=torch.long, device=self.device
        )

        self.curriculum_enabled = bool(
            getattr(config, 'mission_curriculum_enable', True)
        )
        weights = getattr(
            config,
            'mission_curriculum_phase_weights',
            [0.30, 0.14, 0.14, 0.14, 0.14, 0.14],
        )
        if len(weights) != self.PHASE_COUNT or any(float(w) < 0.0 for w in weights):
            raise ValueError('mission_curriculum_phase_weights must be six non-negative values')
        self.curriculum_weights = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        )
        if float(self.curriculum_weights.sum()) <= 0.0:
            raise ValueError('mission_curriculum_phase_weights must have positive sum')
        self.curriculum_weights /= self.curriculum_weights.sum()
        self.curriculum_cross_track_std = float(
            getattr(config, 'mission_curriculum_cross_track_std', 2.0)
        )
        self.curriculum_heading_std = math.radians(float(
            getattr(config, 'mission_curriculum_heading_std_deg', 5.0)
        ))
        self.curriculum_backtransition_cross_track_std = max(float(getattr(
            config,
            'mission_curriculum_backtransition_cross_track_std',
            self.curriculum_cross_track_std,
        )), 0.0)
        self.curriculum_backtransition_heading_std = math.radians(max(float(getattr(
            config,
            'mission_curriculum_backtransition_heading_std_deg',
            math.degrees(self.curriculum_heading_std),
        )), 0.0))
        self.curriculum_backtransition_speed_min = max(float(getattr(
            config,
            'mission_curriculum_backtransition_speed_min',
            self.backtransition_speed,
        )), 0.0)
        self.curriculum_backtransition_speed_max = max(float(getattr(
            config,
            'mission_curriculum_backtransition_speed_max',
            self.cruise_speed,
        )), self.curriculum_backtransition_speed_min)

        self.distance_norm = max(float(
            getattr(config, 'mission_distance_norm', self.route_length)
        ), 1e-6)
        self.altitude_norm = max(float(
            getattr(config, 'mission_altitude_norm', self.cruise_altitude)
        ), 1e-6)
        self.speed_norm = max(float(
            getattr(config, 'mission_speed_norm', 20.0)
        ), 1e-6)
        self.vertical_speed_norm = max(float(
            getattr(config, 'mission_vertical_speed_norm', 8.0)
        ), 1e-6)
        self.rate_norm = max(float(
            getattr(config, 'mission_rate_norm', 3.0)
        ), 1e-6)
        self.gps_age_norm = max(float(
            getattr(config, 'mission_gps_age_norm', 0.2)
        ), 1e-6)
        self.sensor_noise_enabled = bool(
            getattr(config, 'enable_sensor_noise', True)
        )
        self.sensor_velocity_std = float(
            getattr(config, 'sensor_vel_std', 0.05)
        )
        self.sensor_attitude_std = float(
            getattr(config, 'sensor_att_std', 0.003)
        )
        self.sensor_airspeed_std = float(
            getattr(config, 'sensor_airspeed_std', self.sensor_velocity_std)
        )
        self.sensor_rate_std = float(
            getattr(config, 'sensor_omega_std', 0.0003394)
        )
        self.gate_vertical_actuators = bool(
            getattr(config, 'mission_gate_vertical_actuators', True)
        )
        self.gate_fixed_wing_lift_rotors = bool(getattr(
            config, 'mission_gate_fixed_wing_lift_rotors', True
        ))
        self.fixed_wing_lift_action = min(max(float(getattr(
            config, 'mission_fixed_wing_lift_action', -0.25
        )), -1.0), 1.0)
        self.fixed_wing_lift_recovery_action = min(max(float(getattr(
            config, 'mission_fixed_wing_lift_recovery_action', 0.05
        )), -1.0), 1.0)
        self.fixed_wing_lift_recovery_altitude = max(float(getattr(
            config, 'mission_fixed_wing_lift_recovery_altitude', 15.0
        )), self.landing_altitude + 1.0)
        self.fixed_wing_lift_low_speed_action = min(max(float(getattr(
            config, 'mission_fixed_wing_lift_low_speed_action', -0.05
        )), -1.0), 1.0)
        self.fixed_wing_lift_speed_start = max(float(getattr(
            config, 'mission_fixed_wing_lift_speed_start', self.transition_speed
        )), 0.0)
        self.fixed_wing_lift_speed_end = max(float(getattr(
            config, 'mission_fixed_wing_lift_speed_end', self.cruise_speed + 4.0
        )), self.fixed_wing_lift_speed_start + 1e-3)
        # Initialize the cruise ceiling before deriving its low-altitude
        # recovery value.  This keeps the two configuration parameters
        # consistent regardless of object construction order.
        self.fixed_wing_pusher_max_action = min(max(float(getattr(
            config, 'mission_fixed_wing_pusher_max_action', -0.50
        )), -1.0), 1.0)
        self.fixed_wing_pusher_recovery_action = min(max(float(getattr(
            config, 'mission_fixed_wing_pusher_recovery_action', -0.90
        )), -1.0), self.fixed_wing_pusher_max_action)
        self.fixed_wing_pusher_recovery_altitude = max(float(getattr(
            config, 'mission_fixed_wing_pusher_recovery_altitude', 10.0
        )), self.min_cruise_altitude)
        self.fixed_wing_stabilizer_enable = bool(getattr(
            config, 'mission_fixed_wing_stabilizer_enable', True
        ))
        self.fixed_wing_stabilizer_start = math.radians(float(getattr(
            config, 'mission_fixed_wing_stabilizer_start_deg', 15.0
        )))
        self.fixed_wing_stabilizer_full = max(
            math.radians(float(getattr(
                config, 'mission_fixed_wing_stabilizer_full_deg', 35.0
            ))),
            self.fixed_wing_stabilizer_start + 1e-3,
        )
        self.fixed_wing_stabilizer_roll_gain = float(getattr(
            config, 'mission_fixed_wing_stabilizer_roll_gain', 0.90
        ))
        self.fixed_wing_stabilizer_roll_rate_gain = float(getattr(
            config, 'mission_fixed_wing_stabilizer_roll_rate_gain', 0.08
        ))
        self.fixed_wing_stabilizer_pitch_gain = float(getattr(
            config, 'mission_fixed_wing_stabilizer_pitch_gain', 0.70
        ))
        self.fixed_wing_stabilizer_pitch_rate_gain = float(getattr(
            config, 'mission_fixed_wing_stabilizer_pitch_rate_gain', 0.08
        ))
        self.fixed_wing_stabilizer_diff_limit = min(max(float(getattr(
            config, 'mission_fixed_wing_stabilizer_diff_limit', 0.45
        )), 0.0), 1.0)
        self.fixed_wing_stabilizer_common_limit = min(max(float(getattr(
            config, 'mission_fixed_wing_stabilizer_common_limit', 0.35
        )), 0.0), 1.0)
        self.fixed_wing_nav_enable = bool(getattr(
            config, 'mission_fixed_wing_nav_enable', True
        ))
        self.fixed_wing_nav_heading_gain = max(float(getattr(
            config, 'mission_fixed_wing_nav_heading_gain', 1.10
        )), 0.0)
        self.fixed_wing_nav_cross_track_gain = max(float(getattr(
            config, 'mission_fixed_wing_nav_cross_track_gain', 0.055
        )), 0.0)
        self.fixed_wing_nav_lookahead = max(float(getattr(
            config, 'mission_fixed_wing_nav_lookahead', 25.0
        )), 1.0)
        self.fixed_wing_nav_roll_limit = math.radians(float(getattr(
            config, 'mission_fixed_wing_nav_roll_limit_deg', 22.0
        )))
        self.backtransition_nav_enable = bool(getattr(
            config, 'mission_backtransition_nav_enable', True
        ))
        self.backtransition_nav_heading_gain = max(float(getattr(
            config, 'mission_backtransition_nav_heading_gain',
            self.fixed_wing_nav_heading_gain,
        )), 0.0)
        self.backtransition_nav_cross_track_gain = max(float(getattr(
            config, 'mission_backtransition_nav_cross_track_gain',
            self.fixed_wing_nav_cross_track_gain,
        )), 0.0)
        self.backtransition_nav_lookahead = max(float(getattr(
            config, 'mission_backtransition_nav_lookahead', 35.0
        )), 1.0)
        self.backtransition_nav_roll_limit = math.radians(float(getattr(
            config, 'mission_backtransition_nav_roll_limit_deg', 18.0
        )))
        self.backtransition_nav_diff_limit = min(max(float(getattr(
            config, 'mission_backtransition_nav_diff_limit',
            self.backtransition_surface_action_scale,
        )), 0.0), 1.0)
        self.backtransition_stabilizer_enable = bool(getattr(
            config, 'mission_backtransition_stabilizer_enable', True
        ))
        self.backtransition_stabilizer_start = math.radians(float(getattr(
            config, 'mission_backtransition_stabilizer_start_deg', 12.0
        )))
        self.backtransition_stabilizer_full = max(
            math.radians(float(getattr(
                config, 'mission_backtransition_stabilizer_full_deg', 32.0
            ))),
            self.backtransition_stabilizer_start + 1e-3,
        )
        self.backtransition_stabilizer_roll_gain = float(getattr(
            config, 'mission_backtransition_stabilizer_roll_gain', 0.28
        ))
        self.backtransition_stabilizer_roll_rate_gain = float(getattr(
            config, 'mission_backtransition_stabilizer_roll_rate_gain', 0.04
        ))
        self.backtransition_stabilizer_pitch_gain = float(getattr(
            config, 'mission_backtransition_stabilizer_pitch_gain', 0.24
        ))
        self.backtransition_stabilizer_pitch_rate_gain = float(getattr(
            config, 'mission_backtransition_stabilizer_pitch_rate_gain', 0.04
        ))
        self.backtransition_stabilizer_diff_limit = min(max(float(getattr(
            config, 'mission_backtransition_stabilizer_diff_limit', 0.12
        )), 0.0), 1.0)
        self.backtransition_stabilizer_common_limit = min(max(float(getattr(
            config, 'mission_backtransition_stabilizer_common_limit', 0.12
        )), 0.0), 1.0)
        self.aero_envelope_min_forward_speed = max(float(
            getattr(
                config,
                'mission_aero_envelope_min_forward_speed',
                getattr(config, 'mission_aero_envelope_min_speed', 4.0),
            )
        ), 0.0)
        self.transition_aero_envelope_min_forward_speed = max(float(
            getattr(
                config,
                'mission_transition_aero_envelope_min_forward_speed',
                self.transition_speed,
            )
        ), self.aero_envelope_min_forward_speed)
        self.fixed_wing_aero_grace_steps = max(0, int(getattr(
            config, 'mission_fixed_wing_aero_grace_steps', 25
        )))
        self.vertical_lift_differential_scale = float(getattr(
            config, 'mission_vertical_lift_differential_scale', 0.1
        ))
        if not 0.0 <= self.vertical_lift_differential_scale <= 1.0:
            raise ValueError(
                'mission_vertical_lift_differential_scale must be between 0 and 1'
            )
        self.rotor_climb_lift_differential_scale = min(max(float(getattr(
            config, 'mission_rotor_climb_lift_differential_scale', 0.0
        )), 0.0), 1.0)
        # During rotor-borne climb the aerodynamic surfaces are neutral, so a
        # small bounded attitude loop is needed to reject Gazebo contact and
        # motor transients.  PPO still owns the collective and configured
        # differential; this loop only adds a finite counter-torque.
        self.rotor_climb_stabilizer_enable = bool(getattr(
            config, 'mission_rotor_climb_stabilizer_enable', True
        ))
        self.rotor_climb_stabilizer_roll_gain = max(float(getattr(
            config, 'mission_rotor_climb_stabilizer_roll_gain', 0.85
        )), 0.0)
        self.rotor_climb_stabilizer_roll_rate_gain = max(float(getattr(
            config, 'mission_rotor_climb_stabilizer_roll_rate_gain', 0.12
        )), 0.0)
        self.rotor_climb_stabilizer_pitch_gain = max(float(getattr(
            config, 'mission_rotor_climb_stabilizer_pitch_gain', 0.85
        )), 0.0)
        self.rotor_climb_stabilizer_pitch_rate_gain = max(float(getattr(
            config, 'mission_rotor_climb_stabilizer_pitch_rate_gain', 0.12
        )), 0.0)
        self.rotor_climb_stabilizer_diff_limit = min(max(float(getattr(
            config, 'mission_rotor_climb_stabilizer_diff_limit', 0.20
        )), 0.0), 1.0)
        self.rotor_climb_stabilizer_delta = torch.zeros(
            (self.n, 4), device=self.device
        )
        self.rotor_climb_command = torch.zeros(
            (self.n, 4), device=self.device
        )
        self.rotor_climb_pusher_enable_altitude = max(float(getattr(
            config, 'mission_rotor_climb_pusher_enable_altitude', 10.0
        )), self.takeoff_clearance)
        self.ground_lift_differential_scale = min(max(float(getattr(
            config, 'mission_ground_lift_differential_scale', 0.0)
        ), 0.0), 1.0)
        self.rotor_climb_pusher_min_action = min(max(float(getattr(
            config, 'mission_rotor_climb_pusher_min_action', -0.2
        )), -1.0), 1.0)
        self.rotor_climb_pusher_max_action = min(max(float(getattr(
            config, 'mission_rotor_climb_pusher_max_action', -0.50
        )), -1.0), 1.0)
        if self.rotor_climb_pusher_max_action < self.rotor_climb_pusher_min_action:
            raise ValueError(
                'mission_rotor_climb_pusher_max_action must be at least '
                'mission_rotor_climb_pusher_min_action'
            )
        self.rotor_climb_pusher_roll_trim_scale = max(float(getattr(
            config, 'mission_rotor_climb_pusher_roll_trim_scale', 0.01
        )), 0.0)
        self.transition_pusher_max_action = min(max(float(getattr(
            config, 'mission_transition_pusher_max_action', -0.25
        )), -1.0), 1.0)
        self.transition_pusher_speed_limit = max(float(getattr(
            config, 'mission_transition_pusher_speed_limit', 18.0
        )), self.transition_speed + 1e-6)
        self.transition_pusher_high_speed_action = min(max(float(getattr(
            config, 'mission_transition_pusher_high_speed_action', -0.65
        )), -1.0), self.transition_pusher_max_action)
        self.backtransition_lift_action_high = min(max(float(getattr(
            config, 'mission_backtransition_lift_action_high',
            self.fixed_wing_lift_action,
        )), -1.0), 1.0)
        self.backtransition_lift_action_low = min(max(float(getattr(
            config, 'mission_backtransition_lift_action_low',
            self.fixed_wing_lift_recovery_action,
        )), -1.0), 1.0)
        self.backtransition_altitude_gain = max(float(getattr(
            config, 'mission_backtransition_altitude_gain', 0.06
        )), 0.0)
        self.backtransition_velocity_gain = max(float(getattr(
            config, 'mission_backtransition_velocity_gain', 0.16
        )), 0.0)
        self.backtransition_target_sink_speed = max(float(getattr(
            config, 'mission_backtransition_target_sink_speed', 0.80
        )), 0.0)
        self.backtransition_extra_drag_coefficient = max(float(getattr(
            config, 'mission_backtransition_extra_drag_coefficient', 0.0
        )), 0.0)
        self.backtransition_extra_drag_start_distance = max(float(getattr(
            config,
            'mission_backtransition_extra_drag_start_distance',
            self.approach_distance,
        )), self.descent_capture_radius)
        self.vertical_landing_altitude_gain = max(float(getattr(
            config, 'mission_vertical_landing_altitude_gain', 0.01
        )), 0.0)
        self.vertical_landing_velocity_gain = max(float(getattr(
            config, 'mission_vertical_landing_velocity_gain', 0.12
        )), 0.0)
        self.vertical_landing_collective_limit = min(max(float(getattr(
            config, 'mission_vertical_landing_collective_limit', 0.20
        )), 0.0), 1.0)
        self.vertical_landing_target_sink_speed = max(float(getattr(
            config, 'mission_vertical_landing_target_sink_speed', 0.50
        )), 0.0)
        self.vertical_landing_position_gain = max(float(getattr(
            config, 'mission_vertical_landing_position_gain', 0.08
        )), 0.0)
        self.vertical_landing_horizontal_damping = max(float(getattr(
            config, 'mission_vertical_landing_horizontal_damping', 0.45
        )), 0.0)
        self.vertical_landing_max_accel = max(float(getattr(
            config, 'mission_vertical_landing_max_accel', 2.5
        )), 0.0)
        self.vertical_landing_max_tilt = math.radians(float(getattr(
            config, 'mission_vertical_landing_max_tilt_deg', 18.0
        )))
        self.vertical_landing_lift_differential_scale = min(max(float(getattr(
            config, 'mission_vertical_landing_lift_differential_scale', 0.24
        )), 0.0), 1.0)
        self.vertical_landing_roll_gain = max(float(getattr(
            config, 'mission_vertical_landing_roll_gain', 0.70
        )), 0.0)
        self.vertical_landing_pitch_gain = max(float(getattr(
            config, 'mission_vertical_landing_pitch_gain', 0.70
        )), 0.0)
        self.vertical_landing_roll_rate_gain = max(float(getattr(
            config, 'mission_vertical_landing_roll_rate_gain', 0.16
        )), 0.0)
        self.vertical_landing_pitch_rate_gain = max(float(getattr(
            config, 'mission_vertical_landing_pitch_rate_gain', 0.16
        )), 0.0)
        self.vertical_landing_diff_limit = min(max(float(getattr(
            config, 'mission_vertical_landing_diff_limit', 0.06
        )), 0.0), 1.0)
        self.vertical_landing_yaw_rate_gain = max(float(getattr(
            config, 'mission_vertical_landing_yaw_rate_gain', 0.035
        )), 0.0)
        self.vertical_landing_yaw_diff_limit = min(max(float(getattr(
            config, 'mission_vertical_landing_yaw_diff_limit', 0.04
        )), 0.0), 1.0)
        self.hover_radius = max(float(getattr(
            config, 'mission_hover_radius', 3.0
        )), 0.0)
        self.hover_altitude_tolerance = max(float(getattr(
            config, 'mission_hover_altitude_tolerance', 1.0
        )), 0.0)
        self.hover_max_horizontal_speed = max(float(getattr(
            config, 'mission_hover_max_horizontal_speed', 1.0
        )), 0.0)
        self.hover_max_vertical_speed = max(float(getattr(
            config, 'mission_hover_max_vertical_speed', 0.5
        )), 0.0)
        self.hover_max_tilt = math.radians(float(getattr(
            config, 'mission_hover_max_tilt_deg', 12.0
        )))
        self.hover_hold_steps = max(1, int(getattr(
            config, 'mission_hover_hold_steps', 150
        )))
        self.hover_rl_collective_residual_scale = min(max(float(getattr(
            config, 'mission_hover_rl_collective_residual_scale', 0.0
        )), 0.0), 1.0)
        self.hover_rl_differential_residual_scale = min(max(float(getattr(
            config, 'mission_hover_rl_differential_residual_scale', 0.0
        )), 0.0), 1.0)
        self.action_filter_surface_fixed_wing_only = bool(getattr(
            config, 'action_filter_surface_fixed_wing_only', False
        ))

        self.phase = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.phase_entry_step = torch.zeros_like(self.phase)
        self.backtransition_capture_count = torch.zeros_like(self.phase)
        # ``update_before_observation`` may be called more than once while
        # assembling diagnostics.  Keep the hold counter tied to control
        # steps, rather than to callback invocations.
        self.backtransition_capture_last_step = torch.full_like(
            self.phase, -1
        )
        self.hover_stable_count = torch.zeros_like(self.phase)
        self.hover_stable_last_step = torch.full_like(self.phase, -1)
        self.phase_advanced = torch.zeros(
            self.n, dtype=torch.bool, device=self.device
        )
        self.start_phase = torch.zeros_like(self.phase)
        self.target_npos = torch.zeros(self.n, device=self.device)
        self.target_epos = torch.zeros(self.n, device=self.device)
        self.target_altitude = torch.zeros(self.n, device=self.device)
        self.target_heading = torch.zeros(self.n, device=self.device)
        self.target_speed = torch.zeros(self.n, device=self.device)
        self.previous_waypoint_distance = torch.zeros(self.n, device=self.device)
        self.metric_waypoint_distance = torch.zeros(self.n, device=self.device)
        self.metric_landing_distance = torch.zeros(self.n, device=self.device)
        self.metric_gps_age = torch.zeros(self.n, device=self.device)
        self.training_success_count = torch.zeros((), device=self.device)
        self.training_failure_count = torch.zeros((), device=self.device)
        self.rollout_metrics_active = False
        self.rollout_episode_opportunities = torch.zeros((), device=self.device)
        self.rollout_phase_reach_count = torch.zeros(
            self.PHASE_COUNT, device=self.device
        )
        # Rollout-wide action smoothness counters.  Keeping the reductions on
        # the simulator device avoids copying an 8-channel trace for every
        # environment to the host just to populate TensorBoard.
        self.rollout_action_delta_abs_sum = torch.zeros((), device=self.device)
        self.rollout_action_delta_sq_sum = torch.zeros((), device=self.device)
        self.rollout_raw_action_delta_abs_sum = torch.zeros((), device=self.device)
        self.rollout_action_slew_limited_sum = torch.zeros((), device=self.device)
        self.rollout_action_metric_samples = torch.zeros((), device=self.device)
        self._env_ref = None

        if self.num_observation != self.OBSERVATION_SIZE:
            self.num_observation = self.OBSERVATION_SIZE
            self.load_observation_space()
        if self.num_actions != 8:
            self.num_actions = 8
            self.load_action_space()
        self.use_privileged_critic = bool(
            getattr(config, 'use_privileged_critic', True)
        )
        self.num_critic_observation = self.CRITIC_OBSERVATION_SIZE
        self.load_critic_observation_space()
        self.constraint_penalty = torch.zeros(self.n, device=self.device)

        self.reward_functions = [
            VTOLMissionReward(config),
            VTOLMissionEventReward(config),
        ]
        self.termination_conditions = [
            GroundCollision(config),
            ExtremeAngle(config),
            ExtremeOmega(config),
            ExtremeState(config),
            HighSpeed(config),
            Overload(config),
            VTOLMissionBoundary(config),
            VTOLMissionSuccess(config),
            VTOLMissionTimeout(config),
        ]

    def _uniform(self, count, low, high):
        return torch.rand(count, device=self.device) * (high - low) + low

    def _sample_start_phases(self, count):
        if not self.curriculum_enabled:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        return torch.multinomial(
            self.curriculum_weights, count, replacement=True
        )

    def _sample_mission_targets(self, reset):
        """Assign one immutable goal to each episode in the reset subset."""
        count = int(reset.sum().item())
        if count == 0:
            return

        if self.randomize_target:
            distance = self._uniform(
                count, self.target_distance_min, self.target_distance_max
            )
            bearing = self._uniform(
                count, self.target_bearing_min, self.target_bearing_max
            )
            goal_n = self.start_n + distance * torch.cos(bearing)
            goal_e = self.start_e + distance * torch.sin(bearing)
        else:
            goal_n = torch.full(
                (count,), self.landing_n, dtype=torch.float32, device=self.device
            )
            goal_e = torch.full(
                (count,), self.landing_e, dtype=torch.float32, device=self.device
            )

        route_n = goal_n - self.start_n
        route_e = goal_e - self.start_e
        route_length = torch.sqrt(route_n * route_n + route_e * route_e)
        route_heading = torch.atan2(route_e, route_n)
        self.goal_n[reset] = goal_n
        self.goal_e[reset] = goal_e
        self.route_length_batch[reset] = route_length
        self.route_unit_n_batch[reset] = route_n / route_length.clamp_min(1e-6)
        self.route_unit_e_batch[reset] = route_e / route_length.clamp_min(1e-6)
        self.route_heading_batch[reset] = route_heading
        if self.final_heading_follows_route:
            self.final_heading_batch[reset] = route_heading
        else:
            self.final_heading_batch[reset] = self.final_heading

    def reset(self, env):
        self._env_ref = env
        reset = (
            env.is_done.bool()
            | env.bad_done.bool()
            | env.exceed_time_limit.bool()
        )
        count = int(reset.sum().item())
        if count == 0:
            return
        if not hasattr(env.model, 'get_gps_state'):
            raise TypeError('VTOLMissionTask requires GazeboModel GPS support')

        self._sample_mission_targets(reset)
        sampled_phase = self._sample_start_phases(count)
        self.phase[reset] = sampled_phase
        self.start_phase[reset] = sampled_phase
        self.phase_entry_step[reset] = 0
        self.backtransition_capture_count[reset] = 0
        self.backtransition_capture_last_step[reset] = -1
        self.hover_stable_count[reset] = 0
        self.hover_stable_last_step[reset] = -1
        self.phase_advanced[reset] = False
        self.constraint_penalty[reset] = 0.0
        self._initialize_curriculum_states(env, reset)
        self._refresh_guidance()
        self.previous_waypoint_distance[reset] = self._waypoint_distance(env)[reset]
        if self.rollout_metrics_active:
            self.rollout_episode_opportunities += count
            for phase_id in range(1, self.PHASE_COUNT):
                self.rollout_phase_reach_count[phase_id] += (
                    sampled_phase >= phase_id
                ).sum()

    def begin_training_rollout(self):
        """Start GPU-side phase reach accounting for one PPO rollout."""
        self.rollout_metrics_active = True
        self.rollout_episode_opportunities.fill_(self.n)
        self.rollout_phase_reach_count.zero_()
        self.rollout_action_delta_abs_sum.zero_()
        self.rollout_action_delta_sq_sum.zero_()
        self.rollout_raw_action_delta_abs_sum.zero_()
        self.rollout_action_slew_limited_sum.zero_()
        self.rollout_action_metric_samples.zero_()
        for phase_id in range(1, self.PHASE_COUNT):
            self.rollout_phase_reach_count[phase_id] = (
                self.phase >= phase_id
            ).sum()

    def _initialize_curriculum_states(self, env, reset):
        model = env.model
        ground = reset & (self.phase == self.TAKEOFF)
        model.s[ground, 0] = self.start_n
        model.s[ground, 1] = self.start_e
        model.s[ground, 5] = self.route_heading_batch[ground]
        model.set_initial_actuators(ground, [0.0, 0.0, 0.0, 0.0, 0.0])

        # The public release always starts on the same ground pose.  Returning
        # here also prevents the optional curriculum initializer below from
        # drawing unused random states for inactive phases.
        if not self.curriculum_enabled:
            model.s[ground, 3:5] = 0.0
            model.s[ground, 6:12] = 0.0
            model.sync_reset_state(reset)
            return

        fixed_wing_start = torch.clamp(
            1.0 - self.approach_distance / self.route_length_batch,
            min=0.21,
        )
        backtransition_start = torch.clamp(
            1.0 - self.approach_distance / self.route_length_batch,
            min=0.0,
        )
        capture_start = torch.clamp(
            1.0 - self.descent_capture_radius / self.route_length_batch,
            min=0.01,
        )
        phase_ranges = {
            self.ROTOR_CLIMB: (0.00, 0.05, 2.0, self.cruise_altitude - 1.0, 0.0, 2.0),
            self.TRANSITION: (0.02, 0.20, self.cruise_altitude - 2.0,
                              self.cruise_altitude + 2.0, 4.0, self.transition_speed),
            self.FIXED_WING: (0.20, fixed_wing_start,
                              self.cruise_altitude - 2.0, self.cruise_altitude + 2.0,
                              self.transition_speed, self.cruise_speed + 2.0),
            self.BACK_TRANSITION: (backtransition_start, capture_start,
                                   self.landing_hover_altitude, self.cruise_altitude,
                                   self.curriculum_backtransition_speed_min,
                                   self.curriculum_backtransition_speed_max),
            self.VERTICAL_LANDING: (capture_start,
                                    1.0, self.landing_altitude + 1.0,
                                    self.landing_hover_altitude + 2.0, 0.0,
                                    self.backtransition_speed),
        }

        for phase_id, values in phase_ranges.items():
            mask = reset & (self.phase == phase_id)
            f_low, f_high, alt_low, alt_high, speed_low, speed_high = values
            fraction = self._uniform(self.n, f_low, f_high)
            cross_track_std = (
                self.curriculum_backtransition_cross_track_std
                if phase_id == self.BACK_TRANSITION
                else self.curriculum_cross_track_std
            )
            cross_track = torch.randn(
                self.n, device=self.device
            ) * cross_track_std
            model.s[mask, 0] = (
                self.start_n
                + fraction[mask] * (self.goal_n[mask] - self.start_n)
                - cross_track[mask] * self.route_unit_e_batch[mask]
            )
            model.s[mask, 1] = (
                self.start_e
                + fraction[mask] * (self.goal_e[mask] - self.start_e)
                + cross_track[mask] * self.route_unit_n_batch[mask]
            )
            model.s[mask, 2] = self._uniform(self.n, alt_low, alt_high)[mask]
            model.s[mask, 3] = torch.randn(self.n, device=self.device)[mask] * 0.02
            pitch = torch.randn(self.n, device=self.device) * 0.02
            if phase_id == self.FIXED_WING:
                pitch += 0.07
            model.s[mask, 4] = pitch[mask]
            heading_std = (
                self.curriculum_backtransition_heading_std
                if phase_id == self.BACK_TRANSITION
                else self.curriculum_heading_std
            )
            model.s[mask, 5] = (
                self.route_heading_batch[mask]
                + torch.randn(self.n, device=self.device)[mask]
                * heading_std
            )
            speed = self._uniform(self.n, speed_low, speed_high)
            model.s[mask, 6] = speed[mask]
            model.s[mask, 7] = torch.randn(self.n, device=self.device)[mask] * 0.2
            model.s[mask, 8] = torch.where(
                self.phase[mask] == self.FIXED_WING,
                speed[mask] * torch.tan(pitch[mask]),
                torch.randn(self.n, device=self.device)[mask] * 0.2,
            )
            model.s[mask, 9:12] = (
                torch.randn((self.n, 3), device=self.device)[mask] * 0.02
            )

            hover_omega = torch.sqrt(
                (model.mass_curr * model.dynamics.g / (4.0 * 2.0e-5))
                .clamp_min(0.0)
            )
            motors = torch.zeros((self.n, 5), device=self.device)
            if phase_id == self.ROTOR_CLIMB:
                motors[:, 0:4] = hover_omega.reshape(-1, 1)
            elif phase_id == self.TRANSITION:
                motors[:, 0:4] = 0.85 * hover_omega.reshape(-1, 1)
                motors[:, 4] = 1400.0
            elif phase_id == self.FIXED_WING:
                motors[:, 4] = 1800.0
            elif phase_id == self.BACK_TRANSITION:
                motors[:, 0:4] = 0.70 * hover_omega.reshape(-1, 1)
                motors[:, 4] = 1100.0
            else:
                motors[:, 0:4] = hover_omega.reshape(-1, 1)
            model.set_initial_actuators(mask, motors)

        model.sync_reset_state(reset)

    def _refresh_guidance(self, env=None):
        if env is None:
            env = self._env_ref
        if env is None:
            raise RuntimeError('VTOLMissionTask guidance requires an environment')
        approach_n = (
            self.goal_n - self.route_unit_n_batch * self.approach_distance
        )
        approach_e = (
            self.goal_e - self.route_unit_e_batch * self.approach_distance
        )

        self.target_npos[:] = self.goal_n
        self.target_epos[:] = self.goal_e
        terminal_altitude = (
            self.landing_hover_altitude
            if self.terminal_mode == 'hover'
            else self.landing_altitude
        )
        self.target_altitude[:] = terminal_altitude
        self.target_heading[:] = self.final_heading_batch
        self.target_speed[:] = 0.0

        vertical = (self.phase == self.TAKEOFF) | (self.phase == self.ROTOR_CLIMB)
        self.target_npos[vertical] = self.start_n
        self.target_epos[vertical] = self.start_e
        self.target_altitude[vertical] = self.cruise_altitude
        self.target_heading[vertical] = self.route_heading_batch[vertical]
        self.target_speed[vertical] = self.rotor_climb_speed

        cruise = (self.phase == self.TRANSITION) | (self.phase == self.FIXED_WING)
        self.target_npos[cruise] = approach_n[cruise]
        self.target_epos[cruise] = approach_e[cruise]
        self.target_altitude[cruise] = self.cruise_altitude
        self.target_heading[cruise] = self.route_heading_batch[cruise]
        self.target_speed[self.phase == self.TRANSITION] = self.transition_speed
        self.target_speed[self.phase == self.FIXED_WING] = self.cruise_speed

        backtransition = self.phase == self.BACK_TRANSITION
        if torch.any(backtransition):
            # Do not command an immediate drop from cruise altitude as soon as
            # reverse transition starts.  The aircraft has up to
            # ``approach_distance`` metres to brake, so descend on a smooth
            # corridor and reach hover altitude only at the capture boundary.
            npos, epos, _ = env.model.get_position()
            distance = torch.sqrt(
                (npos - self.goal_n) ** 2
                + (epos - self.goal_e) ** 2
            )
            fraction = (
                (distance - self.descent_capture_radius)
                / (self.approach_distance - self.descent_capture_radius)
            ).clamp(0.0, 1.0)
            self.target_altitude[backtransition] = (
                self.landing_hover_altitude
                + fraction[backtransition]
                * (self.cruise_altitude - self.landing_hover_altitude)
            )
            self.target_heading[backtransition] = self.route_heading_batch[
                backtransition
            ]
            self.target_speed[backtransition] = self.approach_speed

    def update_before_observation(self, env):
        self._env_ref = env
        self.phase_advanced.zero_()
        phase_before = self.phase.clone()
        elapsed = env.step_count - self.phase_entry_step
        dwell_ok = elapsed >= self.phase_min_steps[self.phase]
        npos, epos, altitude = env.model.get_position()
        speed = env.model.get_TAS()
        vel_n, vel_e, vel_up = env.model.get_world_velocity()
        horizontal_speed = torch.sqrt(
            (vel_n * vel_n + vel_e * vel_e).clamp_min(0.0)
        )
        contact = env.model.get_ground_contact_state()
        roll, pitch, _ = env.model.get_posture()
        distance_to_landing = torch.sqrt(
            (npos - self.goal_n) ** 2 + (epos - self.goal_e) ** 2
        )
        landing_dn = self.goal_n - npos
        landing_de = self.goal_e - epos
        p, q, _ = env.model.get_angular_velocity()
        closing_speed = (
            vel_n * landing_dn + vel_e * landing_de
        ) / distance_to_landing.clamp_min(1e-6)

        # Require the aircraft to remain inside the capture envelope for a few
        # control cycles.  This prevents a single threshold crossing from
        # switching to vertical mode while the aircraft is still flying away.
        capture_candidate = (
            (distance_to_landing <= self.descent_capture_radius)
            & (altitude <= self.descent_capture_altitude)
            & (horizontal_speed <= self.backtransition_speed)
            & (closing_speed >= -self.backtransition_max_receding_speed)
            & (vel_up.abs() <= self.backtransition_max_vertical_speed)
            & (roll.abs() <= self.descent_capture_max_tilt)
            & (pitch.abs() <= self.descent_capture_max_tilt)
            & (p.abs() <= self.descent_capture_max_roll_rate)
            & (q.abs() <= self.descent_capture_max_pitch_rate)
        )
        in_backtransition = phase_before == self.BACK_TRANSITION
        new_control_step = (
            (env.step_count > 0)
            & (env.step_count != self.backtransition_capture_last_step)
        )
        self.backtransition_capture_count = torch.where(
            in_backtransition & new_control_step & capture_candidate,
            self.backtransition_capture_count + 1,
            torch.where(
                in_backtransition & new_control_step,
                torch.zeros_like(self.backtransition_capture_count),
                self.backtransition_capture_count,
            ),
        )
        self.backtransition_capture_last_step.copy_(env.step_count)
        capture_ready = self.backtransition_capture_count >= (
            self.backtransition_capture_hold_steps
        )

        conditions = (
            (~contact['on_ground']) & (altitude >= self.takeoff_clearance),
            altitude >= self.cruise_altitude - 1.5,
            (speed >= self.transition_speed) & (altitude >= self.cruise_altitude - 4.0),
            distance_to_landing <= self.approach_distance,
            capture_ready,
        )
        for phase_id, condition in enumerate(conditions):
            advance = (phase_before == phase_id) & dwell_ok & condition
            self.phase[advance] = phase_id + 1
            self.phase_entry_step[advance] = env.step_count[advance]
            self.phase_advanced |= advance
            if self.rollout_metrics_active:
                self.rollout_phase_reach_count[phase_id + 1] += advance.sum()
        self._refresh_guidance()
        self._update_hover_stability(env)

    def hover_stable_candidate(self, env):
        """Return environments currently inside the terminal hover envelope."""
        npos, epos, altitude = env.model.get_position()
        vel_n, vel_e, vel_up = env.model.get_world_velocity()
        roll, pitch, _ = env.model.get_posture()
        contact = env.model.get_ground_contact_state()
        horizontal_error = torch.sqrt(
            (npos - self.goal_n) ** 2 + (epos - self.goal_e) ** 2
        )
        horizontal_speed = torch.sqrt(
            (vel_n * vel_n + vel_e * vel_e).clamp_min(0.0)
        )
        return (
            (self.phase == self.VERTICAL_HOVER)
            & ~contact['on_ground']
            & (horizontal_error <= self.hover_radius)
            & (
                (altitude - self.landing_hover_altitude).abs()
                <= self.hover_altitude_tolerance
            )
            & (horizontal_speed <= self.hover_max_horizontal_speed)
            & (vel_up.abs() <= self.hover_max_vertical_speed)
            & (roll.abs() <= self.hover_max_tilt)
            & (pitch.abs() <= self.hover_max_tilt)
        )

    def _update_hover_stability(self, env):
        if self.terminal_mode != 'hover':
            return
        new_control_step = (
            (env.step_count > 0)
            & (env.step_count != self.hover_stable_last_step)
        )
        stable = self.hover_stable_candidate(env)
        self.hover_stable_count = torch.where(
            new_control_step & stable,
            self.hover_stable_count + 1,
            torch.where(
                new_control_step,
                torch.zeros_like(self.hover_stable_count),
                self.hover_stable_count,
            ),
        )
        self.hover_stable_last_step.copy_(env.step_count)

    def _set_phase_action_smoothing(self, env):
        """Set per-phase surface smoothing without moving this state to CPU.

        The learned surface timing is needed during transition and capture;
        applying a low-pass there can delay a corrective turn enough to leave
        the route.  Fixed-wing cruise has the largest high-frequency surface
        jitter, so that is the default phase in which the filter is active.
        Motor channels keep their configured Gazebo lag/slew limits in every
        phase.
        """
        model = getattr(env, 'model', None)
        if model is None or not hasattr(model, 'action_filter_alpha_override'):
            return
        model.action_filter_alpha_override.fill_(-1.0)
        model.action_slew_rate_override.fill_(-1.0)
        if not self.action_filter_surface_fixed_wing_only:
            return

        fixed_wing = self.phase == self.FIXED_WING
        surface_alpha = torch.where(
            fixed_wing,
            torch.full_like(self.phase, model.action_filter_surface_alpha,
                            dtype=env.model.s.dtype),
            torch.ones(self.n, device=self.device, dtype=env.model.s.dtype),
        )
        # A large limit is effectively unbounded in [-1, 1] action space.  In
        # fixed-wing mode use the configured surface slew limit; elsewhere do
        # not add a new delay to the policy's transition/capture commands.
        configured_surface_rate = model.action_slew_rate[:, 5:8].expand(
            self.n, 3
        )
        surface_rate = torch.where(
            fixed_wing.reshape(-1, 1),
            configured_surface_rate,
            torch.full((self.n, 3), 2.0, device=self.device,
                       dtype=env.model.s.dtype),
        )
        model.action_filter_alpha_override[:, 5:8] = surface_alpha.reshape(-1, 1)
        model.action_slew_rate_override[:, 5:8] = surface_rate

    def maybe_override_action(self, env, action):
        self._set_phase_action_smoothing(env)
        if action.shape[1] < 8:
            return action
        if not self.gate_vertical_actuators and not self.gate_backtransition_pusher:
            # A bounded reverse-transition surface command is still useful
            # even when the pusher/vertical gates are disabled.
            if self.backtransition_surface_action_scale >= 1.0:
                return action
        action = action.clone()
        policy_lift_action = action[:, 0:4].clone()
        # GPU-resident diagnostics for the current command.  These are reset
        # on every call so stale values cannot leak across phase changes.
        self.rotor_climb_stabilizer_delta.zero_()
        self.rotor_climb_command.zero_()

        backtransition = self.phase == self.BACK_TRANSITION
        fixed_wing = self.phase == self.FIXED_WING
        if hasattr(env.model, 'aero_force_scale'):
            transition = self.phase == self.TRANSITION
            rotor_borne = (
                (self.phase == self.TAKEOFF)
                | (self.phase == self.ROTOR_CLIMB)
            )
            forward_speed = env.model.s[:, 6].clamp_min(0.0)
            transition_fraction = (
                (forward_speed - self.transition_speed)
                / max(self.cruise_speed - self.transition_speed, 1e-6)
            ).clamp(0.0, 1.0)
            transition_scale = self.transition_aero_scale + (
                1.0 - self.transition_aero_scale
            ) * transition_fraction
            scale = torch.ones_like(env.model.aero_force_scale)
            scale = torch.where(
                rotor_borne,
                torch.full_like(scale, self.rotor_climb_aero_scale),
                scale,
            )
            scale = torch.where(transition, transition_scale, scale)
            scale = torch.where(
                backtransition,
                torch.full_like(scale, self.backtransition_aero_scale),
                scale,
            )
            env.model.aero_force_scale[:] = scale
            if hasattr(env.model.dynamics, 'aero_scale'):
                env.model.dynamics.aero_scale = env.model.aero_force_scale
        if self.backtransition_surface_action_scale < 1.0:
            action[backtransition, 5:8] *= self.backtransition_surface_action_scale

        if self.gate_vertical_actuators:
            vertical = (
                (self.phase == self.TAKEOFF)
                | (self.phase == self.ROTOR_CLIMB)
                | (self.phase == self.VERTICAL_LANDING)
            )
            _, _, altitude = env.model.get_position()
            lift = action[vertical, 0:4]
            collective = lift.mean(dim=1, keepdim=True)
            differential_scale = torch.full_like(altitude, self.vertical_lift_differential_scale)
            differential_scale = torch.where(
                self.phase == self.ROTOR_CLIMB,
                torch.full_like(
                    differential_scale,
                    self.rotor_climb_lift_differential_scale,
                ),
                differential_scale,
            )
            differential_scale = torch.where(
                (self.phase == self.TAKEOFF)
                | (self.phase == self.VERTICAL_LANDING),
                torch.full_like(differential_scale, self.ground_lift_differential_scale),
                differential_scale,
            )
            action[vertical, 0:4] = collective + (
                differential_scale[vertical].reshape(-1, 1)
                * (lift - collective)
            )
            # Keep the pusher off while close to the ground, then allow it to
            # build forward speed during the upper part of rotor climb. This
            # is necessary for the subsequent 11 m/s transition gate while
            # preserving the no-tip-over takeoff envelope.
            pusher_off = (
                (self.phase == self.TAKEOFF)
                | (self.phase == self.VERTICAL_LANDING)
                | (
                    (self.phase == self.ROTOR_CLIMB)
                    & (altitude < self.rotor_climb_pusher_enable_altitude)
                )
            )
            action[pusher_off, 4] = -1.0
            pusher_enabled = (
                (self.phase == self.ROTOR_CLIMB)
                & (altitude >= self.rotor_climb_pusher_enable_altitude)
            )
            action[pusher_enabled, 4] = torch.maximum(
                action[pusher_enabled, 4],
                torch.full_like(action[pusher_enabled, 4],
                                self.rotor_climb_pusher_min_action),
            )
            action[pusher_enabled, 4] = torch.minimum(
                action[pusher_enabled, 4],
                torch.full_like(action[pusher_enabled, 4],
                                self.rotor_climb_pusher_max_action),
            )
            # The pusher propeller has a reaction torque around body x.  Keep
            # the lift rotors equal for policy exploration, but add a small
            # deterministic counter-torque proportional to pusher throttle.
            pusher_fraction = 0.5 * (action[pusher_enabled, 4] + 1.0)
            roll_trim = pusher_fraction * self.rotor_climb_pusher_roll_trim_scale
            trim = torch.stack((roll_trim, -roll_trim, -roll_trim, roll_trim), dim=1)
            action[pusher_enabled, 0:4] = (
                action[pusher_enabled, 0:4] + trim
            ).clamp(-1.0, 1.0)

            # No aerodynamic surface authority exists in rotor climb.  Use
            # the measured Gazebo attitude and body rates to add a bounded
            # counter-torque through the lift-rotor differential.  The
            # [+,-,-,+] mapping for positive roll is the same calibrated
            # convention used by the vertical-landing controller.
            rotor_climb = self.phase == self.ROTOR_CLIMB
            if self.rotor_climb_stabilizer_enable and torch.any(rotor_climb):
                roll, pitch, _ = env.model.get_posture()
                p, q, _ = env.model.get_angular_velocity()
                roll_delta = (
                    -self.rotor_climb_stabilizer_roll_gain * roll
                    - self.rotor_climb_stabilizer_roll_rate_gain * p
                )
                pitch_delta = (
                    -self.rotor_climb_stabilizer_pitch_gain * pitch
                    - self.rotor_climb_stabilizer_pitch_rate_gain * q
                )
                axis_limit = self.rotor_climb_stabilizer_diff_limit
                roll_delta = roll_delta.clamp(-axis_limit, axis_limit)
                pitch_delta = pitch_delta.clamp(-axis_limit, axis_limit)
                stabilizer_delta = torch.stack((
                    pitch_delta - roll_delta,
                    -pitch_delta + roll_delta,
                    pitch_delta + roll_delta,
                    -pitch_delta - roll_delta,
                ), dim=1)
                collective = action[:, 0:4].mean(dim=1, keepdim=True)
                differential = action[:, 0:4] - collective
                differential = differential + stabilizer_delta
                # Keep the deterministic correction finite even if a policy
                # sample is already close to the action-space boundary.
                max_abs = differential.abs().amax(dim=1, keepdim=True)
                limit = torch.as_tensor(
                    axis_limit, device=action.device, dtype=action.dtype
                )
                differential = differential * torch.minimum(
                    torch.ones_like(max_abs),
                    limit / max_abs.clamp_min(1.0e-6),
                )
                action[rotor_climb, 0:4] = (
                    collective + differential
                )[rotor_climb].clamp(-1.0, 1.0)
                self.rotor_climb_stabilizer_delta[rotor_climb] = (
                    stabilizer_delta[rotor_climb]
                )
            self.rotor_climb_command[rotor_climb] = action[
                rotor_climb, 0:4
            ]
            action[vertical, 5:8] = 0.0

            # Vertical landing gets a mass-aware hover feed-forward and a
            # bounded altitude/vertical-speed correction.  This removes the
            # need for PPO to discover the absolute hover throttle from a
            # sparse terminal reward while leaving the other flight phases
            # policy-controlled.  After contact, unload the rotors gradually
            # so the contact solver can settle the vehicle without bouncing.
            terminal_vertical = self.phase == self.VERTICAL_LANDING
            if torch.any(terminal_vertical):
                vel_n, vel_e, vel_up = env.model.get_world_velocity()
                motor_constants = getattr(
                    env.model.dynamics, 'motor_constant', (2.0e-5,)
                )
                motor_constant = float(motor_constants[0])
                hover_omega = torch.sqrt(
                    (
                        env.model.mass_curr * env.model.dynamics.g
                        / (4.0 * max(motor_constant, 1.0e-8))
                    ).clamp_min(0.0)
                )
                rotor_scale = env.model.motor_cmd_scaling[0]
                if env.model.action_is_gazebo_control:
                    hover_action = hover_omega / rotor_scale
                else:
                    hover_action = 2.0 * hover_omega / rotor_scale - 1.0
                target_altitude = (
                    self.landing_hover_altitude
                    if self.terminal_mode == 'hover'
                    else self.landing_altitude
                )
                altitude_error = altitude - target_altitude
                if self.terminal_mode == 'hover':
                    target_sink = torch.zeros_like(altitude)
                else:
                    target_sink = -torch.minimum(
                        torch.full_like(
                            altitude, self.vertical_landing_target_sink_speed
                        ),
                        0.22 * torch.sqrt(altitude_error.clamp_min(0.0)),
                    )
                collective = (
                    hover_action
                    - self.vertical_landing_altitude_gain * altitude_error
                    + self.vertical_landing_velocity_gain * (target_sink - vel_up)
                )
                collective = torch.maximum(
                    collective,
                    hover_action - self.vertical_landing_collective_limit,
                )
                collective = torch.minimum(
                    collective,
                    hover_action + self.vertical_landing_collective_limit,
                )
                contact_state = env.model.get_ground_contact_state()
                touchdown_collective = hover_action - min(
                    self.vertical_landing_collective_limit, 0.10
                )
                if self.terminal_mode == 'landing':
                    collective = torch.where(
                        contact_state['on_ground'], touchdown_collective, collective
                    )
                action[terminal_vertical, 0:4] = collective[
                    terminal_vertical
                ].reshape(-1, 1).clamp(-1.0, 1.0)

                # Cascade the landing position controller through an attitude
                # and angular-rate loop.  The outer loop is expressed in the
                # world north/east frame; the inner loop must use the current
                # yaw to rotate that request into body forward/right axes.
                # This avoids accumulating roll/pitch when the vehicle enters
                # vertical mode with a non-zero attitude or rate.
                npos, epos, _ = env.model.get_position()
                _, _, yaw = env.model.get_posture()
                roll, pitch, _ = env.model.get_posture()
                p, q, r = env.model.get_angular_velocity()
                north_error = self.goal_n - npos
                east_error = self.goal_e - epos
                accel_n = (
                    self.vertical_landing_position_gain * north_error
                    - self.vertical_landing_horizontal_damping * vel_n
                )
                accel_e = (
                    self.vertical_landing_position_gain * east_error
                    - self.vertical_landing_horizontal_damping * vel_e
                )
                accel_norm = torch.sqrt(accel_n * accel_n + accel_e * accel_e)
                accel_scale = torch.minimum(
                    torch.ones_like(accel_norm),
                    self.vertical_landing_max_accel
                    / accel_norm.clamp_min(1.0e-6),
                )
                accel_n = accel_n * accel_scale
                accel_e = accel_e * accel_scale
                # A contact patch should not keep commanding a horizontal
                # tilt after touchdown.  The attitude loop still damps any
                # residual roll/pitch on the ground.
                contact_mask = contact_state['on_ground']
                accel_n = torch.where(contact_mask, torch.zeros_like(accel_n), accel_n)
                accel_e = torch.where(contact_mask, torch.zeros_like(accel_e), accel_e)
                accel_forward = (
                    torch.cos(yaw) * accel_n + torch.sin(yaw) * accel_e
                )
                accel_right = (
                    -torch.sin(yaw) * accel_n + torch.cos(yaw) * accel_e
                )
                desired_pitch = -torch.atan2(
                    accel_forward, torch.full_like(accel_forward, 9.807)
                ).clamp(
                    -self.vertical_landing_max_tilt,
                    self.vertical_landing_max_tilt,
                )
                desired_roll = torch.atan2(
                    accel_right, torch.full_like(accel_right, 9.807)
                ).clamp(
                    -self.vertical_landing_max_tilt,
                    self.vertical_landing_max_tilt,
                )
                pitch_delta = (
                    self.vertical_landing_pitch_gain * (desired_pitch - pitch)
                    - self.vertical_landing_pitch_rate_gain * q
                ).clamp(
                    -self.vertical_landing_diff_limit,
                    self.vertical_landing_diff_limit,
                )
                roll_delta = (
                    self.vertical_landing_roll_gain * (desired_roll - roll)
                    - self.vertical_landing_roll_rate_gain * p
                ).clamp(
                    -self.vertical_landing_diff_limit,
                    self.vertical_landing_diff_limit,
                )
                yaw_delta = (
                    -self.vertical_landing_yaw_rate_gain * r
                ).clamp(
                    -self.vertical_landing_yaw_diff_limit,
                    self.vertical_landing_yaw_diff_limit,
                )
                rotor_delta = torch.stack((
                    pitch_delta - roll_delta + yaw_delta,
                    -pitch_delta + roll_delta + yaw_delta,
                    pitch_delta + roll_delta - yaw_delta,
                    -pitch_delta - roll_delta - yaw_delta,
                ), dim=1)
                terminal_action = collective.reshape(-1, 1) + rotor_delta
                if self.terminal_mode == 'hover':
                    policy_collective = policy_lift_action.mean(
                        dim=1, keepdim=True
                    )
                    policy_differential = (
                        policy_lift_action - policy_collective
                    )
                    terminal_action = (
                        terminal_action
                        + self.hover_rl_collective_residual_scale
                        * torch.tanh(policy_collective)
                        + self.hover_rl_differential_residual_scale
                        * torch.tanh(policy_differential)
                    )
                action[terminal_vertical, 0:4] = terminal_action[
                    terminal_vertical
                ].clamp(-1.0, 1.0)

        # In wing-borne fixed-wing mode the four lift rotors are stopped; the
        # pusher and control surfaces remain policy-controlled.  Leaving the
        # hover rotors at their normalized mean action produces nearly hover
        # thrust and drives the vehicle through the altitude safety boundary.
        if self.gate_fixed_wing_lift_rotors:
            fixed_wing = self.phase == self.FIXED_WING
            _, _, altitude = env.model.get_position()
            speed = env.model.get_TAS()
            speed_fraction = (
                (speed - self.fixed_wing_lift_speed_start)
                / (self.fixed_wing_lift_speed_end - self.fixed_wing_lift_speed_start)
            ).clamp(0.0, 1.0)
            speed_lift_action = self.fixed_wing_lift_low_speed_action + (
                speed_fraction * (
                    self.fixed_wing_lift_action
                    - self.fixed_wing_lift_low_speed_action
                )
            )
            # A zero-airflow fixed-wing state is only a diagnostic/test setup,
            # not a valid handover condition.  Preserve the configured cruise
            # trim there; the low-speed recovery trim applies once the model
            # actually has forward airspeed.
            speed_lift_action = torch.where(
                speed < 0.5,
                torch.full_like(speed_lift_action, self.fixed_wing_lift_action),
                speed_lift_action,
            )
            recovery_fraction = (
                (altitude - self.min_cruise_altitude)
                / (self.fixed_wing_lift_recovery_altitude - self.min_cruise_altitude)
            ).clamp(0.0, 1.0)
            lift_action = self.fixed_wing_lift_recovery_action + recovery_fraction * (
                speed_lift_action
                - self.fixed_wing_lift_recovery_action
            )
            action[fixed_wing, 0:4] = lift_action[fixed_wing].reshape(-1, 1)

        # Keep the wing-borne vehicle inside a recoverable attitude envelope.
        # The learned surfaces are unchanged near trim; as bank or pitch grows,
        # blend toward a bounded PD correction before the hard angle termination
        # is reached.  Positive roll uses positive (left-right) elevon
        # differential for this model's sign convention.
        if self.fixed_wing_stabilizer_enable:
            fixed_wing = self.phase == self.FIXED_WING
            if torch.any(fixed_wing):
                roll, pitch, _ = env.model.get_posture()
                p, q, _ = env.model.get_angular_velocity()
                attitude_mag = torch.maximum(roll.abs(), pitch.abs())
                blend = (
                    (attitude_mag - self.fixed_wing_stabilizer_start)
                    / (self.fixed_wing_stabilizer_full
                       - self.fixed_wing_stabilizer_start)
                ).clamp(0.0, 1.0)
                roll_trim = (
                    self.fixed_wing_stabilizer_roll_gain * roll
                    + self.fixed_wing_stabilizer_roll_rate_gain * p
                ).clamp(
                    -self.fixed_wing_stabilizer_diff_limit,
                    self.fixed_wing_stabilizer_diff_limit,
                )
                pitch_trim = (
                    -self.fixed_wing_stabilizer_pitch_gain * pitch
                    - self.fixed_wing_stabilizer_pitch_rate_gain * q
                ).clamp(
                    -self.fixed_wing_stabilizer_common_limit,
                    self.fixed_wing_stabilizer_common_limit,
                )
                policy_common = 0.5 * (action[:, 5] + action[:, 6])
                policy_diff = 0.5 * (action[:, 5] - action[:, 6])
                common = (
                    (1.0 - blend) * policy_common + blend * pitch_trim
                ).clamp(-1.0, 1.0)
                differential = (
                    (1.0 - blend) * policy_diff + blend * roll_trim
                ).clamp(-1.0, 1.0)
                action[fixed_wing, 5] = (
                    common[fixed_wing] + differential[fixed_wing]
                ).clamp(-1.0, 1.0)
                action[fixed_wing, 6] = (
                    common[fixed_wing] - differential[fixed_wing]
                ).clamp(-1.0, 1.0)
                action[fixed_wing, 7] = (
                    (1.0 - blend[fixed_wing]) * action[fixed_wing, 7]
                    + blend[fixed_wing] * pitch_trim[fixed_wing]
                ).clamp(-1.0, 1.0)

        # Add a bounded route-following roll trim in fixed-wing flight. The
        # learned policy still controls the surfaces inside the normal
        # corridor; this only supplies the missing lateral restoring signal
        # when heading/cross-track error is large enough to leave the route.
        if self.fixed_wing_nav_enable and torch.any(fixed_wing):
            npos, epos, _ = env.model.get_position()
            _, _, heading = env.model.get_posture()
            cross_track = (
                -(npos - self.start_n) * self.route_unit_e_batch
                + (epos - self.start_e) * self.route_unit_n_batch
            )
            heading_error = wrap_PI(self.route_heading_batch - heading)
            # The standard_vtol elevon convention produces a negative roll
            # acceleration for positive left-right differential. Invert the
            # geometric command so a positive heading/cross-track error turns
            # back toward the route instead of amplifying the departure.
            roll_command = -(
                self.fixed_wing_nav_heading_gain * heading_error
                + self.fixed_wing_nav_cross_track_gain * torch.atan(
                    cross_track / self.fixed_wing_nav_lookahead
                )
            ).clamp(
                -self.fixed_wing_nav_roll_limit,
                self.fixed_wing_nav_roll_limit,
            )
            roll_trim = (
                self.fixed_wing_stabilizer_roll_gain * roll_command
            ).clamp(
                -self.fixed_wing_stabilizer_diff_limit,
                self.fixed_wing_stabilizer_diff_limit,
            )
            policy_common = 0.5 * (action[:, 5] + action[:, 6])
            policy_diff = 0.5 * (action[:, 5] - action[:, 6])
            nav_blend = (
                (cross_track.abs() / max(self.max_cross_track * 0.5, 1.0))
                .clamp(0.0, 1.0)
            )
            differential = (
                (1.0 - nav_blend) * policy_diff + nav_blend * roll_trim
            ).clamp(-1.0, 1.0)
            action[fixed_wing, 5] = (
                policy_common[fixed_wing] + differential[fixed_wing]
            ).clamp(-1.0, 1.0)
            action[fixed_wing, 6] = (
                policy_common[fixed_wing] - differential[fixed_wing]
            ).clamp(-1.0, 1.0)

        # Reverse transition still has enough forward airflow for the elevons
        # to provide useful lateral authority.  Reuse the validated fixed-wing
        # sign convention, but blend it in more strongly as cross-track error
        # grows because the rotor re-engagement otherwise weakens the learned
        # surface command and lets the vehicle drift outside the route.
        if self.backtransition_nav_enable and torch.any(backtransition):
            npos, epos, _ = env.model.get_position()
            _, _, heading = env.model.get_posture()
            cross_track = (
                -(npos - self.start_n) * self.route_unit_e_batch
                + (epos - self.start_e) * self.route_unit_n_batch
            )
            heading_error = wrap_PI(self.route_heading_batch - heading)
            roll_command = -(
                self.backtransition_nav_heading_gain * heading_error
                + self.backtransition_nav_cross_track_gain * torch.atan(
                    cross_track / self.backtransition_nav_lookahead
                )
            ).clamp(
                -self.backtransition_nav_roll_limit,
                self.backtransition_nav_roll_limit,
            )
            roll_trim = (
                self.fixed_wing_stabilizer_roll_gain * roll_command
            ).clamp(
                -self.backtransition_nav_diff_limit,
                self.backtransition_nav_diff_limit,
            )
            policy_common = 0.5 * (action[:, 5] + action[:, 6])
            policy_diff = 0.5 * (action[:, 5] - action[:, 6])
            nav_blend = (
                (cross_track.abs() / max(self.max_cross_track * 0.35, 1.0))
                .clamp(0.0, 1.0)
            )
            differential = (
                (1.0 - nav_blend) * policy_diff + nav_blend * roll_trim
            ).clamp(-1.0, 1.0)
            action[backtransition, 5] = (
                policy_common[backtransition] + differential[backtransition]
            ).clamp(-1.0, 1.0)
            action[backtransition, 6] = (
                policy_common[backtransition] - differential[backtransition]
            ).clamp(-1.0, 1.0)

            if self.backtransition_stabilizer_enable:
                roll, pitch, _ = env.model.get_posture()
                p, q, _ = env.model.get_angular_velocity()
                attitude_mag = torch.maximum(roll.abs(), pitch.abs())
                stabilizer_blend = (
                    (attitude_mag - self.backtransition_stabilizer_start)
                    / (
                        self.backtransition_stabilizer_full
                        - self.backtransition_stabilizer_start
                    )
                ).clamp(0.0, 1.0)
                stabilizer_roll = (
                    self.backtransition_stabilizer_roll_gain * roll
                    + self.backtransition_stabilizer_roll_rate_gain * p
                ).clamp(
                    -self.backtransition_stabilizer_diff_limit,
                    self.backtransition_stabilizer_diff_limit,
                )
                stabilizer_pitch = (
                    -self.backtransition_stabilizer_pitch_gain * pitch
                    - self.backtransition_stabilizer_pitch_rate_gain * q
                ).clamp(
                    -self.backtransition_stabilizer_common_limit,
                    self.backtransition_stabilizer_common_limit,
                )
                common = (
                    (1.0 - stabilizer_blend) * policy_common
                    + stabilizer_blend * stabilizer_pitch
                ).clamp(-1.0, 1.0)
                differential = (
                    (1.0 - stabilizer_blend) * differential
                    + stabilizer_blend * stabilizer_roll
                ).clamp(-1.0, 1.0)
                action[backtransition, 5] = (
                    common[backtransition] + differential[backtransition]
                ).clamp(-1.0, 1.0)
                action[backtransition, 6] = (
                    common[backtransition] - differential[backtransition]
                ).clamp(-1.0, 1.0)
                action[backtransition, 7] = (
                    (1.0 - stabilizer_blend[backtransition])
                    * action[backtransition, 7]
                    + stabilizer_blend[backtransition]
                    * stabilizer_pitch[backtransition]
                ).clamp(-1.0, 1.0)

        # During forward transition use a speed-dependent pusher ceiling.  A
        # neutral normalized action maps to roughly half throttle in the
        # actuator model; allowing it unchanged can accelerate past the
        # transition envelope before the policy has learned to unload.
        transition = self.phase == self.TRANSITION
        if torch.any(transition):
            forward_speed = env.model.s[:, 6].clamp_min(0.0)
            speed_fraction = (
                (forward_speed - self.transition_speed)
                / (self.transition_pusher_speed_limit - self.transition_speed)
            ).clamp(0.0, 1.0)
            pusher_cap = self.transition_pusher_max_action + speed_fraction * (
                self.transition_pusher_high_speed_action
                - self.transition_pusher_max_action
            )
            action[transition, 4] = torch.minimum(
                action[transition, 4], pusher_cap[transition]
            )

        fixed_wing = self.phase == self.FIXED_WING
        if torch.any(fixed_wing):
            _, _, altitude = env.model.get_position()
            contact = env.model.get_ground_contact_state()
            recovery_fraction = (
                (altitude - self.min_cruise_altitude)
                / (self.fixed_wing_pusher_recovery_altitude
                   - self.min_cruise_altitude)
            ).clamp(0.0, 1.0)
            recovery_cap = (
                self.fixed_wing_pusher_recovery_action
                + recovery_fraction * (
                    self.fixed_wing_pusher_max_action
                    - self.fixed_wing_pusher_recovery_action
                )
            )
            # A manually forced fixed-wing phase at rest is still a ground
            # state; do not apply the in-flight low-altitude pusher recovery
            # cap before the contact state has cleared.
            recovery_cap = torch.where(
                contact['on_ground'],
                torch.full_like(recovery_cap, self.fixed_wing_pusher_max_action),
                recovery_cap,
            )
            action[fixed_wing, 4] = torch.minimum(
                action[fixed_wing, 4],
                recovery_cap[fixed_wing],
            )

        # Re-engage the lift rotors gradually during reverse transition.  This
        # removes the burden of discovering a four-rotor collective from the
        # very sparse landing signal while retaining learned surface control.
        if torch.any(backtransition):
            npos, epos, altitude = env.model.get_position()
            _, _, vel_up = env.model.get_world_velocity()
            distance = torch.sqrt(
                (npos - self.goal_n) ** 2
                + (epos - self.goal_e) ** 2
            )
            distance_fraction = (
                (distance - self.descent_capture_radius)
                / (self.approach_distance - self.descent_capture_radius)
            ).clamp(0.0, 1.0)
            nominal_altitude = (
                self.landing_hover_altitude
                + distance_fraction * (
                    self.cruise_altitude - self.landing_hover_altitude
                )
            )
            motor_constant = float(env.model.dynamics.motor_constant[0])
            hover_omega = torch.sqrt(
                (
                    env.model.mass_curr * env.model.dynamics.g
                    / (4.0 * max(motor_constant, 1.0e-8))
                ).clamp_min(0.0)
            )
            rotor_scale = env.model.motor_cmd_scaling[0]
            if env.model.action_is_gazebo_control:
                hover_action = hover_omega / rotor_scale
            else:
                hover_action = 2.0 * hover_omega / rotor_scale - 1.0
            altitude_error = nominal_altitude - altitude
            target_sink = -torch.minimum(
                torch.full_like(altitude, self.backtransition_target_sink_speed),
                0.18 * torch.sqrt(
                    (nominal_altitude - self.landing_hover_altitude).clamp_min(0.0)
                ),
            )
            lift_target = (
                hover_action
                + self.backtransition_altitude_gain * altitude_error
                + self.backtransition_velocity_gain * (target_sink - vel_up)
            ).clamp(-1.0, 1.0)
            action[backtransition, 0:4] = lift_target[backtransition].reshape(-1, 1)

            # Ramp an air-brake/parasitic drag coefficient only in the final
            # reverse-transition corridor. This supplies physical braking
            # authority while retaining the strict low-speed capture gate.
            drag_fraction = (
                (self.backtransition_extra_drag_start_distance - distance)
                / (
                    self.backtransition_extra_drag_start_distance
                    - self.descent_capture_radius
                )
            ).clamp(0.0, 1.0)
            # The dynamics object is shared by the whole GPU batch.  Always
            # write a per-environment tensor, including zeros for non-back-
            # transition environments, otherwise one reverse-transition
            # vehicle leaks its air-brake coefficient into every other phase.
            drag_coefficient = torch.where(
                backtransition,
                self.backtransition_extra_drag_coefficient * drag_fraction,
                torch.zeros_like(distance),
            )
            if hasattr(env.model.dynamics, 'extra_drag_coefficient'):
                env.model.dynamics.extra_drag_coefficient = drag_coefficient
        elif hasattr(env.model.dynamics, 'extra_drag_coefficient'):
            env.model.dynamics.extra_drag_coefficient = torch.zeros(
                self.n, device=self.device
            )

        if self.gate_backtransition_pusher:
            npos, epos, _ = env.model.get_position()
            vel_n, vel_e, _ = env.model.get_world_velocity()
            distance = torch.sqrt(
                (npos - self.goal_n) ** 2
                + (epos - self.goal_e) ** 2
            )
            horizontal_speed = torch.sqrt(
                (vel_n * vel_n + vel_e * vel_e).clamp_min(0.0)
            )
            distance_fraction = (
                (distance - self.descent_capture_radius)
                / (self.approach_distance - self.descent_capture_radius)
            ).clamp(0.0, 1.0)
            speed_fraction = (
                horizontal_speed / self.backtransition_speed
            ).clamp(0.0, 1.0)
            # Reverse transition is a braking phase. A full pusher command at
            # the early approach gate would preserve cruise speed and repeatedly hit
            # the high-speed boundary, so only a small residual fraction is
            # permitted and it tapers to zero at the capture radius.
            pusher_fraction = (
                self.backtransition_pusher_max_fraction
                * torch.minimum(distance_fraction, speed_fraction)
            )
            if env.model.action_is_gazebo_control:
                pusher_cap = pusher_fraction
            else:
                # Convert the desired fraction of maximum pusher speed back
                # through GazeboModel's [-1, 1] action mapping.
                max_command_fraction = (
                    env.model.motor_omega_max[4]
                    / env.model.motor_cmd_scaling[4]
                )
                pusher_cap = (
                    2.0 * pusher_fraction * max_command_fraction - 1.0
                )
            # The raw action cap is useful for policy diagnostics.  The model
            # also receives a physical (omega) cap and reapplies it after the
            # action filter, so stale filtered commands cannot bypass this
            # safety limit.
            if hasattr(env.model, 'pusher_omega_cap'):
                inf = torch.full_like(pusher_fraction, float('inf'))
                env.model.pusher_omega_cap = torch.where(
                    backtransition,
                    pusher_fraction * env.model.motor_omega_max[4],
                    inf,
                )
            action[:, 4] = torch.where(
                backtransition,
                torch.minimum(action[:, 4], pusher_cap),
                action[:, 4],
            )
        return action

    def aero_envelope_active(self, env):
        """Return where fixed-wing aerodynamic angles are meaningful."""
        # During early forward transition the lift rotors still carry the
        # aircraft, so a large wing alpha/beta is not itself a loss of control.
        # Apply the hard wing envelope only after transition airspeed is
        # reached.  In wing-borne phases retain the lower threshold so a stall
        # remains terminal, while reverse body flow never counts as useful
        # forward airflow.
        forward_airspeed = env.model.s[:, 6]
        transition_wingborne = (
            (self.phase == self.TRANSITION)
            & (
                forward_airspeed
                >= self.transition_aero_envelope_min_forward_speed
            )
        )
        # Once reverse transition begins, rotor lift is being re-established
        # and the fixed-wing alpha/beta model is no longer a reliable hard
        # safety boundary.  Keep the independent attitude, rate, speed and
        # ground-contact limits active there, but do not terminate on a stale
        # wing angle while the vehicle is reconfiguring.
        phase_elapsed = env.step_count - self.phase_entry_step
        fixed_wing_ready = phase_elapsed >= self.fixed_wing_aero_grace_steps
        wingborne_phase = (
            (self.phase == self.FIXED_WING)
            & fixed_wing_ready
        )
        return transition_wingborne | (
            wingborne_phase
            & (forward_airspeed >= self.aero_envelope_min_forward_speed)
        )

    def _waypoint_distance(self, env):
        npos, epos, altitude = env.model.get_position()
        return torch.sqrt(
            (self.target_npos - npos) ** 2
            + (self.target_epos - epos) ** 2
            + (self.target_altitude - altitude) ** 2
        )

    def step(self, env):
        self.previous_waypoint_distance = self._waypoint_distance(env).detach()
        self.metric_waypoint_distance = self.previous_waypoint_distance
        npos, epos, _ = env.model.get_position()
        self.metric_landing_distance = torch.sqrt(
            (npos - self.goal_n) ** 2 + (epos - self.goal_e) ** 2
        ).detach()
        self.metric_gps_age = env.model.get_gps_state()['age'].detach()
        self.training_success_count += self.mission_success(env).sum()
        self.training_failure_count += env.bad_done.sum()
        if self.rollout_metrics_active and hasattr(env.model, 'action_delta'):
            action_delta = env.model.action_delta
            raw_action_delta = getattr(
                env.model, 'raw_action_delta', action_delta
            )
            slew_limited = getattr(
                env.model,
                'action_slew_limited',
                torch.zeros_like(action_delta, dtype=torch.bool),
            )
            self.rollout_action_delta_abs_sum += action_delta.abs().sum()
            self.rollout_action_delta_sq_sum += (action_delta * action_delta).sum()
            self.rollout_raw_action_delta_abs_sum += raw_action_delta.abs().sum()
            self.rollout_action_slew_limited_sum += slew_limited.float().sum()
            self.rollout_action_metric_samples += action_delta.numel()

    def get_training_metrics(self):
        metrics = {
            'mission/waypoint_distance_mean': self.metric_waypoint_distance.mean(),
            'mission/landing_distance_mean': self.metric_landing_distance.mean(),
            'mission/gps_age_mean': self.metric_gps_age.mean(),
            'mission/success_count': self.training_success_count,
            'mission/failure_count': self.training_failure_count,
            'mission/reward_constraint_penalty_mean': self.constraint_penalty.mean(),
        }
        for phase_id, phase_name in enumerate(self.PHASE_NAMES):
            metrics[f'mission/phase_{phase_name}_fraction'] = (
                self.phase == phase_id
            ).float().mean()
            metrics[f'mission/start_{phase_name}_fraction'] = (
                self.start_phase == phase_id
            ).float().mean()
            if phase_id > 0:
                metrics[f'mission/reach_{phase_name}_count'] = (
                    self.rollout_phase_reach_count[phase_id]
                )
                metrics[f'mission/reach_{phase_name}_rate'] = (
                    self.rollout_phase_reach_count[phase_id]
                    / self.rollout_episode_opportunities.clamp_min(1.0)
                )
        metrics['mission/rollout_episode_opportunities'] = (
            self.rollout_episode_opportunities
        )
        # Rotor-climb diagnostics make sim2sim attitude failures visible in
        # TensorBoard without copying a per-environment trace to the host.
        if self._env_ref is not None:
            roll, pitch, _ = self._env_ref.model.get_posture()
            p, q, _ = self._env_ref.model.get_angular_velocity()
            rotor_mask = self.phase == self.ROTOR_CLIMB
            metrics['mission/rotor_climb_roll_abs_mean'] = torch.where(
                rotor_mask, roll.abs(), torch.zeros_like(roll)
            ).sum() / rotor_mask.float().sum().clamp_min(1.0)
            metrics['mission/rotor_climb_pitch_abs_mean'] = torch.where(
                rotor_mask, pitch.abs(), torch.zeros_like(pitch)
            ).sum() / rotor_mask.float().sum().clamp_min(1.0)
            metrics['mission/rotor_climb_p_abs_mean'] = torch.where(
                rotor_mask, p.abs(), torch.zeros_like(p)
            ).sum() / rotor_mask.float().sum().clamp_min(1.0)
            metrics['mission/rotor_climb_q_abs_mean'] = torch.where(
                rotor_mask, q.abs(), torch.zeros_like(q)
            ).sum() / rotor_mask.float().sum().clamp_min(1.0)
            metrics['mission/rotor_climb_stabilizer_abs_mean'] = torch.where(
                rotor_mask,
                self.rotor_climb_stabilizer_delta.abs().mean(dim=1),
                torch.zeros_like(roll),
            ).sum() / rotor_mask.float().sum().clamp_min(1.0)
            metrics['mission/rotor_climb_command_abs_mean'] = torch.where(
                rotor_mask,
                self.rotor_climb_command.abs().mean(dim=1),
                torch.zeros_like(roll),
            ).sum() / rotor_mask.float().sum().clamp_min(1.0)
        if self.rollout_action_metric_samples > 0:
            samples = self.rollout_action_metric_samples
            metrics['mission/action_delta_abs_mean'] = (
                self.rollout_action_delta_abs_sum / samples
            )
            metrics['mission/action_delta_rms'] = torch.sqrt(
                self.rollout_action_delta_sq_sum / samples
            )
            metrics['mission/raw_action_delta_abs_mean'] = (
                self.rollout_raw_action_delta_abs_sum / samples
            )
            metrics['mission/action_slew_limited_fraction'] = (
                self.rollout_action_slew_limited_sum / samples
            )
        if self.terminal_mode == 'hover':
            metrics['mission/hover_stable_fraction'] = (
                self.hover_stable_candidate(self._env_ref).float().mean()
            )
            metrics['mission/hover_hold_progress_mean'] = (
                self.hover_stable_count.to(torch.float32)
                / float(self.hover_hold_steps)
            ).clamp(0.0, 1.0).mean()
        return metrics

    def landing_success(self, env):
        npos, epos, _ = env.model.get_position()
        roll, pitch, _ = env.model.get_posture()
        speed = env.model.get_TAS()
        climb_rate = env.model.get_climb_rate()
        contact = env.model.get_ground_contact_state()
        distance = torch.sqrt(
            (npos - self.goal_n) ** 2 + (epos - self.goal_e) ** 2
        )
        radius = float(getattr(self.config, 'mission_landing_radius', 3.0))
        max_speed = float(getattr(self.config, 'mission_landing_max_speed', 1.0))
        max_sink = float(getattr(self.config, 'mission_landing_max_sink_speed', 0.6))
        max_tilt = math.radians(float(
            getattr(self.config, 'mission_landing_max_tilt_deg', 12.0)
        ))
        settle_steps = int(getattr(
            self.config, 'mission_landing_settle_steps', 15
        ))
        return (
            (self.phase == self.VERTICAL_LANDING)
            & contact['on_ground']
            & (contact['contact_duration_steps'] >= settle_steps)
            & (distance <= radius)
            & (speed <= max_speed)
            & (climb_rate.abs() <= max_sink)
            & (roll.abs() <= max_tilt)
            & (pitch.abs() <= max_tilt)
        )

    def hover_success(self, env):
        return (
            self.hover_stable_candidate(env)
            & (self.hover_stable_count >= self.hover_hold_steps)
        )

    def mission_success(self, env):
        if self.terminal_mode == 'hover':
            return self.hover_success(env)
        return self.landing_success(env)

    def _build_obs(self, env, clean=False):
        """Build the policy observation from deterministic simulator readings."""
        if clean:
            gps_position = env.model.s[:, 0:3]
            gps_age = torch.zeros(self.n, device=self.device)
            gps_valid = torch.ones(self.n, dtype=torch.bool, device=self.device)
        else:
            gps = env.model.get_gps_state()
            gps_position = gps['position']
            gps_age = gps['age']
            gps_valid = gps['valid']

        npos = gps_position[:, 0]
        epos = gps_position[:, 1]
        altitude = gps_position[:, 2]
        roll, pitch, heading = env.model.get_posture()
        speed = env.model.get_TAS()
        vel_n, vel_e, vel_up = env.model.get_world_velocity()
        p, q, r = env.model.get_angular_velocity()
        sa, ca, sb, cb = env.model.get_aero_sincos()

        if self.sensor_noise_enabled and not clean:
            roll = wrap_PI(roll + torch.randn_like(roll) * self.sensor_attitude_std)
            pitch = wrap_PI(pitch + torch.randn_like(pitch) * self.sensor_attitude_std)
            heading = wrap_PI(
                heading + torch.randn_like(heading) * self.sensor_attitude_std
            )
            vel_n = vel_n + torch.randn_like(vel_n) * self.sensor_velocity_std
            vel_e = vel_e + torch.randn_like(vel_e) * self.sensor_velocity_std
            vel_up = vel_up + torch.randn_like(vel_up) * self.sensor_velocity_std
            p = p + torch.randn_like(p) * self.sensor_rate_std
            q = q + torch.randn_like(q) * self.sensor_rate_std
            r = r + torch.randn_like(r) * self.sensor_rate_std
            # Keep TAS tied to the model's air-relative velocity.  Sensor
            # noise is applied as an independent pitot/TAS measurement error;
            # it must not silently change to a ground-velocity norm.
            speed = torch.clamp_min(
                speed + torch.randn_like(speed) * self.sensor_airspeed_std,
                0.0,
            )

        landing_dn = self.goal_n - npos
        landing_de = self.goal_e - epos
        terminal_altitude = (
            self.landing_hover_altitude
            if self.terminal_mode == 'hover'
            else self.landing_altitude
        )
        landing_dalt = terminal_altitude - altitude
        waypoint_dn = self.target_npos - npos
        waypoint_de = self.target_epos - epos
        waypoint_dalt = self.target_altitude - altitude
        along_track = (
            (npos - self.start_n) * self.route_unit_n_batch
            + (epos - self.start_e) * self.route_unit_e_batch
        )
        cross_track = (
            -(npos - self.start_n) * self.route_unit_e_batch
            + (epos - self.start_e) * self.route_unit_n_batch
        )
        heading_error = wrap_PI(self.target_heading - heading)
        phase_one_hot = torch.nn.functional.one_hot(
            self.phase, num_classes=self.PHASE_COUNT
        ).to(dtype=env.model.s.dtype)

        motor = torch.stack(env.model.get_motor_omega(), dim=1)
        motor_scale = env.model.motor_omega_max.reshape(1, 5)
        surfaces = env.model.u[:, 5:8] / env.model.surface_limit.reshape(1, 3)
        contact = env.model.get_ground_contact_state()
        time_remaining = (
            1.0 - env.step_count.to(env.model.s.dtype) / max(self.max_steps, 1)
        ).clamp(0.0, 1.0)

        obs = torch.hstack((
            (landing_dn / self.distance_norm).reshape(-1, 1),
            (landing_de / self.distance_norm).reshape(-1, 1),
            (landing_dalt / self.altitude_norm).reshape(-1, 1),
            (waypoint_dn / self.distance_norm).reshape(-1, 1),
            (waypoint_de / self.distance_norm).reshape(-1, 1),
            (waypoint_dalt / self.altitude_norm).reshape(-1, 1),
            (along_track / self.distance_norm).reshape(-1, 1),
            (cross_track / self.distance_norm).reshape(-1, 1),
            torch.sin(heading_error).reshape(-1, 1),
            torch.cos(heading_error).reshape(-1, 1),
            phase_one_hot,
            (gps_age / self.gps_age_norm).reshape(-1, 1),
            gps_valid.to(env.model.s.dtype).reshape(-1, 1),
            torch.sin(roll).reshape(-1, 1),
            torch.cos(roll).reshape(-1, 1),
            torch.sin(pitch).reshape(-1, 1),
            torch.cos(pitch).reshape(-1, 1),
            (speed / self.speed_norm).reshape(-1, 1),
            (vel_n / self.speed_norm).reshape(-1, 1),
            (vel_e / self.speed_norm).reshape(-1, 1),
            (vel_up / self.vertical_speed_norm).reshape(-1, 1),
            (self.target_speed / self.speed_norm).reshape(-1, 1),
            (p / self.rate_norm).reshape(-1, 1),
            (q / self.rate_norm).reshape(-1, 1),
            (r / self.rate_norm).reshape(-1, 1),
            sa.reshape(-1, 1),
            ca.reshape(-1, 1),
            sb.reshape(-1, 1),
            cb.reshape(-1, 1),
            motor / motor_scale,
            surfaces,
            contact['on_ground'].to(env.model.s.dtype).reshape(-1, 1),
            (contact['contact_count'].to(env.model.s.dtype) / 4.0).reshape(-1, 1),
            time_remaining.reshape(-1, 1),
        ))
        if obs.shape[1] != self.OBSERVATION_SIZE:
            raise RuntimeError(
                f'VTOL mission observation has {obs.shape[1]} values, '
                f'expected {self.OBSERVATION_SIZE}'
            )
        return obs

    def get_obs(self, env):
        return self._build_obs(env, clean=False)

    def get_clean_obs(self, env):
        return self._build_obs(env, clean=True)

    def get_critic_obs(self, env):
        """Privileged simulator state used only by the asymmetric critic."""
        clean_obs = self.get_clean_obs(env)
        state = env.model.s.clone()
        state[:, 0] = (state[:, 0] - self.start_n) / self.distance_norm
        state[:, 1] = (state[:, 1] - self.start_e) / self.distance_norm
        state[:, 2] /= self.altitude_norm
        state[:, 3:6] /= math.pi
        state[:, 6:9] /= self.speed_norm
        state[:, 9:12] /= self.rate_norm

        contact = env.model.get_ground_contact_state()
        mass_weight = (env.model.mass_curr * env.model.dynamics.g).clamp_min(1e-6)
        max_penetration = max(float(getattr(
            self.config, 'ground_crash_max_penetration', 0.05
        )), 1e-6)
        max_touchdown = max(float(getattr(
            self.config, 'ground_crash_max_touchdown_speed', 1.5
        )), 1e-6)
        settle_steps = max(int(getattr(
            self.config, 'mission_landing_settle_steps', 15
        )), 1)
        acceleration_limit = max(float(getattr(
            self.config, 'acceleration_limit', 35.0
        )), 1e-6)
        body_acceleration = (
            env.model.s[:, 6:9] - env.model.recent_s[:, 6:9]
        ) / max(self.dt, 1e-6)
        acceleration_ratio = torch.linalg.vector_norm(
            body_acceleration, dim=1
        ) / acceleration_limit
        dwell = self.phase_min_steps[self.phase].clamp_min(1)
        phase_elapsed = env.step_count - self.phase_entry_step

        privileged = torch.hstack((
            clean_obs,
            state,
            (contact['normal_force'] / mass_weight).clamp(0.0, 50.0).reshape(-1, 1),
            (contact['max_penetration'] / max_penetration).clamp(0.0, 10.0).reshape(-1, 1),
            (contact['touchdown_vertical_speed'] / max_touchdown).clamp(0.0, 10.0).reshape(-1, 1),
            (contact['last_touchdown_vertical_speed'] / max_touchdown).clamp(0.0, 10.0).reshape(-1, 1),
            (contact['contact_duration_steps'].to(state.dtype) / settle_steps).clamp(0.0, 10.0).reshape(-1, 1),
            acceleration_ratio.clamp(0.0, 10.0).reshape(-1, 1),
            (phase_elapsed.to(state.dtype) / dwell).clamp(0.0, 2.0).reshape(-1, 1),
        ))
        if privileged.shape[1] != self.CRITIC_OBSERVATION_SIZE:
            raise RuntimeError(
                f'VTOL critic observation has {privileged.shape[1]} values, '
                f'expected {self.CRITIC_OBSERVATION_SIZE}'
            )
        return torch.nan_to_num(privileged, nan=0.0, posinf=10.0, neginf=-10.0)
