import torch


class StandardVTOLGroundContact:
    """Vectorized flat-ground contact for the Gazebo Classic standard_vtol.

    The simulator state uses north/east/altitude coordinates, with altitude
    positive up, while body axes are forward/right/down.  Contact points are
    expressed in body axes.  The default points are the four lower corners of
    the collision box in PX4's ``standard_vtol.sdf``:

        collision pose: (0, 0, -0.07) in Gazebo's z-up body frame
        collision size: (0.55, 2.144, 0.05)

    Therefore the lower face is at body z=+0.095 in this simulator's
    forward/right/down convention.
    """

    DEFAULT_CONTACT_POINTS = (
        (0.275, 1.072, 0.095),
        (0.275, -1.072, 0.095),
        (-0.275, 1.072, 0.095),
        (-0.275, -1.072, 0.095),
    )

    def __init__(self, config=None):
        self.enabled = bool(getattr(config, 'ground_contact_enable', False))
        self.ground_height = float(getattr(config, 'ground_height', 0.0))
        self.stiffness = float(getattr(config, 'ground_contact_stiffness', 10000.0))
        self.damping = float(getattr(config, 'ground_contact_damping', 150.0))
        self.friction = float(getattr(config, 'ground_contact_friction', 0.7))
        self.slip_speed = max(
            float(getattr(config, 'ground_contact_slip_speed', 0.05)), 1e-6
        )
        self.max_normal_force = float(
            getattr(config, 'ground_contact_max_normal_force', 5000.0)
        )

        points = getattr(config, 'ground_contact_points', self.DEFAULT_CONTACT_POINTS)
        if len(points) < 1 or any(len(point) != 3 for point in points):
            raise ValueError('ground_contact_points must contain at least one [x, y, z] point')
        self.contact_points = tuple(tuple(float(value) for value in point) for point in points)

        if self.stiffness <= 0.0:
            raise ValueError('ground_contact_stiffness must be positive')
        if self.damping < 0.0:
            raise ValueError('ground_contact_damping must be non-negative')
        if self.friction < 0.0:
            raise ValueError('ground_contact_friction must be non-negative')

    @staticmethod
    def body_to_world_matrix(state):
        """Return the body-FRD to north/east/altitude-up transform."""
        phi = state[:, 3]
        theta = state[:, 4]
        psi = state[:, 5]

        sphi = torch.sin(phi)
        cphi = torch.cos(phi)
        st = torch.sin(theta)
        ct = torch.cos(theta)
        spsi = torch.sin(psi)
        cpsi = torch.cos(psi)

        row_north = torch.stack((
            ct * cpsi,
            sphi * cpsi * st - cphi * spsi,
            cphi * cpsi * st + sphi * spsi,
        ), dim=1)
        row_east = torch.stack((
            ct * spsi,
            sphi * spsi * st + cphi * cpsi,
            cphi * spsi * st - sphi * cpsi,
        ), dim=1)
        row_up = torch.stack((
            st,
            -sphi * ct,
            -cphi * ct,
        ), dim=1)
        return torch.stack((row_north, row_east, row_up), dim=1)

    def _points_for_state(self, state):
        return state.new_tensor(self.contact_points).reshape(1, -1, 3).expand(
            state.shape[0], -1, -1
        )

    def point_kinematics(self, state):
        """Return point positions and velocities in world coordinates."""
        rotation = self.body_to_world_matrix(state)
        points_body = self._points_for_state(state)
        offsets_world = torch.einsum('nij,nkj->nki', rotation, points_body)
        positions_world = offsets_world.clone()
        positions_world[:, :, 0] += state[:, 0].reshape(-1, 1)
        positions_world[:, :, 1] += state[:, 1].reshape(-1, 1)
        positions_world[:, :, 2] += state[:, 2].reshape(-1, 1)

        body_velocity = state[:, 6:9].reshape(-1, 1, 3)
        angular_velocity = state[:, 9:12].reshape(-1, 1, 3).expand_as(points_body)
        point_velocity_body = body_velocity + torch.cross(
            angular_velocity, points_body, dim=2
        )
        point_velocity_world = torch.einsum(
            'nij,nkj->nki', rotation, point_velocity_body
        )
        return positions_world, point_velocity_world, rotation, points_body

    def compute(self, state):
        """Compute aggregate body force/moment and per-environment diagnostics."""
        if not self.enabled:
            batch = state.shape[0]
            point_count = self.number_of_points
            zeros = torch.zeros(batch, device=state.device, dtype=state.dtype)
            zero_points = torch.zeros(
                (batch, point_count), device=state.device, dtype=state.dtype
            )
            diagnostics = {
                'on_ground': torch.zeros(batch, dtype=torch.bool, device=state.device),
                'contact_count': torch.zeros(batch, dtype=torch.long, device=state.device),
                'max_penetration': zeros.clone(),
                'min_clearance': zeros.clone(),
                'total_normal_force': zeros.clone(),
                'max_downward_speed': zeros.clone(),
                'point_penetration': zero_points,
                'point_normal_force': zero_points.clone(),
            }
            return (
                torch.zeros((batch, 3), device=state.device, dtype=state.dtype),
                torch.zeros((batch, 3), device=state.device, dtype=state.dtype),
                diagnostics,
            )

        positions, velocities, rotation, points_body = self.point_kinematics(state)
        penetration = (self.ground_height - positions[:, :, 2]).clamp_min(0.0)
        geometric_contact = penetration > 0.0

        normal_force = self.stiffness * penetration - self.damping * velocities[:, :, 2]
        normal_force = normal_force.clamp_min(0.0) * geometric_contact.to(state.dtype)
        if self.max_normal_force > 0.0:
            normal_force = normal_force.clamp_max(self.max_normal_force)

        tangent_velocity = velocities[:, :, 0:2]
        tangent_speed = torch.linalg.norm(tangent_velocity, dim=2)
        tangent_direction = tangent_velocity / tangent_speed.unsqueeze(-1).clamp_min(1e-8)
        friction_magnitude = (
            self.friction
            * normal_force
            * torch.tanh(tangent_speed / self.slip_speed)
        )
        tangent_force = -friction_magnitude.unsqueeze(-1) * tangent_direction

        force_world = torch.cat((tangent_force, normal_force.unsqueeze(-1)), dim=2)
        force_body_points = torch.einsum(
            'nji,nkj->nki', rotation, force_world
        )
        moment_body_points = torch.cross(points_body, force_body_points, dim=2)
        force_body = force_body_points.sum(dim=1)
        moment_body = moment_body_points.sum(dim=1)

        force_contact = normal_force > 1e-6
        diagnostics = {
            'on_ground': torch.any(force_contact, dim=1),
            'contact_count': force_contact.sum(dim=1),
            'max_penetration': penetration.max(dim=1).values,
            'min_clearance': (positions[:, :, 2] - self.ground_height).min(dim=1).values,
            'total_normal_force': normal_force.sum(dim=1),
            'max_downward_speed': torch.where(
                geometric_contact,
                (-velocities[:, :, 2]).clamp_min(0.0),
                torch.zeros_like(normal_force),
            ).max(dim=1).values,
            'point_penetration': penetration,
            'point_normal_force': normal_force,
        }

        return force_body, moment_body, diagnostics

    def resting_cg_altitude(self, state, clearance=0.0):
        """Place the lowest collision-box point at ground plus ``clearance``."""
        rotation = self.body_to_world_matrix(state)
        offsets_world = torch.einsum(
            'nij,nkj->nki', rotation, self._points_for_state(state)
        )
        lowest_offset = offsets_world[:, :, 2].min(dim=1).values
        clearance_t = torch.as_tensor(clearance, device=state.device, dtype=state.dtype)
        return self.ground_height + clearance_t - lowest_offset

    @property
    def number_of_points(self):
        return len(self.contact_points)
