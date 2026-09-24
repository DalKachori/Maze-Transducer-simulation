"""
environment.py
---------------
The main entry point for external scripts. A maze-solving algorithm only
ever needs to talk to an `Environment` instance:

    from environment import Environment

    env = Environment(rows=10, cols=10, cell_size=0.3, seed=42)
    env.add_sensor(angle_deg=0)     # front
    env.add_sensor(angle_deg=90)    # left
    env.add_sensor(angle_deg=-90)   # right

    while not env.at_goal():
        readings = env.sense()          # {'s0': SensorReading, ...}
        # ... your algorithm decides what to do ...
        env.move(forward=0.02, dtheta_deg=0)

The algorithm only ever sees `sense()` output - it has no direct access
to `env.maze`, so it's solving the maze exactly like a real robot would
(you can of course still call `env.maze` yourself for logging/plots).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from maze_gen import Maze, generate_maze
from sensors import SensorArray, SensorReading

Pose = Tuple[float, float, float]  # x, y, theta(rad)


@dataclass
class MoveResult:
    success: bool                 # False if the move was blocked by a wall
    pose: Pose                    # resulting pose (unchanged if blocked)
    collided: bool = False


class Environment:
    def __init__(self, rows: int = 10, cols: int = 10, cell_size: float = 0.3,
                 seed: Optional[int] = None, loop_prob: float = 0.0,
                 temperature_c: float = 20.0, humidity_percent: float = 50.0,
                 collision_step: float = 0.005):
        """
        rows, cols, cell_size, seed, loop_prob : forwarded to generate_maze().
        temperature_c, humidity_percent        : ambient conditions that feed
                                                  the sensor physics (true
                                                  speed of sound). Change
                                                  these mid-run to see how a
                                                  fixed firmware assumption
                                                  drifts from reality.
        collision_step                         : max sub-step length (m) used
                                                  when sweeping a move for
                                                  wall collisions, so fast
                                                  moves can't tunnel through
                                                  thin walls.
        """
        self._rows, self._cols, self._cell_size = rows, cols, cell_size
        self._seed = seed
        self._loop_prob = loop_prob
        self.collision_step = collision_step

        self.temperature_c = temperature_c
        self.humidity_percent = humidity_percent

        self.maze: Maze = generate_maze(rows, cols, cell_size, seed, loop_prob)
        self.sensors = SensorArray(seed=seed)

        self.time = 0.0
        self.pose: Pose = (*self.maze.start_position(), 0.0)
        self._trail = [self.pose[:2]]

    # -- setup ---------------------------------------------------------

    def add_sensor(self, angle_deg: float, offset: Tuple[float, float] = (0.0, 0.0),
                   name: Optional[str] = None, **physics_overrides) -> int:
        """Mount one more ultrasonic sensor.

        angle_deg : direction the sensor points, relative to the robot's
                    forward heading (0 = forward, 90 = left, -90 = right,
                    180 = backward). Add as many sensors, at whatever
                    angles, as you like.
        offset    : (forward, left) mounting position offset in meters,
                    relative to the robot's reference point. Since the
                    robot currently has zero size this defaults to (0,0);
                    it's supported for when you give the robot real
                    dimensions later.
        physics_overrides : any field of SensorParams, e.g.
                    max_range=2.0, beam_half_angle_deg=20,
                    specular_angle_threshold_deg=35,
                    firmware_assumed_temp_c=20, reflectivity=0.6, ...
        """
        return self.sensors.add(angle_deg, offset, name, **physics_overrides)

    def remove_sensor(self, sensor_id: int) -> None:
        self.sensors.remove(sensor_id)

    def clear_sensors(self) -> None:
        self.sensors.clear()

    # -- lifecycle -------------------------------------------------------

    def reset(self, seed: Optional[int] = None, regenerate_maze: bool = True) -> None:
        """Regenerate the maze (new seed if given) and reset the robot to
        the start cell. Sensors are kept as configured."""
        if regenerate_maze:
            self._seed = seed if seed is not None else self._seed
            self.maze = generate_maze(self._rows, self._cols, self._cell_size,
                                       self._seed, self._loop_prob)
        self.time = 0.0
        self.pose = (*self.maze.start_position(), 0.0)
        self._trail = [self.pose[:2]]

    def set_pose(self, x: float, y: float, theta_rad: float) -> None:
        self.pose = (x, y, theta_rad)
        self._trail.append((x, y))

    def get_pose(self) -> Pose:
        return self.pose

    # -- motion (robot treated as a point / zero size) --------------------

    def move(self, forward: float, dtheta_deg: float = 0.0) -> MoveResult:
        """Rotate by dtheta_deg then move `forward` meters (negative = back).

        The path is swept in small steps and stopped at the first wall it
        would cross, so the caller finds out immediately if it drove into
        a wall (`success=False`) rather than silently teleporting through
        it or past it.
        """
        x, y, theta = self.pose
        theta += math.radians(dtheta_deg)

        if forward == 0.0:
            self.pose = (x, y, theta)
            return MoveResult(True, self.pose)

        steps = max(1, int(math.ceil(abs(forward) / self.collision_step)))
        step_len = forward / steps
        cx, cy = x, y
        collided = False
        for _ in range(steps):
            nx = cx + step_len * math.cos(theta)
            ny = cy + step_len * math.sin(theta)
            if self.maze.path_blocked((cx, cy), (nx, ny)) or not self.maze.in_bounds(nx, ny):
                collided = True
                break
            cx, cy = nx, ny

        self.pose = (cx, cy, theta)
        self._trail.append((cx, cy))
        return MoveResult(success=not collided, pose=self.pose, collided=collided)

    def teleport(self, x: float, y: float, theta_rad: float = 0.0) -> None:
        """Directly set the pose with no collision check (for tests/resets)."""
        self.set_pose(x, y, theta_rad)

    # -- sensing -----------------------------------------------------------

    def sense(self) -> Dict[str, SensorReading]:
        """Fire every mounted sensor once and return {name: SensorReading}.

        This is the ONLY window a maze-solving algorithm needs into the
        world - it never needs `self.maze` directly.
        """
        x, y, theta = self.pose
        return self.sensors.sense_all(x, y, theta, self.maze,
                                       self.temperature_c, self.humidity_percent)

    # -- goal / status -------------------------------------------------------

    def at_goal(self, tolerance: Optional[float] = None) -> bool:
        gx, gy = self.maze.goal_position()
        x, y, _ = self.pose
        tol = tolerance if tolerance is not None else self.maze.cell_size * 0.4
        return math.hypot(x - gx, y - gy) <= tol

    def trail(self):
        """Positions visited so far (for plotting a path)."""
        return list(self._trail)

    # -- introspection (fine to use for logging / your own visualizations,
    #    just don't feed it into your algorithm's decision-making if you
    #    want a fair "robot doesn't see the whole maze" test) --------------

    def ground_truth_maze(self) -> Maze:
        return self.maze
