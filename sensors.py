"""
sensors.py
----------
Amplitude-based model of a hobby ultrasonic transducer (HC-SR04 class).

The idea
--------
The sensor fires a fan of rays across its beam. Every ray carries an
AMPLITUDE. It starts at the transmitter's directivity for that direction
(strong on-axis, weak at the edge of the beam) and is reduced by:

    * spherical spreading  (1 / total path length)
    * air absorption       (exp(-alpha * path))
    * each wall bounce     (reflectivity, and the fraction scattered
                            instead of mirrored)

Rays bounce off walls like mirrors (up to `max_bounces`). At every bounce
point the wall also scatters some sound back toward the sensor - a lot if
that direction is close to the mirror direction (glassy wall), a little
if not (diffuse part). Each of these returns is an ECHO: it has a path
length (-> arrival time) and an amplitude, which includes the receiver's
own directivity at the angle the echo arrives from.

The receiver is a plain threshold detector: it reports the time of the
FIRST echo whose amplitude crosses the threshold. The threshold is set by
the rated `max_range`: a perfectly reflective flat wall facing the sensor
at max_range is *just* detectable. That's how such ranges are rated.

Nothing below is special-cased. These all fall out of the model:
    * one stray weak ray cannot trigger a reading
    * walls hit at a shallow angle mirror the sound away -> no echo
    * corner double-bounces come back (as long, strong-ish echoes)
    * poorly reflective walls shorten the usable range
    * the beam edge matters much less than the beam centre

The measured time then goes through the (real) timing chain: firmware
speed-of-sound assumption -> timing jitter -> quantization.

Not modelled: 3D effects (the beam is a flat fan in the maze plane), and
temporal echo overlap (each echo is judged on its own amplitude).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from maze_gen import Maze

Point = Tuple[float, float]


def speed_of_sound(temp_c: float, humidity_percent: float = 50.0) -> float:
    """Approximate speed of sound in air (m/s)."""
    return 331.3 + 0.606 * temp_c + 0.0124 * humidity_percent


def _directivity(angle_rad: float, half_rad: float) -> float:
    """One-way pressure gain vs off-axis angle. 1.0 on-axis, 0.5 (-6 dB)
    at `half_rad`, Gaussian roll-off - a good fit for a small piston."""
    return math.exp(-math.log(2.0) * (angle_rad / half_rad) ** 2)


@dataclass
class SensorParams:
    max_range: float = 4.0                # m. Sets the detection threshold (see module doc)
    min_range: float = 0.02               # m. Blind zone (transducer ringdown)
    beam_half_angle_deg: float = 15.0     # off-axis angle where pressure is halved (-6 dB)
    beam_samples: int = 15                # rays across the beam (spans +/-1.5 half-angles)
    reflectivity: float = 1.0             # 0..1 wall amplitude reflection coefficient
    specular_width_deg: float = 8.0       # how tightly a wall mirrors (small = glassy)
    diffuse_fraction: float = 0.1         # part of the sound scattered in all directions
    max_bounces: int = 3                  # wall bounces followed per ray
    air_absorption: float = 0.15          # nepers/m (~1.3 dB/m at 40 kHz)
    amplitude_noise: float = 0.05         # relative receiver noise on echo amplitude
    distance_resolution: float = 0.003    # m, timer quantization
    noise_std_base: float = 0.0015        # m, timing jitter floor
    noise_std_per_meter: float = 0.008    # m of extra std per m of range
    firmware_assumed_temp_c: float = 20.0  # speed of sound baked into the firmware


@dataclass
class SensorMount:
    angle_deg: float                       # mounting angle, 0 = robot forward, CCW+
    offset: Point = (0.0, 0.0)             # (forward, left) offset in robot frame
    name: str = ""
    params: SensorParams = field(default_factory=SensorParams)


@dataclass
class Echo:
    """One possible return: sensor -> bounces -> scatter point -> sensor."""
    path: float                # total round-trip path, m
    level: float               # amplitude / detection threshold (>= 1 is detectable)
    bounces: int               # wall hits before returning (1 = direct echo)
    ray: int                   # index of the ray that produced it
    points: List[Point]        # polyline: sensor, wall hits..., back to sensor
    detected: bool = False     # True for the echo the receiver actually latched


@dataclass
class RayTrace:
    angle: float                               # world-frame launch angle, rad
    points: List[Point]                        # sensor, then each wall hit (or far end if escaped)
    escaped: bool = False                      # left the range without hitting anything
    best_level: float = 0.0                    # level of this ray's earliest detectable echo
    best_bounces: int = 0                      # (or its strongest, if none is detectable)


@dataclass
class SensorReading:
    sensor: str
    distance: Optional[float]        # reported distance in m, or None (no echo)
    true_distance: Optional[float]   # geometric distance of the echo that triggered it
    timeout: bool = False            # no echo above threshold
    blind_zone: bool = False         # echo inside min_range - reading unreliable
    origin: Point = (0.0, 0.0)
    beam_angle: float = 0.0          # world-frame beam axis, rad
    rays: List[RayTrace] = field(default_factory=list)
    echoes: List[Echo] = field(default_factory=list)


class UltrasonicSensor:
    def __init__(self, mount: SensorMount, sensor_id: int, rng: Optional[random.Random] = None):
        self.mount = mount
        self.id = sensor_id
        self.rng = rng or random.Random()
        p = mount.params
        two_r = 2.0 * p.max_range
        # Amplitude of an ideal flat wall facing the sensor at max_range.
        self.threshold = math.exp(-p.air_absorption * two_r) / two_r

    @property
    def name(self) -> str:
        return self.mount.name or f"s{self.id}"

    def _world_origin(self, robot_x: float, robot_y: float, robot_theta: float) -> Point:
        fx, fy = self.mount.offset
        ox = robot_x + fx * math.cos(robot_theta) - fy * math.sin(robot_theta)
        oy = robot_y + fx * math.sin(robot_theta) + fy * math.cos(robot_theta)
        return ox, oy

    # -- acoustics ------------------------------------------------------

    def _cast(self, ox: float, oy: float, beam_angle: float, maze: Maze
              ) -> Tuple[List[RayTrace], List[Echo]]:
        p = self.mount.params
        thr = self.threshold
        half = math.radians(p.beam_half_angle_deg)
        fov = 1.5 * half
        n = max(1, p.beam_samples)
        rho = p.reflectivity
        kd = p.diffuse_fraction
        ks = 1.0 - kd
        sigma = math.radians(p.specular_width_deg)
        absorb = p.air_absorption
        two_r = 2.0 * p.max_range

        traces: List[RayTrace] = []
        echoes: List[Echo] = []

        for i in range(n):
            off = 0.0 if n == 1 else (2.0 * i / (n - 1) - 1.0) * fov
            ang = beam_angle + off
            dx, dy = math.cos(ang), math.sin(ang)
            gain = _directivity(off, half)          # amplitude carried by this ray
            px, py, path = ox, oy, 0.0
            pts: List[Point] = [(ox, oy)]
            trace = RayTrace(ang, pts)
            ray_echoes: List[Echo] = []

            for k in range(1, p.max_bounces + 1):
                hit = maze.raycast(px, py, dx, dy, two_r - path)
                if hit is None:
                    length = min(two_r - path, p.max_range)
                    pts.append((px + dx * length, py + dy * length))
                    trace.escaped = True
                    break
                t, (nx, ny) = hit
                path += t
                px, py = px + dx * t, py + dy * t
                pts.append((px, py))
                cos_i = -(dx * nx + dy * ny)                      # > 0
                rdx, rdy = dx + 2 * cos_i * nx, dy + 2 * cos_i * ny  # mirror direction

                # Sound scattered from this point straight back to the sensor
                bx, by = ox - px, oy - py
                back = math.hypot(bx, by)
                total = path + back
                if back > 1e-9 and total <= two_r:
                    ux, uy = bx / back, by / back
                    cos_o = ux * nx + uy * ny
                    if cos_o > 0 and (k == 1 or maze.visible(px, py, ox, oy)):
                        alpha = math.acos(max(-1.0, min(1.0, rdx * ux + rdy * uy)))
                        scatter = rho * (ks * math.exp(-(alpha / sigma) ** 2)
                                         + kd * cos_i * cos_o)
                        arrive = math.atan2(-uy, -ux) - beam_angle
                        arrive = (arrive + math.pi) % (2 * math.pi) - math.pi
                        amp = (gain * scatter * _directivity(arrive, half)
                               * math.exp(-absorb * total) / total)
                        if amp / thr > 0.01:
                            ray_echoes.append(Echo(total, amp / thr, k, i,
                                                   pts + [(ox, oy)]))

                # Continue along the mirror direction with what's left
                gain *= rho * ks
                dx, dy = rdx, rdy
                if gain * math.exp(-absorb * path) / path < 0.8 * thr:
                    break       # can never become detectable

            if ray_echoes:
                strong = [e for e in ray_echoes if e.level >= 1.0]
                pick = (min(strong, key=lambda e: e.path) if strong
                        else max(ray_echoes, key=lambda e: e.level))
                trace.best_level, trace.best_bounces = pick.level, pick.bounces
            traces.append(trace)
            echoes.extend(ray_echoes)

        return traces, echoes

    # -- measurement ------------------------------------------------------

    def sense(self, robot_x: float, robot_y: float, robot_theta: float, maze: Maze,
              env_temp_c: float = 20.0, env_humidity: float = 50.0) -> SensorReading:
        p = self.mount.params
        ox, oy = self._world_origin(robot_x, robot_y, robot_theta)
        beam_angle = robot_theta + math.radians(self.mount.angle_deg)

        rays, echoes = self._cast(ox, oy, beam_angle, maze)
        reading = SensorReading(sensor=self.name, distance=None, true_distance=None,
                                origin=(ox, oy), beam_angle=beam_angle,
                                rays=rays, echoes=echoes)

        # Threshold detector: first echo (in time) that crosses the threshold.
        winner = None
        for e in sorted(echoes, key=lambda e: e.path):
            if e.level * (1.0 + self.rng.gauss(0.0, p.amplitude_noise)) >= 1.0:
                winner = e
                break
        if winner is None:
            reading.timeout = True
            return reading

        winner.detected = True
        reading.true_distance = winner.path / 2.0
        reported = self._to_reported(reading.true_distance, env_temp_c, env_humidity)

        if reported < p.min_range:
            reading.blind_zone = True
            reading.distance = p.min_range
        elif reported > p.max_range:
            reading.timeout = True
        else:
            reading.distance = reported
        return reading

    def _to_reported(self, true_dist: float, env_temp_c: float, env_humidity: float) -> float:
        """Echo delay -> firmware distance: real speed of sound vs the one the
        firmware assumes, timing jitter, then timer quantization."""
        p = self.mount.params
        t = 2.0 * true_dist / speed_of_sound(env_temp_c, env_humidity)
        d = t * speed_of_sound(p.firmware_assumed_temp_c, 50.0) / 2.0
        d += self.rng.gauss(0.0, p.noise_std_base + p.noise_std_per_meter * d)
        if p.distance_resolution > 0:
            d = round(d / p.distance_resolution) * p.distance_resolution
        return max(0.0, d)


class SensorArray:
    """Container for all mounted sensors."""

    def __init__(self, seed: Optional[int] = None):
        self._sensors: Dict[int, UltrasonicSensor] = {}
        self._next_id = 0
        self._rng = random.Random(seed)

    def add(self, angle_deg: float, offset: Point = (0.0, 0.0), name: Optional[str] = None,
            **param_overrides) -> int:
        params = SensorParams(**param_overrides)
        mount = SensorMount(angle_deg=angle_deg, offset=offset, name=name or "", params=params)
        sid = self._next_id
        self._next_id += 1
        self._sensors[sid] = UltrasonicSensor(mount, sid, rng=random.Random(self._rng.random()))
        if not mount.name:
            mount.name = f"s{sid}"
        return sid

    def remove(self, sensor_id: int) -> None:
        self._sensors.pop(sensor_id, None)

    def clear(self) -> None:
        self._sensors.clear()

    def items(self):
        return self._sensors.items()

    def __len__(self):
        return len(self._sensors)

    def sense_all(self, robot_x: float, robot_y: float, robot_theta: float, maze: Maze,
                  env_temp_c: float = 20.0, env_humidity: float = 50.0
                  ) -> Dict[str, SensorReading]:
        return {s.name: s.sense(robot_x, robot_y, robot_theta, maze, env_temp_c, env_humidity)
                for s in self._sensors.values()}
