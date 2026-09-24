# Ultrasonic Maze-Solver Sensor Simulator

A self-contained Python simulator for developing and testing maze-solving
algorithms against a physically-grounded model of ultrasonic distance
sensors, on automatically generated mazes.

```
maze_gen.py     - random maze generation + raycasting/visibility geometry
sensors.py      - the ultrasonic sensor acoustic model
environment.py  - the main API external scripts use (Environment class)
visualizer.py   - matplotlib GUI: simple view + a "detailed" acoustics view
demo.py         - runnable examples: manual drive, autopilot, headless
```

Requires only `matplotlib` (for the GUI) — the simulation core
(`maze_gen.py`, `sensors.py`, `environment.py`) has **zero** dependencies
beyond the standard library, so you can `import environment` from any
test harness without pulling in a plotting stack.

## Quick start

```python
from environment import Environment

env = Environment(rows=10, cols=10, cell_size=0.3, seed=42, loop_prob=0.05)

env.add_sensor(angle_deg=0,   name="front")
env.add_sensor(angle_deg=90,  name="left")
env.add_sensor(angle_deg=-90, name="right")

while not env.at_goal():
    readings = env.sense()                 # {'front': SensorReading, ...}
    front = readings["front"].distance     # meters, or None if no echo
    ...                                    # your algorithm decides
    env.move(forward=0.02, dtheta_deg=0)   # advance / steer
```

Watch it live instead of headless:

```bash
python3 demo.py manual                 # drive with arrow keys
python3 demo.py autopilot              # a simple left-hand wall-follower
python3 demo.py autopilot detailed     # same, but opens in the acoustics view
python3 demo.py headless               # no GUI, just runs the loop and prints the result
```

In any GUI mode, press **`d`** at any time to toggle between the simple
view and the detailed acoustics view.

## The `Environment` API

- `Environment(rows, cols, cell_size, seed=None, loop_prob=0.0, temperature_c=20.0, humidity_percent=50.0, collision_step=0.005)`
  Generates the maze. `loop_prob` > 0 breaks some walls after generation
  to add loops (0 = a "perfect" maze, exactly one path anywhere).
- `add_sensor(angle_deg, offset=(0,0), name=None, **physics_overrides) -> id`
  Mount as many ultrasonic sensors as you like, at any angle
  (0° = forward, 90° = left, -90° = right, 180° = behind). `offset` is a
  (forward, left) mounting position in meters — defaults to (0, 0)
  since the robot currently has zero footprint; wire it up once you give
  the robot real dimensions. Any field of `SensorParams` (see below) can
  be overridden per-sensor.
- `sense() -> Dict[str, SensorReading]` — fires every sensor once. This is
  the **only** thing your algorithm should read from; it doesn't need
  `env.maze` at all (that would be cheating — a real robot doesn't get to
  see through walls). It does *not* advance `env.time` by itself — bump
  `env.time` yourself in your own loop (see `demo.py`, which adds a fixed
  step per decision).
- `move(forward, dtheta_deg=0.0) -> MoveResult` — rotates then drives the
  point-robot forward/back. Sweeps the path in small steps so it can't
  tunnel through a wall; `MoveResult.success=False` if it hit one on the
  way (in which case the pose stops right at the wall).
- `at_goal(tolerance=None)`, `reset(seed=None)`, `get_pose()`,
  `set_pose(x, y, theta)`, `trail()` (visited positions, for plotting).
- `env.temperature_c` / `env.humidity_percent` — change these mid-run to
  see the firmware-bias effect described below kick in.
- `env.ground_truth_maze()` — the actual `Maze`, for your own
  logging/plots. Don't feed it to your algorithm's decision-making if you
  want a fair "robot only sees what it can sense" test.

## How the sensor model actually works

This isn't a simple "cast a ray, add some noise" rangefinder. It's an
amplitude-based acoustic model: every sensor fires a **fan of rays**
across its beam, each ray carries a physically-motivated **amplitude**
that gets whittled down as it travels and bounces, and the sensor reports
the first echo whose amplitude clears a detection threshold — the same
way a real time-of-flight ultrasonic receiver works.

### 1. Transmit: a fan of rays, not one ray or a uniform cone

`beam_samples` rays are cast across the beam, spanning roughly
±1.5×`beam_half_angle_deg` (wide enough to capture the beam's weaker
edges, not just its -6 dB point). Each ray starts with an amplitude set
by the transducer's **directivity** — a Gaussian roll-off that's
strongest on-axis and half-power (-6 dB) at `beam_half_angle_deg` off
axis. A ray straight down the beam center carries far more energy than
one near the edge — off-axis hits are real, but weak, exactly as they'd
be on real hardware.

### 2. Propagation: spherical spreading + air absorption

Every echo's amplitude is reduced by `1 / total_path_length` (spherical
spreading) and by `exp(-air_absorption * total_path_length)` (air
absorption, in nepers/m — ultrasonic frequencies attenuate in air more
than audible sound does).

### 3. Bounces: specular + diffuse reflection, up to `max_bounces`

At each wall hit, a ray reflects like a mirror and keeps traveling (up to
`max_bounces` times), *and* the wall scatters some sound directly back
toward the sensor at that instant — this is a candidate **echo**. How
much scatters back depends on:

- **Specular component** (`reflectivity` × `(1 - diffuse_fraction)`) —
  strong only when the geometry is close to retroreflective (the mirror
  direction actually points back near the sensor); how forgiving that
  alignment needs to be is set by `specular_width_deg` (small = glassy,
  mirror-like wall; large = more forgiving).
- **Diffuse component** (`reflectivity` × `diffuse_fraction`) — a
  Lambertian term (∝ cos(incidence) × cos(exit)) that scatters some
  energy in all directions regardless of angle, the way a matte surface
  does.

This is exactly what gives the model its realistic quirks for free,
without any of them being hard-coded as special cases:

- A wall hit nearly edge-on has almost no specular return (the mirror
  direction points away from the sensor) — it only shows up if the
  diffuse term is enough, which is often not the case for near-mirror
  surfaces. Real ultrasonic sensors miss shallow walls for this reason.
- Corners give a second-bounce path straight back to the sensor even
  when neither wall alone would return anything — a classic "phantom"
  long echo.
- A ray only continues past a bounce carrying `reflectivity × (1 -
  diffuse_fraction)` of its energy; it's dropped early once it can never
  plausibly become detectable, so `max_bounces` is a ceiling, not a
  guarantee every ray traces that far.

### 4. Receive: directivity again, then a threshold detector

Each candidate echo's return angle is also weighted by the same
directivity pattern (a real transducer receives the way it transmits).
The **detection threshold** is set once per sensor from `max_range`: it's
the amplitude an ideal, fully-reflective, dead-ahead flat wall would
produce *exactly at max_range* — which is literally how manufacturers
rate a sensor's range. Every echo's `level` is that amplitude divided by
this threshold, so `level >= 1.0` means "detectable."

The receiver reports the **first echo in time** whose amplitude — after
a small multiplicative receiver-noise term (`amplitude_noise`) — clears
the threshold. This matters: a weak-but-early echo can register before a
much stronger echo that took a longer path arrives later, exactly like a
real threshold-crossing timer. One lucky ray clearing the threshold is
enough — there's no requirement that most of the beam agrees, which is
also true of real hardware; it just now has to actually be strong enough
to fight through spreading, absorption, and reflection losses to get
there, rather than being handed a free pass.

### 5. The timing chain (once an echo is picked)

The picked echo's round-trip path becomes a time via the *actual*
environmental speed of sound (`env.temperature_c`, `env.humidity_percent`).
Converting that time back to a distance uses the sensor's
`firmware_assumed_temp_c` instead — cheap firmware bakes in a fixed
speed-of-sound constant, so if the real environment differs from that
assumption, every reading gets a small systematic bias. Then timing
jitter (`noise_std_base` + `noise_std_per_meter` × distance) and
quantization (`distance_resolution`) are applied, same as real
hardware's ADC/timer limits.

### What's *not* modeled

- **3D** — the beam is a flat fan in the maze's 2D plane, not a real 3D
  cone.
- **Temporal echo overlap** — each echo is judged on its own amplitude
  independently; two echoes arriving close together in time don't
  constructively/destructively interfere with each other the way real
  overlapping wavefronts can.

## `SensorParams` reference

| field | meaning |
|---|---|
| `max_range` | meters; also *defines* the detection threshold (see step 4) — lowering it makes the sensor deafer, not just shorter-ranged |
| `min_range` | meters; blind zone (transducer ringdown) |
| `beam_half_angle_deg` | off-axis angle where transmit/receive gain is -6 dB |
| `beam_samples` | number of rays cast across the beam |
| `reflectivity` | 0–1, overall wall amplitude reflection coefficient |
| `specular_width_deg` | angular tolerance of the mirror-like return; small = glassy wall |
| `diffuse_fraction` | 0–1, portion of reflected energy that's Lambertian/scattered vs. specular |
| `max_bounces` | wall bounces traced per ray before giving up |
| `air_absorption` | nepers/m ultrasonic attenuation in air |
| `amplitude_noise` | relative receiver noise applied to each echo's amplitude before the threshold check |
| `distance_resolution` | meters; timer/ADC quantization step |
| `noise_std_base`, `noise_std_per_meter` | meters; timing jitter, constant + distance-scaled |
| `firmware_assumed_temp_c` | speed-of-sound constant baked into the (simulated) firmware |

## `SensorReading` fields

| field | meaning |
|---|---|
| `distance` | reported distance in meters, or `None` if nothing crossed threshold |
| `true_distance` | geometric one-way distance of the echo that actually triggered the reading (no timing chain applied) |
| `timeout` | no echo cleared the detection threshold |
| `blind_zone` | the winning echo maps to somewhere closer than `min_range` — reading is unreliable |
| `rays` | one `RayTrace` per beam sample — `angle`, `points` (polyline of wall hits), `escaped`, and `best_level`/`best_bounces` (its strongest or earliest-detectable echo, for coloring) |
| `echoes` | every candidate `Echo` that ray tracing found with level > -40 dB, each with `path` (round-trip distance), `level` (amplitude ÷ threshold), `bounces`, `points` (full polyline back to the sensor), and `detected` (True on exactly the one the receiver latched) |

`rays` and `echoes` are what the visualizer's detailed view and echogram
are built from — reach into them yourself if you want your own plots or
debugging output.

## The visualizer: two views, toggle with `d`

**Simple view** (default) — one line per sensor ray, out to its first
wall hit:
- solid green — that ray produced a detectable direct echo
- dashed amber — that ray's detectable echo came back via a bounce
- dotted red — nothing that ray produced was strong enough to detect

**Detailed view** (`detailed=True`, or press `d`) — shows the acoustics
underneath the single reported number:
- every ray's full bounce path, in blue, with a dot at each wall hit
- return legs from later bounces back to the sensor: amber = strong
  enough to detect, red dotted = too weak
- the one echo the receiver actually latched, drawn as a thick white
  path from sensor → bounces → back to sensor
- an **echogram** panel: every candidate echo plotted as a stem at its
  distance, height = its level in dB relative to the detection threshold
  (the dashed red line at 0 dB is the threshold). Circles are direct
  echoes, triangles bounced at least once; the reading is the ringed
  point — the first stem, left to right, that clears 0 dB.
- the side panel adds each reading's echo type (`direct` / `N bounces`)
  and its level in dB

## Extending it

- **Give the robot real size**: use each sensor's `offset` argument, and
  add your own robot-footprint check inside `Environment.move()` (right
  now the robot is treated as a point, per the current requirement).
- **Different maze styles**: `generate_maze(..., loop_prob=...)` — 0 for
  a perfect maze, higher for more loops/shortcuts.
- **Harder or gentler sensors**: everything is a `SensorParams` field —
  e.g. raise `air_absorption` or lower `reflectivity` to simulate foggy
  air or absorbent walls; lower `specular_width_deg` for glassier walls
  that miss more at shallow angles; raise `amplitude_noise` for noisier
  hardware.
- **New acoustic effects**: the whole model lives in `sensors.py`'s
  `UltrasonicSensor._cast()` (ray tracing → echo generation) and
  `sense()`/`_to_reported()` (threshold detection → timing chain) — both
  are linear, well-commented pipelines that are easy to extend.
- **Maze performance**: `Maze.raycast()` uses a precomputed per-segment
  cache (`_seg_data()`) since multi-bounce ray tracing calls it far more
  often than the old single-ray model did; `Maze.visible(a, b)` is a
  plain line-of-sight check built on the same raycast, used to confirm a
  bounce point can actually see back to the sensor before crediting it
  with an echo.
