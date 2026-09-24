"""
demo.py
-------
Two runnable examples that show how an external script plugs into the
simulator. Neither example imports anything from `visualizer` except to
watch the run - your own algorithm scripts only need `environment.py`
and `sensors.py`.

    python3 demo.py manual      # drive the robot yourself, arrow keys
    python3 demo.py autopilot   # a simple left-hand-wall-follower

Add "detailed" (or press d in the window) for the acoustic bounce view:

    python3 demo.py autopilot detailed

The autopilot only ever looks at `env.sense()` output (front/left/right
distances) - it never touches env.maze - so it's a fair stand-in for
"my maze-solving algorithm" that you'd swap in here.
"""

import sys

from environment import Environment
from visualizer import MazeVisualizer

CELL = 0.3
STEP_FORWARD = 0.03      # meters advanced per frame
TURN_STEP = 6.0          # degrees per key-press / decision


def build_env(seed=42) -> Environment:
    env = Environment(rows=8, cols=8, cell_size=CELL, seed=seed, loop_prob=0.06,
                       temperature_c=24.0, humidity_percent=45.0)
    # Three sensors: front, left, right, with HC-SR04-like defaults (4 m rated
    # range, -6 dB beam half-angle 15 deg). Override any SensorParams field
    # per sensor if you like. Note max_range also sets the detection
    # threshold, so lowering it makes the sensor deafer, not just shorter.
    env.add_sensor(angle_deg=0, name="front")
    env.add_sensor(angle_deg=90, name="left")
    env.add_sensor(angle_deg=-90, name="right")
    return env


# ---------------------------------------------------------------------------
# Manual driving demo
# ---------------------------------------------------------------------------

def run_manual(detailed=False):
    env = build_env()
    viz = MazeVisualizer(env, title="Manual drive - arrow keys", detailed=detailed)
    pressed = set()

    def on_key(key):
        if key in ("up", "down", "left", "right"):
            pressed.add(key)

    def step(env):
        if "up" in pressed:
            env.move(STEP_FORWARD)
        if "down" in pressed:
            env.move(-STEP_FORWARD)
        if "left" in pressed:
            env.move(0, dtheta_deg=TURN_STEP)
        if "right" in pressed:
            env.move(0, dtheta_deg=-TURN_STEP)
        pressed.clear()
        env.time += 0.1
        if env.at_goal():
            print("Reached the goal!")
            raise StopIteration

    viz.on_key = on_key
    viz.run(step, interval_ms=80)


# ---------------------------------------------------------------------------
# Simple autopilot: left-hand wall follower
#
# Classic maze-solving heuristic (guaranteed to find the goal in a
# perfect/simply-connected maze): keep a wall on your left; if you can
# turn left, do; else go straight if clear; else turn right; else
# it's a dead end, turn around.
#
# This is intentionally simple - the point is to show the *interface*,
# not to ship a state-of-the-art solver. Swap this function out for your
# own algorithm; it only ever reads `env.sense()`.
# ---------------------------------------------------------------------------

FRONT_STOP = CELL * 0.3          # turn away if a wall gets this close ahead
TARGET_LEFT = CELL * 0.35        # desired standoff distance from the left wall
NO_WALL = 999.0                  # stand-in for "nothing detected"


def autopilot_step(env: Environment):
    readings = env.sense()

    def dist(name):
        r = readings[name]
        if r.timeout or r.distance is None:
            return NO_WALL
        return r.distance

    front, left = dist("front"), dist("left")

    if front < FRONT_STOP:
        # wall dead ahead: rotate in place until the way is clear
        env.move(0, dtheta_deg=-TURN_STEP)
    elif left < NO_WALL:
        # proportional control: hug the left wall at TARGET_LEFT
        error = left - TARGET_LEFT
        dtheta = max(-TURN_STEP, min(TURN_STEP, error * 45.0))
        env.move(STEP_FORWARD, dtheta_deg=dtheta)
    else:
        # no wall on the left at all - drift gently left to go find one
        env.move(STEP_FORWARD, dtheta_deg=TURN_STEP * 0.4)

    env.time += 0.1
    if env.at_goal():
        print(f"Reached the goal at t={env.time:.1f}s")
        raise StopIteration


def run_autopilot(detailed=False):
    env = build_env()
    viz = MazeVisualizer(env, title="Autopilot - left-hand wall follower", detailed=detailed)
    viz.run(autopilot_step, interval_ms=60)


# ---------------------------------------------------------------------------
# Headless example: no GUI at all, just the raw API an algorithm would use.
# ---------------------------------------------------------------------------

def run_headless(max_steps=4000):
    env = build_env()
    for i in range(max_steps):
        autopilot_step_headless(env)
        if env.at_goal():
            print(f"Reached goal in {i} steps, t={env.time:.1f}s")
            return
    print("Did not reach the goal within max_steps.")


def autopilot_step_headless(env: Environment):
    try:
        autopilot_step(env)
    except StopIteration:
        pass


USAGE = """\
Usage: python3 demo.py <mode> [detailed]

  manual      open the GUI, drive the robot yourself with arrow keys
  autopilot   open the GUI, watch a simple wall-follower run
  headless    NO GUI - just runs the loop in text mode and prints the result
  detailed    (optional, after manual/autopilot) start in the bounce/echo view;
              you can also press d in the window to toggle it

If you ran this with no arguments (or your IDE's "run" button did), you
got 'headless' by design - "Did not reach the goal within max_steps."
is normal output for that mode, not an error. For a visible window, run:

    python3 demo.py autopilot
"""

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(USAGE)
    mode = sys.argv[1] if len(sys.argv) > 1 else "headless"
    detailed = "detailed" in sys.argv[2:]
    if mode == "manual":
        run_manual(detailed)
    elif mode == "autopilot":
        run_autopilot(detailed)
    elif mode == "headless":
        run_headless()
    else:
        print(f"Unknown mode: {mode!r}\n")
        print(USAGE)
