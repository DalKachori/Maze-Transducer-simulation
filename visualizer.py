"""
visualizer.py
-------------
matplotlib front-end for the simulator. Completely separate from the
physics/API - your maze-solving algorithm never imports this.

Two view modes (press  d  in the window to toggle, or pass detailed=True):

SIMPLE (default) - the original look
    - Robot: triangle pointing along its heading; faint dotted trail.
    - One line per sensor ray, out to its first wall hit:
        solid green   -> that ray produced a detectable direct echo
        dashed amber  -> that ray's detectable echo came via a bounce
                         (corner / multipath)
        dotted red    -> no echo strong enough (grazing wall, too far,
                         or off the beam centre)
    - Side panel: live distance reading per sensor.

DETAILED - what the acoustics are actually doing
    - Every ray's full bounce path (blue), with a dot at each wall hit.
    - Echo return legs from later bounces back to the sensor:
        amber solid  -> strong enough to detect
        red dotted   -> too weak (below the detection threshold)
    - The echo the receiver actually latched is drawn as one thick white
      path: sensor -> bounces -> back to sensor.
    - An "echogram" below the readouts: every candidate echo as a stem at
      its distance, height = amplitude relative to the detection
      threshold (dB). Above the red 0 dB line = detectable; the receiver
      reports the FIRST detectable stem. Circles are direct echoes,
      triangles are echoes that bounced first; the ringed one is the
      reading.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Rectangle

from environment import Environment
from sensors import SensorReading

_BG = "#101418"
_WALL = "#3d4b57"
_ROBOT = "#f2f2f2"
_TRAIL = "#4d5b66"
_GOAL = "#2ecc71"
_HIT = "#37e08e"
_MULTIPATH = "#f5b942"
_MISS = "#e5484d"
_PATH = "#4c8dff"
_TEXT = "#d8dee4"
_PANEL_BG = "#181d22"
_SENSOR_COLORS = ["#37e08e", "#5b9cff", "#f5b942", "#c77dff", "#ff7a90"]

_DOTTED = (0, (1, 2))
_DASHED = (0, (4, 2))


def _segments(points):
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


class MazeVisualizer:
    def __init__(self, env: Environment, figsize=(10, 6), title: str = "Ultrasonic Maze Simulator",
                 detailed: bool = False):
        self.env = env
        self.detailed = detailed
        plt.rcParams["toolbar"] = "None"
        self.fig, (self.ax_maze, self.ax_panel) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.4, 1.3]}
        )
        self.fig.patch.set_facecolor(_BG)
        self.fig.suptitle(title, color=_TEXT, fontsize=13, fontweight="bold")
        self._setup_axes()
        self._draw_static_maze()

        # Echogram lives in its own axes under the text panel (detailed mode only)
        px, py, pw, ph = self.ax_panel.get_position().bounds
        self._panel_full = (px, py, pw, ph)
        self._panel_top = (px, py + ph * 0.52, pw, ph * 0.48)
        self.ax_echo = self.fig.add_axes((px + pw * 0.14, py + ph * 0.06, pw * 0.82, ph * 0.38))

        self._robot_patch = None
        self._trail_line, = self.ax_maze.plot([], [], color=_TRAIL, lw=1, ls=":", zorder=2)
        self._dynamic = []          # artists redrawn every frame
        self._panel_text = self.ax_panel.text(0.06, 0.97, "", color=_TEXT, fontsize=9.5,
                                              family="monospace", va="top", ha="left",
                                              transform=self.ax_panel.transAxes)
        self._apply_layout()
        self.on_key: Optional[Callable[[str], None]] = None
        self.fig.canvas.mpl_connect("key_press_event", self._handle_key)

    # -- setup ---------------------------------------------------------

    def _setup_axes(self):
        m = self.env.maze
        self.ax_maze.set_facecolor(_BG)
        self.ax_maze.set_xlim(-0.05 * m.width, m.width * 1.05)
        self.ax_maze.set_ylim(-0.05 * m.height, m.height * 1.05)
        self.ax_maze.set_aspect("equal")
        self.ax_maze.set_xticks([])
        self.ax_maze.set_yticks([])
        for spine in self.ax_maze.spines.values():
            spine.set_visible(False)

        self.ax_panel.set_facecolor(_PANEL_BG)
        self.ax_panel.set_xticks([])
        self.ax_panel.set_yticks([])
        for spine in self.ax_panel.spines.values():
            spine.set_visible(False)

    def _draw_static_maze(self):
        m = self.env.maze
        for (x1, y1), (x2, y2) in m.segments:
            self.ax_maze.plot([x1, x2], [y1, y2], color=_WALL, lw=2.2, solid_capstyle="round",
                               zorder=3)
        gx, gy = m.goal_position()
        s = m.cell_size
        self.ax_maze.add_patch(Rectangle((gx - s * 0.4, gy - s * 0.4), s * 0.8, s * 0.8,
                                          facecolor=_GOAL, alpha=0.25, edgecolor=_GOAL,
                                          lw=1.2, zorder=1))
        self.ax_maze.text(gx, gy, "goal", color=_GOAL, ha="center", va="center",
                           fontsize=8, alpha=0.9, zorder=1)

    def _apply_layout(self):
        if self.detailed:
            self.ax_panel.set_position(self._panel_top)
            self.ax_echo.set_visible(True)
        else:
            self.ax_panel.set_position(self._panel_full)
            self.ax_echo.set_visible(False)

    def toggle_detail(self):
        self.detailed = not self.detailed
        self._apply_layout()
        self.redraw()

    # -- per-frame drawing ------------------------------------------------

    def _clear_dynamic(self):
        for a in self._dynamic:
            a.remove()
        self._dynamic = []

    def _add_lines(self, segs, color, lw, ls="-", alpha=1.0, z=4):
        if segs:
            lc = LineCollection(segs, colors=color, linewidths=lw, linestyles=ls,
                                alpha=alpha, zorder=z)
            self.ax_maze.add_collection(lc)
            self._dynamic.append(lc)

    def _draw_robot(self):
        x, y, theta = self.env.pose
        r = self.env.maze.cell_size * 0.28
        pts = [
            (x + r * math.cos(theta), y + r * math.sin(theta)),
            (x + r * 0.6 * math.cos(theta + 2.6), y + r * 0.6 * math.sin(theta + 2.6)),
            (x + r * 0.6 * math.cos(theta - 2.6), y + r * 0.6 * math.sin(theta - 2.6)),
        ]
        if self._robot_patch is not None:
            self._robot_patch.remove()
        self._robot_patch = Polygon(pts, closed=True, facecolor=_ROBOT, edgecolor="black",
                                     lw=0.5, zorder=5)
        self.ax_maze.add_patch(self._robot_patch)

    def _draw_rays_simple(self, readings: Dict[str, SensorReading]):
        direct, bounced, weak = [], [], []
        for r in readings.values():
            for ray in r.rays:
                seg = (ray.points[0], ray.points[1])       # first leg only
                if ray.best_level >= 1.0:
                    (direct if ray.best_bounces == 1 else bounced).append(seg)
                else:
                    weak.append(seg)
        self._add_lines(weak, _MISS, 1.0, _DOTTED, 0.35)
        self._add_lines(bounced, _MULTIPATH, 1.0, _DASHED, 0.7)
        self._add_lines(direct, _HIT, 1.0, "-", 0.9)

    def _draw_rays_detailed(self, readings: Dict[str, SensorReading]):
        forward, first_ok, returns_ok, returns_weak, winners, verts = [], [], [], [], [], []
        for r in readings.values():
            # Follow each ray only as far as its last bounce that still made
            # a non-negligible echo (>= -20 dB re threshold); the rest is noise.
            last = {}
            for e in r.echoes:
                if e.level >= 0.1:
                    last[e.ray] = max(last.get(e.ray, 1), e.bounces)
            for i, ray in enumerate(r.rays):
                n = last.get(i, 1)
                pts = ray.points[:n + 1]
                segs = _segments(pts)
                if ray.best_level >= 1.0 and ray.best_bounces == 1:
                    first_ok.append(segs[0])
                    segs = segs[1:]
                forward.extend(segs)
                if not (ray.escaped and len(pts) == len(ray.points)):
                    verts.extend(pts[1:])
            for e in r.echoes:
                if e.bounces >= 2 and e.level >= 0.25:
                    leg = (e.points[-2], e.points[-1])
                    (returns_ok if e.level >= 1.0 else returns_weak).append(leg)
                if e.detected:
                    winners.extend(_segments(e.points))
        self._add_lines(forward, _PATH, 0.9, "-", 0.5, 3.5)
        self._add_lines(returns_weak, _MISS, 0.9, _DOTTED, 0.5, 3.6)
        self._add_lines(first_ok, _HIT, 1.1, "-", 0.85, 3.7)
        self._add_lines(returns_ok, _MULTIPATH, 1.1, "-", 0.9, 3.8)
        self._add_lines(winners, "white", 2.6, "-", 0.95, 6)
        if verts:
            xs, ys = zip(*verts)
            dots, = self.ax_maze.plot(xs, ys, "o", ms=3, color="#e8ecf0", alpha=0.8,
                                       mec="none", zorder=4)
            self._dynamic.append(dots)

    def _fmt_reading(self, r: SensorReading) -> str:
        if r.timeout:
            return "-- (no echo)"
        if r.blind_zone:
            return f"{r.distance*100:5.1f} cm (blind zone)"
        return f"{r.distance*100:5.1f} cm"

    def _draw_panel(self, readings: Dict[str, SensorReading]):
        lines = [f"t = {self.env.time:6.2f} s",
                 f"T = {self.env.temperature_c:4.1f} C   RH = {self.env.humidity_percent:4.0f}%",
                 ""]
        for name, r in readings.items():
            lines.append(f"{name:>7s}: {self._fmt_reading(r)}")
            if self.detailed:
                win = next((e for e in r.echoes if e.detected), None)
                if win is not None:
                    kind = "direct" if win.bounces == 1 else f"{win.bounces} bounces"
                    lines.append(f"         {kind}, {20*math.log10(win.level):+.0f} dB")
                else:
                    lines.append("         no echo above threshold")
        lines.append("")
        lines.append("d: toggle detailed view")
        if not self.detailed:
            lines.append("arrows: drive (manual mode)")
        self._panel_text.set_text("\n".join(lines))

    def _draw_echogram(self, readings: Dict[str, SensorReading]):
        ax = self.ax_echo
        ax.cla()
        ax.set_facecolor(_PANEL_BG)
        lo, hi = -30.0, 35.0
        xmax = min(self.env.maze.width,
                   max((s.mount.params.max_range for _, s in self.env.sensors.items()), default=1.0))
        ax.set_xlim(0, xmax)
        ax.set_ylim(lo, hi)
        ax.tick_params(colors=_TEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(_TRAIL)
        ax.set_xlabel("echo distance (m)", color=_TEXT, fontsize=8)
        ax.set_ylabel("level vs threshold (dB)", color=_TEXT, fontsize=8)
        ax.axhline(0, color=_MISS, lw=1, ls="--")
        for i, (name, r) in enumerate(readings.items()):
            c = _SENSOR_COLORS[i % len(_SENSOR_COLORS)]
            ax.plot([], [], "s", color=c, ms=4, label=name)
            for e in r.echoes:
                db = max(lo, 20 * math.log10(e.level))
                x = e.path / 2.0
                ax.vlines(x, lo, db, colors=c, lw=1, alpha=0.7 if e.level >= 1 else 0.3)
                ax.plot(x, db, "o" if e.bounces == 1 else "^", color=c, ms=3.5,
                        alpha=1.0 if e.level >= 1 else 0.4, mec="none")
                if e.detected:
                    ax.plot(x, db, "o", ms=9, mfc="none", mec="white", mew=1.4)
        ax.legend(loc="upper right", fontsize=7, facecolor=_PANEL_BG, edgecolor=_TRAIL,
                  labelcolor=_TEXT, framealpha=0.9)

    def redraw(self):
        readings = self.env.sense()
        self._draw_robot()
        self._clear_dynamic()
        if self.detailed:
            self._draw_rays_detailed(readings)
            self._draw_echogram(readings)
        else:
            self._draw_rays_simple(readings)
        self._draw_panel(readings)
        trail = self.env.trail()
        if trail:
            xs, ys = zip(*trail)
            self._trail_line.set_data(xs, ys)
        self.fig.canvas.draw_idle()
        return readings

    # -- interaction --------------------------------------------------------

    def _handle_key(self, event):
        if event.key is None:
            return
        if event.key == "d":
            self.toggle_detail()
        elif self.on_key is not None:
            self.on_key(event.key)

    def show(self):
        self.redraw()
        plt.show()

    def run(self, step_fn: Callable[[Environment], None], n_frames: Optional[int] = None,
            interval_ms: int = 120):
        """Animate: on every frame call step_fn(env), then redraw.

        step_fn is where your algorithm (or a manual controller) calls
        env.move(...) using env.sense(). Pass n_frames=None to run until
        the window is closed or step_fn raises StopIteration.
        """
        from matplotlib.animation import FuncAnimation

        def _update(frame):
            try:
                step_fn(self.env)
            except StopIteration:
                plt.close(self.fig)
                return []
            self.redraw()
            return []

        self._anim = FuncAnimation(self.fig, _update, frames=n_frames,
                                    interval=interval_ms, repeat=False, cache_frame_data=False)
        plt.show()
