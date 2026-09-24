"""
maze_gen.py
-----------
Automatic maze generation for the ultrasonic-sensor simulator.

A maze is a grid of `rows` x `cols` cells. Each cell knows which of its
four sides (N, E, S, W) have a wall. The maze is generated with a
recursive-backtracker (randomized DFS), which produces a "perfect" maze
(exactly one path between any two cells, no loops). An optional
`loop_prob` randomly knocks down extra walls after generation to add
loops/braids, which is closer to some real-world environments and makes
the solving problem harder / more interesting.

Geometry convention
--------------------
- Cell (row, col) occupies the axis-aligned square
      x in [col*cell_size, (col+1)*cell_size]
      y in [row*cell_size, (row+1)*cell_size]
- +x is "east", +y is "north".
- The maze is exposed to the rest of the simulator purely as a set of
  wall line segments plus a few conveniences (start/goal, bounds,
  raycasting, collision testing). Nothing else in the codebase needs to
  know about the underlying grid representation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


# --------------------------------------------------------------------------
# Low-level geometry helpers (shared by raycasting and collision checks)
# --------------------------------------------------------------------------

def _seg_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Tuple[float, float]]:
    """Intersection of segment p1-p2 with segment p3-p4.

    Returns (t, u) where the intersection point is p1 + t*(p2-p1) and also
    p3 + u*(p4-p3), with t, u in [0, 1]; or None if they don't intersect.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None  # parallel / collinear

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None


# --------------------------------------------------------------------------
# Maze
# --------------------------------------------------------------------------

@dataclass
class Maze:
    rows: int
    cols: int
    cell_size: float
    walls: List[List[dict]] = field(repr=False)  # walls[r][c] = {'N','E','S','W': bool}
    start_cell: Tuple[int, int] = (0, 0)
    goal_cell: Tuple[int, int] = (0, 0)
    segments: List[Segment] = field(default_factory=list, repr=False)
    _seg_cache: Optional[list] = field(default=None, repr=False)

    # -- geometry ----------------------------------------------------------

    @property
    def width(self) -> float:
        return self.cols * self.cell_size

    @property
    def height(self) -> float:
        return self.rows * self.cell_size

    def cell_center(self, cell: Tuple[int, int]) -> Point:
        r, c = cell
        return ((c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size)

    def start_position(self) -> Point:
        return self.cell_center(self.start_cell)

    def goal_position(self) -> Point:
        return self.cell_center(self.goal_cell)

    def cell_of(self, x: float, y: float) -> Tuple[int, int]:
        c = int(x // self.cell_size)
        r = int(y // self.cell_size)
        c = min(max(c, 0), self.cols - 1)
        r = min(max(r, 0), self.rows - 1)
        return r, c

    # -- raycasting / collision ---------------------------------------------

    def _seg_data(self):
        """Per-segment (x1, y1, dx, dy, nx, ny), built once and reused."""
        if self._seg_cache is None or len(self._seg_cache) != len(self.segments):
            cache = []
            for (x1, y1), (x2, y2) in self.segments:
                sx, sy = x2 - x1, y2 - y1
                L = math.hypot(sx, sy)
                cache.append((x1, y1, sx, sy, -sy / L, sx / L))
            self._seg_cache = cache
        return self._seg_cache

    def raycast(self, ox: float, oy: float, dx: float, dy: float,
                max_range: float, min_t: float = 1e-6) -> Optional[Tuple[float, Point]]:
        """Nearest wall hit along a ray, restricted to max_range.

        Returns (distance, unit normal facing back toward the ray) or None.
        Hits closer than `min_t` are ignored, so a ray that starts ON a wall
        (a bounce point) can't immediately re-hit the wall it just left.
        """
        norm = math.hypot(dx, dy)
        if norm < 1e-12:
            return None
        dx, dy = dx / norm, dy / norm

        best_t = max_range
        best = None
        for seg in self._seg_data():
            x1, y1, sx, sy, _nx, _ny = seg
            denom = dx * sy - dy * sx
            if -1e-12 < denom < 1e-12:
                continue
            qx, qy = x1 - ox, y1 - oy
            t = (qx * sy - qy * sx) / denom
            if t <= min_t or t > best_t:
                continue
            u = (qx * dy - qy * dx) / denom
            if 0.0 <= u <= 1.0:
                best_t, best = t, seg
        if best is None:
            return None
        nx, ny = best[4], best[5]
        if nx * dx + ny * dy > 0:
            nx, ny = -nx, -ny
        return best_t, (nx, ny)

    def visible(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True if nothing blocks the straight line (x0,y0) -> (x1,y1)."""
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return True
        return self.raycast(x0, y0, dx, dy, L - 1e-6) is None

    def path_blocked(self, p_from: Point, p_to: Point) -> bool:
        """True if the straight segment p_from->p_to crosses any wall.

        Used for collision checking of the (zero-size) robot as it moves.
        """
        for seg in self.segments:
            if _seg_intersect(p_from, p_to, seg[0], seg[1]) is not None:
                return True
        return False

    def in_bounds(self, x: float, y: float, margin: float = 1e-6) -> bool:
        return margin <= x <= self.width - margin and margin <= y <= self.height - margin


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

_DX = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
_OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}


def generate_maze(rows: int = 10, cols: int = 10, cell_size: float = 0.5,
                   seed: Optional[int] = None, loop_prob: float = 0.0,
                   start_cell: Optional[Tuple[int, int]] = None,
                   goal_cell: Optional[Tuple[int, int]] = None) -> Maze:
    """Generate a random maze.

    Parameters
    ----------
    rows, cols : grid dimensions.
    cell_size  : size of one cell in meters (also sets the physical scale
                 the sensor physics operates in).
    seed       : RNG seed for reproducibility. None = random every call.
    loop_prob  : probability, per interior wall, of knocking it down after
                 the perfect maze is built. 0.0 = perfect maze (a tree,
                 exactly one path between any two cells). Try 0.05-0.15
                 for a maze with some loops/shortcuts.
    start_cell, goal_cell : defaults to bottom-left / top-right corners.
    """
    rng = random.Random(seed)

    walls = [[{'N': True, 'E': True, 'S': True, 'W': True} for _ in range(cols)]
             for _ in range(rows)]
    visited = [[False] * cols for _ in range(rows)]

    # Recursion depth can get large for big mazes; use an explicit stack.
    def carve_iterative(r0, c0):
        stack = [(r0, c0)]
        visited[r0][c0] = True
        while stack:
            r, c = stack[-1]
            dirs = list(_DX.items())
            rng.shuffle(dirs)
            advanced = False
            for d, (dc, dr) in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    walls[r][c][d] = False
                    walls[nr][nc][_OPPOSITE[d]] = False
                    visited[nr][nc] = True
                    stack.append((nr, nc))
                    advanced = True
                    break
            if not advanced:
                stack.pop()

    carve_iterative(0, 0)

    # Optional braiding: knock down some extra interior walls to add loops.
    if loop_prob > 0:
        for r in range(rows):
            for c in range(cols):
                for d, (dc, dr) in _DX.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and walls[r][c][d]:
                        if rng.random() < loop_prob:
                            walls[r][c][d] = False
                            walls[nr][nc][_OPPOSITE[d]] = False

    start_cell = start_cell or (0, 0)
    goal_cell = goal_cell or (rows - 1, cols - 1)

    maze = Maze(rows=rows, cols=cols, cell_size=cell_size, walls=walls,
                start_cell=start_cell, goal_cell=goal_cell)
    maze.segments = _walls_to_segments(maze)
    return maze


def _walls_to_segments(maze: Maze) -> List[Segment]:
    cs = maze.cell_size
    segs: List[Segment] = []
    for r in range(maze.rows):
        for c in range(maze.cols):
            w = maze.walls[r][c]
            x0, y0 = c * cs, r * cs
            x1, y1 = x0 + cs, y0 + cs
            if w['S']:
                segs.append(((x0, y0), (x1, y0)))
            if w['N']:
                segs.append(((x0, y1), (x1, y1)))
            if w['W']:
                segs.append(((x0, y0), (x0, y1)))
            if w['E']:
                segs.append(((x1, y0), (x1, y1)))
    # De-duplicate shared walls between adjacent cells (each interior wall
    # would otherwise be listed twice - harmless for raycasting but doubles
    # the work, so we drop exact duplicates).
    seen = set()
    unique = []
    for (p1, p2) in segs:
        key = tuple(sorted([p1, p2]))
        if key not in seen:
            seen.add(key)
            unique.append((p1, p2))
    return unique
