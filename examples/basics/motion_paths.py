"""Working with paths: absolute, relative, merged, and edge-padded.

Ports the "Working with Paths" tutorial. Positions and orientations are vectorized "paths"; every
field computation runs once over the whole path. Shows absolute path assignment, relative
``move``/``rotate``, merging two rotations into a spiral, and the edge-padding rule that keeps a
shorter path static while a longer one keeps moving.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import magpylib_jax as magpy


def main() -> dict:
    # Absolute path: set position and orientation arrays at once.
    ts = np.linspace(0, 10, 21)
    pos = np.array([(0.1 * t, 0.0, 0.1 * np.sin(t)) for t in ts])
    ori = R.from_rotvec(np.array([(0.0, -0.1 * np.cos(t) * 0.785, 0.0) for t in ts]))
    sensor = magpy.Sensor(position=pos, orientation=ori)
    print(f"absolute path length: {sensor.position.shape[0]}")

    # Relative motion: a scalar move shifts the whole existing path.
    sensor2 = sensor.copy()
    sensor2.move((0.0, 0.0, 0.05))
    shift = np.asarray(sensor2.position) - np.asarray(sensor.position)
    print(f"relative move dz:     {shift[:, 2].mean():.3f} (constant)")

    # Merging paths: a self-rotation then an about-origin rotation makes a spiral.
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 0.1), dimension=(0.02, 0.02, 0.02))
    cube.position = np.linspace((0, 0, 0), (0.1, 0, 0), 60)
    cube.rotate_from_rotvec(np.linspace((0, 0, 0), (0, 0, 360), 30), start=0)
    cube.rotate_from_rotvec(np.linspace((0, 0, 0), (0, 0, 360), 30), anchor=0, start=30)
    print(f"merged spiral length: {cube.position.shape[0]}")

    # Edge-padding: the shorter loop stays put while the longer one keeps moving.
    loop_long = magpy.current.Circle(current=1.0, diameter=1.0,
                                     position=[(0, 0, i) for i in range(4)])
    loop_short = magpy.current.Circle(current=1.0, diameter=1.0,
                                      position=[(0, 0, i) for i in range(2)])
    b = np.asarray(magpy.getB([loop_long, loop_short], (0, 0, 0)))
    print(f"edge-padded B shape:  {b.shape}")

    return {"path_len": sensor.position.shape[0], "spiral_len": cube.position.shape[0],
            "padded_shape": b.shape}


if __name__ == "__main__":
    main()
