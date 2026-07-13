"""Collections and sensors: a Helmholtz coil pair.

Ports the "Working with Collections" practical example: build two coils by grouping circular
current windings into ``Collection`` objects, combine them into a single Helmholtz source, and read
the field both directly and through a ``Sensor``. A collection of sources acts as one source.
"""

from __future__ import annotations

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    # Build one coil from five circular windings, then mirror it into a pair.
    coil1 = magpy.Collection()
    for z in np.linspace(-0.0005, 0.0005, 5):
        coil1.add(magpy.current.Circle(current=1.0, diameter=0.02, position=(0.0, 0.0, z)))
    coil1.position = (0.0, 0.0, -0.005)
    coil2 = coil1.copy(position=(0.0, 0.0, 0.005))
    helmholtz = coil1 + coil2

    print(f"coil1 windings:  {len(coil1.children)}")
    print(f"helmholtz sources: {len(helmholtz.sources)}")

    # The collection acts as a single source: field on axis and at the center.
    b_center = np.asarray(helmholtz.getB((0.0, 0.0, 0.0)))

    # Read the same field with a Sensor observer (collection as observer input works too).
    sensor = magpy.Sensor(position=(0.0, 0.0, 0.0))
    b_sensor = np.asarray(sensor.getB(helmholtz))

    # Uniformity: sweep a short on-axis line and report the field spread.
    line = np.linspace((0, 0, -0.002), (0, 0, 0.002), 9)
    bz = np.asarray(helmholtz.getB(line))[:, 2]
    uniformity = (bz.max() - bz.min()) / abs(bz.mean())

    print(f"B at center:        {b_center} T")
    print(f"B via sensor:       {b_sensor} T")
    print(f"on-axis Bz spread:  {uniformity:.2%} of mean")

    return {"b_center": b_center, "b_sensor": b_sensor, "uniformity": uniformity}


if __name__ == "__main__":
    main()
