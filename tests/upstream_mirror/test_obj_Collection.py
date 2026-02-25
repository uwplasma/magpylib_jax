import pickle
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib_jax as mpj
from magpylib_jax import MagpylibBadUserInput


def test_Collection_basics():
    """test Collection fundamentals, test against magpylib2 fields"""
    data_path = Path("/Users/rogerio/local/magpylib/tests/testdata/testdata_Collection.p")
    if not data_path.exists():
        pytest.skip("Reference collection testdata not available.")
    with data_path.resolve().open("rb") as f:
        data = pickle.load(f)
    mags, dims2, dims3, posos, angs, axs, anchs, movs, rvs, _ = data

    B1, B2 = [], []
    for mag, dim2, dim3, ang, ax, anch, mov, poso, rv in zip(
        mags, dims2, dims3, angs, axs, anchs, movs, posos, rvs, strict=False
    ):
        rot = R.from_rotvec(rv)

        pm1b = mpj.magnet.Cuboid(polarization=mag[0], dimension=dim3[0])
        pm2b = mpj.magnet.Cuboid(polarization=mag[1], dimension=dim3[1])
        pm3b = mpj.magnet.Cuboid(polarization=mag[2], dimension=dim3[2])
        pm4b = mpj.magnet.Cylinder(polarization=mag[3], dimension=dim2[0])
        pm5b = mpj.magnet.Cylinder(polarization=mag[4], dimension=dim2[1])
        pm6b = mpj.magnet.Cylinder(polarization=mag[5], dimension=dim2[2])

        pm1 = mpj.magnet.Cuboid(polarization=mag[0], dimension=dim3[0])
        pm2 = mpj.magnet.Cuboid(polarization=mag[1], dimension=dim3[1])
        pm3 = mpj.magnet.Cuboid(polarization=mag[2], dimension=dim3[2])
        pm4 = mpj.magnet.Cylinder(polarization=mag[3], dimension=dim2[0])
        pm5 = mpj.magnet.Cylinder(polarization=mag[4], dimension=dim2[1])
        pm6 = mpj.magnet.Cylinder(polarization=mag[5], dimension=dim2[2])

        col1 = mpj.Collection(pm1, pm2, pm3)
        col1.add(pm4, pm5, pm6)

        for a, aa, aaa, mv in zip(ang, ax, anch, mov, strict=False):
            for pm in [pm1b, pm2b, pm3b, pm4b, pm5b, pm6b]:
                pm.move(mv).rotate_from_angax(a, aa, aaa).rotate(rot, aaa)

            col1.move(mv).rotate_from_angax(a, aa, aaa, start=-1).rotate(
                rot, aaa, start=-1
            )

        B1 += [mpj.getB([pm1b, pm2b, pm3b, pm4b, pm5b, pm6b], poso, sumup=True)]
        B2 += [col1.getB(poso)]

    B1 = np.array(B1)
    B2 = np.array(B2)

    np.testing.assert_allclose(B1, B2)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("sens_col.getB(src_col).shape", (4, 3)),
        ("src_col.getB(sens_col).shape", (4, 3)),
        ("mixed_col.getB().shape", (4, 3)),
        ("sens_col.getB(src1, src2).shape", (2, 4, 3)),
        ("src_col.getB(sens1, sens2, sens3, sens4).shape", (4, 3)),
        ("src1.getB(sens_col).shape", (4, 3)),
        ("sens1.getB(src_col).shape", (3,)),
        ("sens1.getB(mixed_col).shape", (3,)),
        ("src1.getB(mixed_col).shape", (4, 3)),
        ("src_col.getB(mixed_col).shape", (4, 3)),
        ("sens_col.getB(mixed_col).shape", (4, 3)),
        ("mpj.getB([src1, src2], [sens1, sens2, sens3, sens4]).shape", (2, 4, 3)),
        ("mpj.getB(mixed_col,mixed_col).shape", (4, 3)),
        ("mpj.getB([src1, src2], [[1, 2, 3], (2, 3, 4)]).shape", (2, 2, 3)),
        ("src_col.getB([[1, 2, 3], (2, 3, 4)]).shape", (2, 3)),
        ("src_col.getB([1, 2, 3]).shape", (3,)),
        ("src1.getB(np.array([1,2,3])).shape", (3,)),
    ],
)
def test_col_getB(test_input, expected):
    """testing some Collection stuff with getB"""
    src1 = mpj.magnet.Cuboid(
        polarization=(1, 0, 1), dimension=(1, 1, 1), position=(0, 0, 0)
    )
    src2 = mpj.magnet.Cylinder(
        polarization=(0, 1, 0), dimension=(1, 1), position=(-1, 0, 0)
    )
    sens1 = mpj.Sensor(position=(0, 0, 1))
    sens2 = mpj.Sensor(position=(0, 0, 1))
    sens3 = mpj.Sensor(position=(0, 0, 1))
    sens4 = mpj.Sensor(position=(0, 0, 1))

    sens_col = sens1 + sens2 + sens3 + sens4
    src_col = src1 + src2
    mixed_col = sens_col + src_col
    variables = {
        "np": np,
        "mpj": mpj,
        "src1": src1,
        "src2": src2,
        "sens1": sens1,
        "sens2": sens2,
        "sens3": sens3,
        "sens4": sens4,
        "sens_col": sens_col,
        "src_col": src_col,
        "mixed_col": mixed_col,
    }

    assert eval(test_input, variables) == expected


@pytest.mark.parametrize(
    "test_input",
    [
        "src1.getB()",
        "src1.getB(src1)",
        "mpj.getB(src1, src1)",
        "src1.getB(src_col)",
        "mpj.getB(src1, src_col)",
        "sens1.getB()",
        "mpj.getB(sens1, src1)",
        "sens1.getB(sens1)",
        "mpj.getB(sens1, sens1)",
        "mpj.getB(sens1, mixed_col)",
        "mpj.getB(sens1, src_col)",
        "sens1.getB(sens_col)",
        "mpj.getB(sens1, sens_col)",
        "mixed_col.getB(src1)",
        "mpj.getB(mixed_col, src1)",
        "mixed_col.getB(sens1)",
        "mixed_col.getB(mixed_col)",
        "mixed_col.getB(src_col)",
        "mpj.getB(mixed_col, src_col)",
        "mixed_col.getB(sens_col)",
        "src_col.getB()",
        "src_col.getB(src1)",
        "mpj.getB(src_col, src1)",
        "src_col.getB(src_col)",
        "mpj.getB(src_col, src_col)",
        "sens_col.getB()",
        "mpj.getB(sens_col, src1)",
        "sens_col.getB(sens1)",
        "mpj.getB(sens_col, sens1)",
        "mpj.getB(sens_col, mixed_col)",
        "mpj.getB(sens_col, src_col)",
        "sens_col.getB(sens_col)",
        "mpj.getB(sens_col, sens_col)",
    ],
)
def test_bad_col_getB_inputs(test_input):
    """more undocumented Collection checking"""
    src1 = mpj.magnet.Cuboid(
        polarization=(1, 0, 1), dimension=(8, 4, 6), position=(0, 0, 0)
    )

    src2 = mpj.magnet.Cylinder(
        polarization=(0, 1, 0), dimension=(8, 5), position=(-15, 0, 0)
    )

    sens1 = mpj.Sensor(position=(0, 0, 6))
    sens2 = mpj.Sensor(position=(0, 0, 6))
    sens3 = mpj.Sensor(position=(0, 0, 6))
    sens4 = mpj.Sensor(position=(0, 0, 6))

    sens_col = sens1 + sens2 + sens3 + sens4
    src_col = src1 + src2
    mixed_col = sens_col + src_col
    variables = {
        "mpj": mpj,
        "src1": src1,
        "src2": src2,
        "sens1": sens1,
        "sens2": sens2,
        "sens3": sens3,
        "sens4": sens4,
        "sens_col": sens_col,
        "src_col": src_col,
        "mixed_col": mixed_col,
    }
    with pytest.raises(MagpylibBadUserInput):
        assert eval(test_input, variables) is not None


def test_col_get_item():
    """test get_item with collections"""
    pm1 = mpj.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm2 = mpj.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm3 = mpj.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))

    col = mpj.Collection(pm1, pm2, pm3)
    assert col[1] == pm2, "get_item failed"
    assert len(col) == 3, "__len__ failed"


def test_col_getH():
    """test collection getH"""
    pm1 = mpj.magnet.Sphere(polarization=(1, 2, 3), diameter=3)
    pm2 = mpj.magnet.Sphere(polarization=(1, 2, 3), diameter=3)
    col = mpj.Collection(pm1, pm2)
    H = col.getH((0, 0, 0))
    H1 = pm1.getH((0, 0, 0))
    np.testing.assert_array_equal(H, 2 * H1, err_msg="col getH fail")


def test_col_reset_path():
    """testing display"""
    pm1 = mpj.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm2 = mpj.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    col = mpj.Collection(pm1, pm2)
    col.move([(1, 2, 3)] * 10)
    col.reset_path()
    assert col[0].position.ndim == 1, "col reset path fail"
    assert col[1].position.ndim == 1, "col reset path fail"
    assert col.position.ndim == 1, "col reset path fail"


def test_Collection_squeeze():
    """testing squeeze output"""
    pm1 = mpj.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm2 = mpj.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    col = mpj.Collection(pm1, pm2)
    sensor = mpj.Sensor(pixel=[(1, 2, 3), (1, 2, 3)])
    B = col.getB(sensor)
    assert B.shape == (2, 3)
    H = col.getH(sensor)
    assert H.shape == (2, 3)

    B = col.getB(sensor, squeeze=False)
    assert B.shape == (1, 1, 1, 2, 3)
    H = col.getH(sensor, squeeze=False)
    assert H.shape == (1, 1, 1, 2, 3)


def test_Collection_with_Dipole():
    """Simple test of Dipole in Collection"""
    src = mpj.misc.Dipole(moment=(1, 2, 3), position=(1, 2, 3))
    col = mpj.Collection(src)
    sens = mpj.Sensor()

    B = mpj.getB(col, sens)
    Btest = np.array([3.81801774e-09, 7.63603548e-09, 1.14540532e-08])
    np.testing.assert_allclose(B, Btest)


def test_adding_sources():
    """test if all sources can be added"""
    s1 = mpj.magnet.Cuboid()
    s2 = mpj.magnet.Cylinder()
    s3 = mpj.magnet.CylinderSegment()
    s4 = mpj.magnet.Sphere()
    s5 = mpj.current.Circle()
    s6 = mpj.current.Polyline()
    s7 = mpj.misc.Dipole()
    x1 = mpj.Sensor()
    c1 = mpj.Collection()
    c2 = mpj.Collection()

    for obj in [s1, s2, s3, s4, s5, s6, s7, x1, c1]:
        c2.add(obj)

    strs = ""
    for src in c2:
        strs += str(src)[:3]

    assert strs == "CubCylCylSphCirPolDipSenCol"


def test_set_children_styles():
    """test if styles get applied"""
    src1 = mpj.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    src2 = mpj.magnet.Cylinder(polarization=(1, 2, 3), dimension=(1, 2))
    col = src1 + src2
    col.set_children_styles(magnetization_show=False)
    assert src1.style.magnetization.show is False, "failed updating styles to src1"
    assert src2.style.magnetization.show is False, "failed updating styles to src2"
    with pytest.raises(
        ValueError,
        match="The following style properties are invalid",
    ):
        col.set_children_styles(bad_input="somevalue")


def test_reprs():
    """test repr strings"""
    c = mpj.Collection()
    assert repr(c)[:10] == "Collection"

    s1 = mpj.magnet.Sphere(polarization=(1, 2, 3), diameter=5)
    c = mpj.Collection(s1)
    assert repr(c)[:10] == "Collection"

    x1 = mpj.Sensor()
    c = mpj.Collection(x1)
    assert repr(c)[:10] == "Collection"

    x1 = mpj.Sensor()
    s1 = mpj.magnet.Sphere(polarization=(1, 2, 3), diameter=5)
    c = mpj.Collection(s1, x1)
    assert repr(c)[:10] == "Collection"

    x1 = mpj.magnet.Cuboid(style_label="x1")
    x2 = mpj.magnet.Cuboid(style_label="x2")
    cc = x1 + x2
    rep = cc._repr_html_()
    rep = re.sub("id=[0-9]*[0-9]", "id=REGEX", rep)
    test = "<pre>Collection nolabel (id=REGEX)<br>├── Cuboid x1"
    test += " (id=REGEX)<br>└── Cuboid x2 (id=REGEX)</pre>"
    assert rep == test


def test_collection_describe():
    """test describe method"""

    x = mpj.magnet.Cuboid(style_label="x")
    y = mpj.magnet.Cuboid(style_label="y")
    z = mpj.magnet.Cuboid(style_label="z")
    u = mpj.magnet.Cuboid(style_label="u")
    c = x + y + z + u

    desc = c.describe(format="label, type", return_string=True).split("\n")
    test = [
        "Collection nolabel",
        "├── Collection nolabel",
        "│   ├── Collection nolabel",
        "│   │   ├── Cuboid x",
        "│   │   └── Cuboid y",
        "│   └── Cuboid z",
        "└── Cuboid u",
    ]
    assert test == desc

    desc = c.describe(format="label", return_string=True).split("\n")
    test = [
        "Collection",
        "├── Collection",
        "│   ├── Collection",
        "│   │   ├── x",
        "│   │   └── y",
        "│   └── z",
        "└── u",
    ]
    assert test == desc

    desc = c.describe(format="type", return_string=True).split("\n")
    test = [
        "Collection",
        "├── Collection",
        "│   ├── Collection",
        "│   │   ├── Cuboid",
        "│   │   └── Cuboid",
        "│   └── Cuboid",
        "└── Cuboid",
    ]
    assert test == desc

    desc = c.describe(format="label,type,id", return_string=True).split("\n")
    test = [
        "Collection nolabel (id=REGEX)",
        "├── Collection nolabel (id=REGEX)",
        "│   ├── Collection nolabel (id=REGEX)",
        "│   │   ├── Cuboid x (id=REGEX)",
        "│   │   └── Cuboid y (id=REGEX)",
        "│   └── Cuboid z (id=REGEX)",
        "└── Cuboid u (id=REGEX)",
    ]
    assert "".join(test) == re.sub("id=*[0-9]*[0-9]", "id=REGEX", "".join(desc))

    c = mpj.Collection(*[mpj.magnet.Cuboid() for _ in range(100)])
    c.add(*[mpj.current.Circle() for _ in range(50)])
    c.add(*[mpj.misc.CustomSource() for _ in range(25)])

    desc = c.describe(format="type+label", return_string=True).split("\n")
    test = [
        "Collection nolabel",
        "├── 100x Cuboids",
        "├── 50x Circles",
        "└── 25x CustomSources",
    ]
    assert test == desc

    x = mpj.magnet.Cuboid(style_label="x")
    y = mpj.magnet.Cuboid(style_label="y")
    cc = x + y
    desc = cc.describe(format="label, properties", return_string=True).split("\n")
    test = [
        "Collection",
        "│   • position: [0. 0. 0.] m",
        "│   • orientation: [0. 0. 0.] deg",
        "│   • centroid: [0. 0. 0.]",
        "│   • dipole_moment: [0. 0. 0.]",
        "│   • volume: 0.0",
        "├── x",
        "│       • position: [0. 0. 0.] m",
        "│       • orientation: [0. 0. 0.] deg",
        "│       • dimension: None m",
        "│       • magnetization: None A/m",
        "│       • polarization: None T",
        "│       • centroid: [0. 0. 0.]",
        "│       • dipole_moment: [0. 0. 0.]",
        "│       • meshing: None",
        "│       • volume: 0.0",
        "└── y",
        "        • position: [0. 0. 0.] m",
        "        • orientation: [0. 0. 0.] deg",
        "        • dimension: None m",
        "        • magnetization: None A/m",
        "        • polarization: None T",
        "        • centroid: [0. 0. 0.]",
        "        • dipole_moment: [0. 0. 0.]",
        "        • meshing: None",
        "        • volume: 0.0",
    ]
    assert "".join(test) == re.sub("id=*[0-9]*[0-9]", "id=REGEX", "".join(desc))

    desc = cc.describe()
    assert desc is None


def test_col_getBH_input_format():
    """
    Collections should produce the same BHJM shapes as individual
    sources.
    """
    cube = mpj.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(2, 2, 2),
    )
    coll = mpj.Collection(cube)

    for obs in [(0, 0, 0), [(0, 0, 0)], [[(0, 0, 0)]]]:
        shape1 = cube.getB(obs, squeeze=False).shape
        shape2 = coll.getB(obs, squeeze=False).shape
        assert np.all(shape1 == shape2)


def test_Collection_volume():
    """Test Collection volume calculation (sum of individual magnet volumes)."""

    sphere = mpj.magnet.Sphere(
        diameter=2.0, polarization=(0, 0, 1)
    )
    cuboid = mpj.magnet.Cuboid(
        dimension=(1.0, 2.0, 3.0), polarization=(0, 0, 1)
    )
    cylinder = mpj.magnet.Cylinder(
        dimension=(2.0, 1.0), polarization=(0, 0, 1)
    )

    collection = mpj.Collection(sphere, cuboid, cylinder)

    sphere_vol = (4 / 3) * np.pi * 1.0**3
    cuboid_vol = 1.0 * 2.0 * 3.0
    cylinder_vol = np.pi * 1.0**2 * 1.0

    assert abs(sphere.volume - sphere_vol) < 1e-10
    assert abs(cuboid.volume - cuboid_vol) < 1e-10
    assert abs(cylinder.volume - cylinder_vol) < 1e-10

    calculated = collection.volume
    expected = sphere_vol + cuboid_vol + cylinder_vol
    assert abs(calculated - expected) < 1e-10


def test_Collection_with_zero_volume_objects():
    """Test Collection volume with objects that have zero volume."""

    dipole = mpj.misc.Dipole(moment=(1, 0, 0))
    triangle = mpj.misc.Triangle(
        vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], polarization=(0, 0, 1)
    )
    circle = mpj.current.Circle(current=1.0, diameter=2.0)
    sensor = mpj.Sensor()

    collection = mpj.Collection(dipole, triangle, circle, sensor)

    calculated = collection.volume
    expected = 0
    assert calculated == expected


def test_Collection_mixed_volume():
    """Test Collection volume with mix of volumetric and non-volumetric objects."""

    sphere = mpj.magnet.Sphere(
        diameter=2.0, polarization=(0, 0, 1)
    )
    cuboid = mpj.magnet.Cuboid(
        dimension=(1.0, 2.0, 3.0), polarization=(0, 0, 1)
    )

    dipole = mpj.misc.Dipole(moment=(1, 0, 0))
    triangle = mpj.misc.Triangle(
        vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], polarization=(0, 0, 1)
    )
    sensor = mpj.Sensor()

    collection = mpj.Collection(sphere, cuboid, dipole, triangle, sensor)

    sphere_vol = (4 / 3) * np.pi * 1.0**3
    cuboid_vol = 1.0 * 2.0 * 3.0
    expected = sphere_vol + cuboid_vol

    calculated = collection.volume
    assert abs(calculated - expected) < 1e-10


def test_Collection_centroid_empty():
    """Test empty Collection centroid - should return position"""
    expected = (13, 14, 15)
    empty_col = mpj.Collection(position=expected)
    assert np.allclose(empty_col.centroid, expected)


def test_Collection_centroid_with_objects():
    """Test Collection centroid with objects - volume-weighted centroid"""
    expected = (0, 0, 1)
    obj1 = mpj.magnet.Cuboid(
        dimension=(1, 1, 1), polarization=(0, 0, 1), position=(0, 0, 0)
    )
    obj2 = mpj.magnet.Cuboid(
        dimension=(1, 1, 1), polarization=(0, 0, 1), position=(0, 0, 2)
    )

    col = mpj.Collection(obj1, obj2, position=(1, 1, 1))
    assert np.allclose(col.centroid, expected)


def test_Collection_centroid_zero_volume():
    """Test Collection centroid with zero-volume objects"""
    expected = (1, 1, 1)
    obj1 = mpj.current.Circle(diameter=1, current=1, position=(0, 0, 0))
    obj2 = mpj.misc.Dipole(moment=(1, 0, 0), position=(2, 0, 0))

    col = mpj.Collection(obj1, obj2, position=expected)
    assert np.allclose(col.centroid, expected)


def test_Collection_centroid_path_object():
    """Test object with path - centroid should match position path"""
    expected = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    path_obj = mpj.magnet.Cuboid(
        dimension=(1, 1, 1), polarization=(0, 0, 1), position=expected
    )
    assert np.allclose(path_obj.centroid, expected)
