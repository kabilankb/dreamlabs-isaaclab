import json

import numpy as np
import pytest

from dreamlabs_isaaclab import mujoco_scene_assets as m


def test_lathe_has_the_expected_topology():
    shape = m.lathe([(0.0, 0.0), (0.05, 0.0), (0.05, 0.1), (0.0, 0.1)], count=16)
    assert shape.points.shape == (4 * 16, 3)
    assert shape.faces.shape == (3 * 16 * 2, 3) and shape.uv.shape == (3 * 16 * 2, 3, 2)
    assert shape.faces.max() < len(shape.points)
    radii = np.hypot(shape.points[:, 0], shape.points[:, 1])
    assert radii.max() == pytest.approx(0.05)


def test_ellipsoid_has_the_requested_radii():
    shape = m.ellipsoid(0.032, 0.032, 0.030)
    low, high = m.bounds([shape])
    assert high[2] == pytest.approx(0.030) and low[2] == pytest.approx(-0.030)
    assert high[0] == pytest.approx(0.032, abs=1e-3)


def test_ground_puts_the_footprint_centre_on_the_axis_and_the_lowest_point_at_zero():
    shape = m.Shape(np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 7.0], [2.0, 3.0, 6.0]]), np.array([[0, 1, 2]]))
    m.ground([shape])
    low, high = m.bounds([shape])
    assert low[2] == 0.0 and (low[0] + high[0]) / 2 == 0.0 and (low[1] + high[1]) / 2 == 0.0


def test_read_obj_triangulates_and_keeps_uv_and_normals(tmp_path):
    obj = tmp_path / "q.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n"
                   "f 1/1/1 2/2/1 3/3/1 4/4/1\n")
    shape = m.read_obj(obj)
    assert shape.faces.shape == (2, 3) and shape.uv.shape == (2, 3, 2) and shape.normals.shape == (2, 3, 3)


def test_the_mujoco_layout_matches_the_table_and_object_heights():
    layout = {k: v[2] - m.TABLE_TOP_Z for k, v in m.MUJOCO_LAYOUT.items()}
    assert set(layout) == set(m.FOLDERS)
    assert all(0.0 <= h < 0.1 for h in layout.values())  # every body origin is within 10 cm of the table top


def test_every_object_builds_from_fake_sources_with_the_documented_size(tmp_path):
    for name in ("bowl", "orange", "water_bottle", "plate"):
        parts, mass, _ = m.build(name, tmp_path)
        low, high = m.bounds([p["shape"] for p in parts])
        assert low[2] == pytest.approx(0.0, abs=1e-9) and mass > 0
        assert (low[0] + high[0]) / 2 == pytest.approx(0.0, abs=1e-6)
    bowl, _, _ = m.build("bowl", tmp_path)
    low, high = m.bounds([p["shape"] for p in bowl])
    assert high[2] == pytest.approx(0.055, abs=1e-6) and high[0] == pytest.approx(0.083, abs=1e-3)
    bottle, mass, _ = m.build("water_bottle", tmp_path)
    assert m.bounds([p["shape"] for p in bottle])[1][2] == pytest.approx(0.184, abs=1e-3) and mass == pytest.approx(0.276)


def test_write_usd_makes_a_rigid_body_with_collision(tmp_path):
    pxr = pytest.importorskip("pxr")
    parts, mass, _ = m.build("orange", tmp_path)
    size = m.write_usd(tmp_path / "orange.usd", "orange", parts, mass)
    stage = pxr.Usd.Stage.Open(str(tmp_path / "orange.usd"))
    root = stage.GetDefaultPrim()
    assert root.HasAPI(pxr.UsdPhysics.RigidBodyAPI) and root.GetName() == "orange"
    assert pxr.UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(0.11)
    mesh = stage.GetPrimAtPath("/orange/part0")
    assert mesh.HasAPI(pxr.UsdPhysics.CollisionAPI)
    assert pxr.UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexHull"
    assert size["size"]["z"] == pytest.approx(0.06, abs=1e-3)
    json.dumps(size)


def test_the_attribution_names_the_license():
    assert "CC BY 4.0" in m.ATTRIBUTION and "YCB" in m.ATTRIBUTION
