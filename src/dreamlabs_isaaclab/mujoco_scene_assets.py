"""Dream's MuJoCo YAM scene, as USD assets for Isaac Lab.

Dream's ``yam_mujoco`` example puts a bowl, an apple, a banana, an orange, a water bottle and a plate on a table in front
of two YAM arms. This module rebuilds those six objects as USD, one YAMLab-style asset folder each::

    <out>/<Name>/<name>/<name>.usd        rigid body, mass, collision, visual
                       asset_size.json    {"size", "min", "max"} in metres, like the gear assets
                       textures/...       the fruit and glaze textures

Every object's origin is the centre of its footprint at its lowest point, +Z up, metres, so a task can rest it on a
table at the table's height. The shapes are the example's own: the two fruit are its scanned meshes, the bowl and the
plate are its lathed profiles, the orange, bottle and their dimensions and masses are its primitives.

    python -m dreamlabs_isaaclab.mujoco_scene_assets [--out DIR] [--source DIR]

Needs ``pxr`` (Isaac Sim's Python has it, no simulator is started) and numpy. The fruit are YCB scans under CC BY 4.0;
they are converted on this machine from the Dream example you installed, not redistributed (see ATTRIBUTION below).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path.home() / ".dreamlabs-isaaclab/assets/yam_mujoco_scene"

#: Where each object sits in Dream's MuJoCo scene (world frame, metres) and its table-top height. The table top is at
#: z = 0.76; each ``pos`` is the object body's origin there (a body origin is its centre, not its bottom).
TABLE_TOP_Z = 0.76
MUJOCO_LAYOUT = {
    "apple": (0.15, 0.105, 0.792),
    "banana": (-0.115, 0.085, 0.79),
    "orange": (0.0, 0.13, 0.794),
    "water_bottle": (0.22, -0.11, 0.845),
    "bowl": (0.0, -0.04, 0.76),
    "plate": (-0.22, -0.045, 0.764),
}

ATTRIBUTION = """The apple and banana meshes and textures are from the YCB Object and Model Set (Calli, Singh, Walsman,
Srinivasa, Abbeel, Dollar), Google 16k scans, licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/),
as recentred, rescaled and resized by Dream Labs' yam_mujoco example and converted here to USD. No endorsement by the
dataset authors is implied. The robot and scene code of that example is MIT licensed (I2RT Robotics).
"""


@dataclass
class Shape:
    """One mesh: points (N, 3), triangles (M, 3), optional per-face-vertex uv (M, 3, 2) and normals (M, 3, 3)."""

    points: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None = None
    normals: np.ndarray | None = None


# ---- geometry ---------------------------------------------------------------------------------------------------------


def lathe(profile: list[tuple[float, float]], count: int = 96) -> Shape:
    """A surface of revolution around Z from a (radius, z) profile; u runs around, v along the profile."""
    points = np.array([[r * math.cos(2 * math.pi * i / count), r * math.sin(2 * math.pi * i / count), z]
                       for r, z in profile for i in range(count)])
    faces, uv = [], []
    rows = len(profile)
    for j in range(rows - 1):
        for i in range(count):
            a, b = j * count + i, j * count + (i + 1) % count
            faces += [[a, b, b + count], [a, b + count, a + count]]
            u0, u1, v0, v1 = i / count, (i + 1) / count, j / (rows - 1), (j + 1) / (rows - 1)
            uv += [[[u0, v0], [u1, v0], [u1, v1]], [[u0, v0], [u1, v1], [u0, v1]]]
    return Shape(points, np.array(faces), np.array(uv))


def ellipsoid(rx: float, ry: float, rz: float, lat: int = 24, lon: int = 48) -> Shape:
    profile = [(math.sin(math.pi * k / lat), -math.cos(math.pi * k / lat)) for k in range(lat + 1)]
    shape = lathe([(r, z) for r, z in profile], lon)
    shape.points = shape.points * np.array([rx, ry, rz])
    return shape


def torus(major: float, minor: float, z: float, count: int = 96, ring: int = 12) -> Shape:
    profile = [(major + minor * math.cos(2 * math.pi * k / ring), z + minor * math.sin(2 * math.pi * k / ring))
               for k in range(ring + 1)]
    return lathe(profile, count)


def read_obj(path: Path) -> Shape:
    """A textured OBJ (v, vt, vn, f with v/vt/vn) as a triangle mesh with face-varying uv and normals."""
    v, vt, vn, faces = [], [], [], []
    for line in path.read_text().splitlines():
        head, _, rest = line.partition(" ")
        if head == "v":
            v.append([float(x) for x in rest.split()[:3]])
        elif head == "vt":
            vt.append([float(x) for x in rest.split()[:2]])
        elif head == "vn":
            vn.append([float(x) for x in rest.split()[:3]])
        elif head == "f":
            corner = [tuple(int(i) - 1 if i else -1 for i in (c.split("/") + ["", ""])[:3]) for c in rest.split()]
            for k in range(1, len(corner) - 1):
                faces.append((corner[0], corner[k], corner[k + 1]))
    f = np.array([[c[0] for c in tri] for tri in faces])
    uv = np.array([[vt[c[1]] for c in tri] for tri in faces]) if vt else None
    normals = np.array([[vn[c[2]] for c in tri] for tri in faces]) if vn else None
    return Shape(np.array(v), f, uv, normals)


def bounds(shapes: list[Shape]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.concatenate([s.points for s in shapes])
    return pts.min(axis=0), pts.max(axis=0)


def ground(shapes: list[Shape]) -> list[Shape]:
    """Move shapes so the footprint centre of the whole object is at x = y = 0 and its lowest point at z = 0."""
    low, high = bounds(shapes)
    shift = np.array([-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]])
    for s in shapes:
        s.points = s.points + shift
    return shapes


# ---- USD --------------------------------------------------------------------------------------------------------------


def write_usd(path: Path, name: str, parts: list[dict], mass: float) -> dict:
    """Author one object. ``parts``: dicts with ``shape``, ``collision`` (None, 'convexHull', 'convexDecomposition'),
    and ``color`` (rgb) or ``texture`` (a file under textures/)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))

    looks = UsdGeom.Scope.Define(stage, f"/{name}/Looks")  # noqa: F841 - groups the materials
    for index, part in enumerate(parts):
        shape: Shape = part["shape"]
        mesh = UsdGeom.Mesh.Define(stage, f"/{name}/part{index}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(shape.points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(shape.faces), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(shape.faces.reshape(-1).astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        if shape.normals is not None:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(shape.normals.reshape(-1, 3).astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        else:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(_vertex_normals(shape)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        primvars = UsdGeom.PrimvarsAPI(mesh)
        if shape.uv is not None:
            st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            st.Set(Vt.Vec2fArray.FromNumpy(shape.uv.reshape(-1, 2).astype(np.float32)))

        material = UsdShade.Material.Define(stage, f"/{name}/Looks/mat{index}")
        shader = UsdShade.Shader.Define(stage, f"/{name}/Looks/mat{index}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(part.get("roughness", 0.55)))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        if part.get("texture") and shape.uv is not None:
            reader = UsdShade.Shader.Define(stage, f"/{name}/Looks/mat{index}/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            texture = UsdShade.Shader.Define(stage, f"/{name}/Looks/mat{index}/tex")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{part['texture']}")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*part.get("color", (0.7, 0.7, 0.7))))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

        if part.get("collision"):
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(part["collision"])
    stage.GetRootLayer().Save()
    low, high = bounds([p["shape"] for p in parts])
    return {"size": dict(zip("xyz", (high - low).tolist())), "min": dict(zip("xyz", low.tolist())),
            "max": dict(zip("xyz", high.tolist()))}


def _vertex_normals(shape: Shape) -> np.ndarray:
    normals = np.zeros_like(shape.points)
    tri = shape.points[shape.faces]
    face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(normals, shape.faces[:, k], face_normal)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.where(length == 0, 1, length)).astype(np.float32)


# ---- the six objects --------------------------------------------------------------------------------------------------

BOWL_PROFILE = [(0, 0.003), (0.029, 0.003), (0.036, 0.006), (0.045, 0.011), (0.055, 0.018), (0.065, 0.027),
                (0.074, 0.038), (0.081, 0.050), (0.083, 0.054), (0.082, 0.057), (0.079, 0.058), (0.076, 0.055),
                (0.073, 0.047), (0.067, 0.037), (0.058, 0.027), (0.048, 0.019), (0.036, 0.012), (0.025, 0.010),
                (0, 0.010)]


def build(name: str, source: Path) -> tuple[list[dict], float, list[str]]:
    """The parts, the mass in kg and the texture files an object needs, from the example's own numbers."""
    fruits = source / "fruits"
    if name in ("apple", "banana"):
        shape = read_obj(fruits / name / "visual.obj")
        collision = "convexHull" if name == "apple" else "convexDecomposition"
        mass = 0.12 if name == "apple" else 16 * 0.0028125
        ground([shape])
        return [{"shape": shape, "collision": collision, "texture": f"{name}.png", "roughness": 0.7}], mass, [f"fruits/{name}/texture_map.png"]
    if name == "bowl":
        shape = lathe(BOWL_PROFILE, 96)
        ground([shape])
        return [{"shape": shape, "collision": "convexDecomposition", "texture": "brown_glaze.png", "roughness": 0.25}], 2.0, ["brown_glaze.png"]
    if name == "orange":
        body = ellipsoid(0.032, 0.032, 0.030)
        ground([body])
        return [{"shape": body, "collision": "convexHull", "color": (0.92, 0.33, 0.025), "roughness": 0.6}], 0.11, []
    if name == "water_bottle":
        profile = [(0, -0.075), (0.0295, -0.075), (0.0295, -0.05), (0.029, -0.05), (0.029, 0.074)]
        body = lathe(profile, 64)
        shoulder = ellipsoid(0.029, 0.029, 0.020)
        shoulder.points = shoulder.points + np.array([0, 0, 0.074])
        neck = lathe([(0.013, 0.085), (0.013, 0.109), (0, 0.109)], 48)
        parts = [body, shoulder, neck]
        ground(parts)
        blue, cap = (0.36, 0.58, 0.65), (0.12, 0.24, 0.30)
        return [{"shape": body, "collision": "convexHull", "color": blue, "roughness": 0.3},
                {"shape": shoulder, "collision": "convexHull", "color": blue, "roughness": 0.3},
                {"shape": neck, "collision": "convexHull", "color": cap, "roughness": 0.4}], 0.276, []
    if name == "plate":
        disk = lathe([(0, -0.004), (0.069, -0.004), (0.069, 0.004), (0, 0.004)], 96)
        rim = torus(0.068, 0.005, 0.004)
        ground([disk, rim])
        return [{"shape": disk, "collision": "convexHull", "texture": "blue_glaze.png", "roughness": 0.25},
                {"shape": rim, "collision": None, "texture": "blue_glaze.png", "roughness": 0.25}], 0.30, ["blue_glaze.png"]
    raise KeyError(name)


FOLDERS = {"apple": "Apple", "banana": "Banana", "bowl": "Bowl", "orange": "Orange", "water_bottle": "WaterBottle", "plate": "Plate"}


def yam_asset_args(root: Path = DEFAULT_OUT) -> dict[str, str]:
    """``{object: instance folder}`` for YAMLab's PutAppleInBowl-v0, as ``--asset name=path`` takes them."""
    out = {name: str(root / folder / f"{name}_000") for name, folder in FOLDERS.items()}
    missing = [n for n, d in out.items() if not (Path(d) / f"{n}_000.usd").exists()]
    if missing:
        raise SystemExit(f"Converted assets missing for {missing} in {root}: run python -m dreamlabs_isaaclab.mujoco_scene_assets")
    return out


def find_source() -> Path:
    """The ``assets`` folder of the installed Dream yam_mujoco example."""
    candidates = sorted((Path.home() / "dream-env/lib").glob("python3.*/site-packages/dream_examples/yam_mujoco/assets"))
    if not candidates:
        raise SystemExit("Dream's yam_mujoco example is not installed: pip install dreamlabs in ~/dream-env, or pass --source.")
    return candidates[0]


def convert_all(out: Path, source: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION)
    report = {}
    for name, folder in FOLDERS.items():
        parts, mass, textures = build(name, source)
        instance = f"{name}_000"
        target = out / folder / instance
        (target / "textures").mkdir(parents=True, exist_ok=True)
        for texture in textures:
            src = source / texture
            shutil.copyfile(src, target / "textures" / (f"{name}.png" if "fruits/" in texture else src.name))
        size = write_usd(target / f"{instance}.usd", name, parts, mass)
        (target / "asset_size.json").write_text(json.dumps(size, indent=2))
        report[name] = {"usd": str(target / f"{instance}.usd"), "mass_kg": mass,
                        "size_cm": [round(v * 100, 1) for v in (size["size"]["x"], size["size"]["y"], size["size"]["z"])]}
    layout = {k: {"mujoco_world_pos": v, "above_table_top_m": round(v[2] - TABLE_TOP_Z, 4)} for k, v in MUJOCO_LAYOUT.items()}
    (out / "scene_layout.json").write_text(json.dumps({"table_top_z": TABLE_TOP_Z, "objects": layout}, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=Path, default=None, help="The example's assets folder (default: the installed one)")
    args = parser.parse_args(argv)
    report = convert_all(args.out.expanduser(), (args.source or find_source()).expanduser())
    for name, info in report.items():
        print(f"{name:13s} {info['size_cm']} cm  {info['mass_kg']:.3f} kg  {info['usd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
