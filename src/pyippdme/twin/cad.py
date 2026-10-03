# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""OpenCASCADE layer: read CAD files, tessellate, ray-cast and measure distances.

Everything that needs the ``OCP`` package (``pip install cadquery-ocp``) is
here, so the rest of :mod:`pyippdme.twin` stays importable without it. The
shapes stay exact B-reps: ``PtMeas`` rays hit the real surface, not its
tessellation, which is only used for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.gp import gp_Ax2, gp_Dir, gp_Lin, gp_Pnt, gp_Trsf
from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESControl import IGESControl_Reader
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

from pyippdme.twin.geometry import Matrix, Mesh
from pyippdme.types.vec3 import Vec3

Shape: TypeAlias = Any  # a TopoDS_Shape; OCP has no type information


def _silence_opencascade() -> None:
    """OpenCASCADE prints transfer statistics to stdout; a library should not."""
    try:
        from OCP.Message import Message, Message_Gravity

        printers = Message.DefaultMessenger_s().Printers()
        for printer in list(printers):
            printer.SetTraceLevel(Message_Gravity.Message_Fail)
    except Exception:  # noqa: S110 # nosec B110 (cosmetic only)
        pass


_silence_opencascade()

SUPPORTED_SUFFIXES = (".step", ".stp", ".iges", ".igs", ".stl", ".brep")


class CadError(Exception):
    """A CAD file could not be read."""


def to_trsf(m: Matrix) -> gp_Trsf:
    t = gp_Trsf()
    t.SetValues(
        m[0, 0], m[0, 1], m[0, 2], m[0, 3],
        m[1, 0], m[1, 1], m[1, 2], m[1, 3],
        m[2, 0], m[2, 1], m[2, 2], m[2, 3],
    )  # fmt: skip
    return t


def moved(shape: Shape, m: Matrix) -> Shape:
    """``shape`` placed by ``m``; cheap (a location change, no geometry copy)."""
    return shape.Moved(TopLoc_Location(to_trsf(m)))


def transformed_copy(shape: Shape, m: Matrix) -> Shape:
    return BRepBuilderAPI_Transform(shape, to_trsf(m), True).Shape()


def bounding_box(shape: Shape) -> tuple[Vec3, Vec3]:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    if box.IsVoid():
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    lo, hi = box.CornerMin(), box.CornerMax()
    return (lo.X(), lo.Y(), lo.Z()), (hi.X(), hi.Y(), hi.Z())


def tessellate(shape: Shape, deflection: float | None = None, angular: float = 0.3) -> Mesh:
    """Triangulate ``shape``; ``deflection`` defaults to 0.05 % of its size (a display mesh).

    ``angular`` is the largest angle between neighbouring normals in radians.
    """
    lo, hi = bounding_box(shape)
    size = max(hi[i] - lo[i] for i in range(3))
    if size <= 0.0:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    BRepMesh_IncrementalMesh(shape, deflection or max(size * 5e-4, 1e-3), False, angular, True)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, location)
        explorer.Next()
        if tri is None:
            continue
        trsf = location.Transformation()
        base = len(vertices)
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            vertices.append((p.X(), p.Y(), p.Z()))
        reverse = face.Orientation() == TopAbs_REVERSED
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            a, b, c = a - 1 + base, b - 1 + base, c - 1 + base
            faces.append((a, c, b) if reverse else (a, b, c))
    return Mesh(np.asarray(vertices, dtype=float).reshape(-1, 3), np.asarray(faces, dtype=np.int64))


@dataclass(slots=True)
class CadBody:
    """One named solid of a CAD file, with the colour the file gives it (0-1 RGB)."""

    name: str
    shape: Shape
    color: tuple[float, float, float] | None = None
    _mesh: Mesh | None = field(default=None, repr=False)

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self._mesh = tessellate(self.shape)
        return self._mesh


def _read_stl(path: Path) -> list[CadBody]:
    shape = TopoDS_Shape()
    if not StlAPI_Reader().Read(shape, str(path)):
        raise CadError(f"cannot read {path}")
    return [CadBody(path.stem, shape)]


def _read_iges(path: Path) -> list[CadBody]:
    reader = IGESControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise CadError(f"cannot read {path}")
    reader.TransferRoots()
    return [CadBody(path.stem, reader.OneShape())]


def _label_name(label: TDF_Label) -> str:
    attribute = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
        return str(attribute.Get().ToExtString())
    return ""


def _read_step(path: Path) -> list[CadBody]:
    """Read a STEP file with its assembly structure: one body per named leaf part."""
    document = TDocStd_Document(TCollection_ExtendedString("pyippdme"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.Transfer(document):
        raise CadError(f"cannot read {path}")
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    roots = TDF_LabelSequence()
    shapes.GetFreeShapes(roots)
    bodies: list[CadBody] = []

    def visit(label: TDF_Label, parent: Matrix | None, prefix: str) -> None:
        shape = shapes.GetShape_s(label)
        name = _label_name(label)
        referred = TDF_Label()
        if shapes.GetReferredShape_s(label, referred):
            name = name or _label_name(referred)
        components = TDF_LabelSequence()
        target = referred if not referred.IsNull() else label
        if shapes.IsAssembly_s(target):
            shapes.GetComponents_s(target, components)
            for i in range(1, components.Length() + 1):
                visit(components.Value(i), parent, f"{prefix}{name}/" if name else prefix)
            return
        color = Quantity_Color()
        rgb = None
        if colors.GetColor(shapes.GetShape_s(target), XCAFDoc_ColorSurf, color) or colors.GetColor(
            shape, XCAFDoc_ColorSurf, color
        ):
            rgb = (color.Red(), color.Green(), color.Blue())
        bodies.append(CadBody(f"{prefix}{name}" or path.stem, shape, rgb))

    for i in range(1, roots.Length() + 1):
        visit(roots.Value(i), None, "")
    if not bodies:
        step = STEPControl_Reader()
        if step.ReadFile(str(path)) != IFSelect_RetDone:
            raise CadError(f"cannot read {path}")
        step.TransferRoots()
        bodies.append(CadBody(path.stem, step.OneShape()))
    return bodies


def _read_brep(path: Path) -> list[CadBody]:
    from OCP.BRep import BRep_Builder

    shape = TopoDS_Shape()
    from OCP.BRepTools import BRepTools

    if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
        raise CadError(f"cannot read {path}")
    return [CadBody(path.stem, shape)]


def load_cad(path: str | Path) -> list[CadBody]:
    """Read a STEP, IGES, STL or BREP file into named bodies (STEP assemblies are split)."""
    p = Path(path)
    if not p.is_file():
        raise CadError(f"no such file: {p}")
    suffix = p.suffix.lower()
    if suffix in (".step", ".stp"):
        return _read_step(p)
    if suffix in (".iges", ".igs"):
        return _read_iges(p)
    if suffix == ".stl":
        return _read_stl(p)
    if suffix == ".brep":
        return _read_brep(p)
    raise CadError(f"unsupported CAD format {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}")


def write_step_assembly(bodies: list[CadBody], path: str | Path) -> None:
    """Write ``bodies`` as one STEP file that keeps each body's name."""
    from OCP.STEPControl import STEPControl_AsIs

    document = TDocStd_Document(TCollection_ExtendedString("pyippdme"))
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    for body in bodies:
        label = shapes.AddShape(body.shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(body.name))
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(document, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise CadError(f"cannot write {path}")


def write_step(bodies: list[CadBody], path: str | Path) -> None:
    """Write ``bodies`` as a STEP file (one solid each, names are not kept by this writer)."""
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    for body in bodies:
        writer.Transfer(body.shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise CadError(f"cannot write {path}")


# -- primitives ------------------------------------------------------------------------


def make_box(dx: float, dy: float, dz: float, origin: Vec3 = (0.0, 0.0, 0.0)) -> Shape:
    return BRepPrimAPI_MakeBox(gp_Pnt(*origin), dx, dy, dz).Shape()


def make_cylinder(
    radius: float, height: float, origin: Vec3 = (0.0, 0.0, 0.0), axis: Vec3 = (0.0, 0.0, 1.0)
) -> Shape:
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*origin), gp_Dir(*axis)), radius, height).Shape()


def rounded_box(
    dx: float, dy: float, dz: float, origin: Vec3 = (0.0, 0.0, 0.0), radius: float = 0.0
) -> Shape:
    """Build a box with all edges rounded by ``radius`` (a plain box if the radius is zero)."""
    box = make_box(dx, dy, dz, origin)
    if radius <= 0.0:
        return box
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.TopAbs import TopAbs_EDGE

    fillet = BRepFilletAPI_MakeFillet(box)
    explorer = TopExp_Explorer(box, TopAbs_EDGE)
    while explorer.More():
        fillet.Add(min(radius, min(dx, dy, dz) / 2.5), TopoDS.Edge(explorer.Current()))
        explorer.Next()
    try:
        fillet.Build()
        return fillet.Shape() if fillet.IsDone() else box
    except Exception:  # a failed fillet only costs the rounding
        return box


def make_sphere(radius: float, center: Vec3 = (0.0, 0.0, 0.0)) -> Shape:
    return BRepPrimAPI_MakeSphere(gp_Pnt(*center), radius).Shape()


def make_compound(shapes: list[Shape]) -> Shape:
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound


def common(a: Shape, b: Shape) -> Shape:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common

    return BRepAlgoAPI_Common(a, b).Shape()


def make_prism_xz(profile: list[tuple[float, float]], width_y: float) -> Shape:
    """Extrude a polygon given in (x, z) along +Y by ``width_y``."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec

    polygon = BRepBuilderAPI_MakePolygon()
    for x, z in profile:
        polygon.Add(gp_Pnt(x, 0.0, z))
    polygon.Close()
    face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0.0, width_y, 0.0)).Shape()


def fuse(a: Shape, b: Shape) -> Shape:
    return BRepAlgoAPI_Fuse(a, b).Shape()


def cut(a: Shape, b: Shape) -> Shape:
    return BRepAlgoAPI_Cut(a, b).Shape()


# -- queries ---------------------------------------------------------------------------


class RayCaster:
    """Exact ray-vs-B-rep intersections for one shape (placed by a pose at query time)."""

    def __init__(self, shape: Shape, tolerance: float = 1e-6) -> None:
        self._shape = shape
        self._intersector = IntCurvesFace_ShapeIntersector()
        self._intersector.Load(shape, tolerance)

    def first_hit(
        self, origin: Vec3, direction: Vec3, max_distance: float = 1e9
    ) -> tuple[Vec3, float] | None:
        """Nearest hit along the (unit) ``direction`` as ``(point, distance)``, else ``None``."""
        self._intersector.Perform(gp_Lin(gp_Pnt(*origin), gp_Dir(*direction)), 0.0, max_distance)
        best: tuple[Vec3, float] | None = None
        for i in range(1, self._intersector.NbPnt() + 1):
            w = self._intersector.WParameter(i)
            if best is None or w < best[1]:
                p = self._intersector.Pnt(i)
                best = ((p.X(), p.Y(), p.Z()), w)
        return best


def min_distance(a: Shape, b: Shape) -> float:
    """Smallest distance between two shapes (0 when they touch or overlap)."""
    extrema = BRepExtrema_DistShapeShape(a, b)
    if not extrema.IsDone():
        return math.inf
    return float(extrema.Value())
