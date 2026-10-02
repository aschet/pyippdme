# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""``Tool.Id()`` XML payload (Annex G, referenced by 6.10.3 ``Id()``).

The schema (``src/pyippdme/schemas/Tool_ID.xsd``) models a tool description
document rooted at ``<toolID>``, wrapping exactly one of six concrete tool
types: two tactile (``toolTactileTouchTrigger``, ``toolTactileMeasuring``,
both plain extensions of the abstract ``tool``/``toolTactile`` base with no
extra fields) and four optical (``toolOptical1D``, ``toolOptical2D-rs``,
``toolOptical2D-rt``, ``toolOptical3D``, each adding livestream/binary-socket
support flags, data-acquisition function descriptions, and a region of
interest appropriate to their dimensionality).

Region-of-interest elements (``line``, ``polygon2D``, ``circle``,
``circleRel``, ``polygon3D``, ``sphere``, ``sphereRel``, ``cylinder``,
``cylinderRel``) all extend the abstract ``ROI`` type without adding fields
of their own in this schema, so a single :class:`Roi` models all nine, with
``kind`` recording which element name wrapped it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as ET  # nosec B405

from defusedxml.ElementTree import fromstring as defused_fromstring

from pyippdme.exceptions import IppDmeProtocolError

#: The 16 enumerated ``basicfunction`` values (Annex G).
BASIC_FUNCTIONS = (
    "GoTo",
    "PtMeas",
    "ScanOnPlane",
    "ScanOnLine",
    "ScanOnCircle",
    "ScanOnHelix",
    "ScanOnCurve",
    "PTMeasSelfCenter",
    "PTMeasSelfCenterLocked",
    "DataAcquire",
    "DeleteAcquisition",
    "DeleteAllAcquisition",
    "FeatureExtract",
    "GetRawDataBin",
    "GetRawDataShaMem",
    "ReleaseShaMem",
)
#: The 7 enumerated ``cnc_axis`` values (Annex G).
CNC_AXES = ("X", "Y", "Z", "A", "B", "C", "R")
#: The 9 concrete ROI element names a ``<roi>`` may wrap.
ROI_KINDS = (
    "line",
    "polygon2D",
    "circle",
    "circleRel",
    "polygon3D",
    "sphere",
    "sphereRel",
    "cylinder",
    "cylinderRel",
)


@dataclass(frozen=True, slots=True)
class FixedAlignMode:
    """``<alignmode_fixed/>``: the tool's orientation cannot be changed."""


@dataclass(frozen=True, slots=True)
class IndexedAlignMode:
    """``<alignmode_indexed>``: orientation changes in discrete steps."""

    aligncaa: bool
    step: float


@dataclass(frozen=True, slots=True)
class ContinuousAlignMode:
    """``<alignmode_continuous>``: orientation changes continuously."""

    aligncaa: bool


AlignMode = FixedAlignMode | IndexedAlignMode | ContinuousAlignMode


@dataclass(frozen=True, slots=True)
class DataAcquisitionFunction:
    """``dataAcquisition``: the field-of-view size of a data-acquisition function."""

    size_x: float | None = None
    size_y: float | None = None
    size_z: float | None = None


@dataclass(frozen=True, slots=True)
class DataAcquisitionFunctions:
    """Which of ``SingleShot``/``MultiShot``/``Sweep`` acquisition an optical tool supports."""

    single_shot: DataAcquisitionFunction | None = None
    multi_shot: DataAcquisitionFunction | None = None
    sweep: DataAcquisitionFunction | None = None


@dataclass(frozen=True, slots=True)
class Roi:
    """A region of interest; see the module docstring for ``kind``."""

    kind: str
    name: str
    roi_type: str
    include: bool

    def __post_init__(self) -> None:
        if self.kind not in ROI_KINDS:
            raise ValueError(f"Unknown ROI kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class ToolId:
    """Fields common to every tool type (the abstract ``tool`` complex type)."""

    id: str
    basic_functions: tuple[str, ...]
    cnc_axes: tuple[str, ...]
    supports_optimization_mode: bool
    align_mode: AlignMode
    move_on_acquisition: bool


@dataclass(frozen=True, slots=True)
class ToolIdTactileTouchTrigger(ToolId):
    """A tactile ``TouchTrigger`` tool (``toolTactileTouchTrigger``); adds no fields."""


@dataclass(frozen=True, slots=True)
class ToolIdTactileMeasuring(ToolId):
    """A tactile measuring (analog/scanning) tool (``toolTactileMeasuring``); adds no fields."""


@dataclass(frozen=True, slots=True)
class ToolIdOptical(ToolId):
    """Fields common to every optical tool type."""

    supports_livestream: bool = False
    supports_binary_socket: bool = False
    daq_functions: DataAcquisitionFunctions = field(default_factory=DataAcquisitionFunctions)


@dataclass(frozen=True, slots=True)
class ToolIdOptical1D(ToolIdOptical):
    """A 1D optical tool (``toolOptical1D``), e.g. a laser line/point sensor."""

    roi: Roi | None = None


@dataclass(frozen=True, slots=True)
class ToolIdOptical2DRs(ToolIdOptical):
    """A 2D optical tool with rectangular sampling (``toolOptical2D-rs``)."""

    roi: Roi | None = None


@dataclass(frozen=True, slots=True)
class ToolIdOptical2DRt(ToolIdOptical):
    """A 2D optical tool with rotational sampling (``toolOptical2D-rt``)."""

    roi: Roi | None = None


@dataclass(frozen=True, slots=True)
class ToolIdOptical3D(ToolIdOptical):
    """A 3D optical tool (``toolOptical3D``), e.g. a structured-light/laser scanner."""

    roi: Roi | None = None


#: The concrete optical types that carry an ``roi`` field (``ToolIdOptical`` itself does not).
_OpticalWithRoi = ToolIdOptical1D | ToolIdOptical2DRs | ToolIdOptical2DRt | ToolIdOptical3D

_ELEMENT_NAMES: dict[type[ToolId], str] = {
    ToolIdTactileTouchTrigger: "toolTactileTouchTrigger",
    ToolIdTactileMeasuring: "toolTactileMeasuring",
    ToolIdOptical1D: "toolOptical1D",
    ToolIdOptical2DRs: "toolOptical2D-rs",
    ToolIdOptical2DRt: "toolOptical2D-rt",
    ToolIdOptical3D: "toolOptical3D",
}
_ELEMENT_TYPES = {name: cls for cls, name in _ELEMENT_NAMES.items()}


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _parse_bool(text: str | None) -> bool:
    return text == "true"


def to_xml(tool: ToolId) -> str:
    """Serialize ``tool`` as a ``<toolID>`` document (Annex G)."""
    root = ET.Element("toolID")
    element_name = _ELEMENT_NAMES[type(tool)]
    child = ET.SubElement(root, element_name)
    ET.SubElement(child, "id").text = tool.id
    basicfunctions = ET.SubElement(child, "basicfunctions")
    for fn in tool.basic_functions:
        ET.SubElement(basicfunctions, "basicfunction").text = fn
    cnc_axes = ET.SubElement(child, "cnc_axes")
    for axis in tool.cnc_axes:
        ET.SubElement(cnc_axes, "cnc_axis").text = axis
    ET.SubElement(child, "supports_optimization_mode").text = _bool(tool.supports_optimization_mode)
    _write_align_mode(child, tool.align_mode)
    ET.SubElement(child, "moveOnAcquisition").text = _bool(tool.move_on_acquisition)
    if isinstance(tool, ToolIdOptical):
        ET.SubElement(child, "supports_livestream").text = _bool(tool.supports_livestream)
        ET.SubElement(child, "supports_binary_socket").text = _bool(tool.supports_binary_socket)
        _write_daq_functions(child, tool.daq_functions)
    if isinstance(tool, _OpticalWithRoi):
        _write_roi(child, tool.roi)
    return ET.tostring(root, encoding="unicode")


def _write_align_mode(parent: ET.Element, align_mode: AlignMode) -> None:
    if isinstance(align_mode, FixedAlignMode):
        ET.SubElement(parent, "alignmode_fixed")
    elif isinstance(align_mode, IndexedAlignMode):
        el = ET.SubElement(parent, "alignmode_indexed")
        ET.SubElement(el, "aligncaa").text = _bool(align_mode.aligncaa)
        ET.SubElement(el, "step").text = repr(align_mode.step)
    else:
        el = ET.SubElement(parent, "alignmode_continuous")
        ET.SubElement(el, "aligncaa").text = _bool(align_mode.aligncaa)


def _write_daq_function(parent: ET.Element, tag: str, fn: DataAcquisitionFunction | None) -> None:
    if fn is None:
        return
    el = ET.SubElement(parent, tag)
    if fn.size_x is not None:
        ET.SubElement(el, "SizeX").text = repr(fn.size_x)
    if fn.size_y is not None:
        ET.SubElement(el, "SizeY").text = repr(fn.size_y)
    if fn.size_z is not None:
        ET.SubElement(el, "SizeZ").text = repr(fn.size_z)


def _write_daq_functions(parent: ET.Element, functions: DataAcquisitionFunctions) -> None:
    el = ET.SubElement(parent, "dataAcquisitionFunctions")
    _write_daq_function(el, "daqfunctionSingleShot", functions.single_shot)
    _write_daq_function(el, "daqfunctionMultiShot", functions.multi_shot)
    _write_daq_function(el, "daqfunctionSweep", functions.sweep)


def _write_roi(parent: ET.Element, roi: Roi | None) -> None:
    if roi is None:
        return
    el = ET.SubElement(parent, "roi")
    shape = ET.SubElement(el, roi.kind)
    ET.SubElement(shape, "name").text = roi.name
    ET.SubElement(shape, "roiType").text = roi.roi_type
    ET.SubElement(shape, "include").text = _bool(roi.include)


def from_xml(text: str) -> ToolId:
    """Parse a ``<toolID>`` document (Annex G) back into a :class:`ToolId`.

    May carry data from an untrusted remote peer, so uses ``defusedxml``.
    """
    try:
        root = defused_fromstring(text)
    except (ET.ParseError, ValueError) as exc:
        raise IppDmeProtocolError(f"Invalid Tool.Id() XML: {exc}") from exc
    if root.tag != "toolID" or len(root) != 1:
        raise IppDmeProtocolError("Expected a single <toolID> root with one tool child")
    child = root[0]
    cls = _ELEMENT_TYPES.get(child.tag)
    if cls is None:
        raise IppDmeProtocolError(f"Unknown tool element <{child.tag}>")

    def text_of(tag: str) -> str | None:
        el = child.find(tag)
        return el.text if el is not None else None

    tool_id = text_of("id") or ""
    basicfunctions_el = child.find("basicfunctions")
    basic_functions = (
        tuple(el.text or "" for el in basicfunctions_el.findall("basicfunction"))
        if basicfunctions_el is not None
        else ()
    )
    cnc_axes_el = child.find("cnc_axes")
    cnc_axes = (
        tuple(el.text or "" for el in cnc_axes_el.findall("cnc_axis"))
        if cnc_axes_el is not None
        else ()
    )
    supports_optimization_mode = _parse_bool(text_of("supports_optimization_mode"))
    align_mode = _parse_align_mode(child)
    move_on_acquisition = _parse_bool(text_of("moveOnAcquisition"))

    if cls is ToolIdTactileTouchTrigger or cls is ToolIdTactileMeasuring:
        return cls(
            id=tool_id,
            basic_functions=basic_functions,
            cnc_axes=cnc_axes,
            supports_optimization_mode=supports_optimization_mode,
            align_mode=align_mode,
            move_on_acquisition=move_on_acquisition,
        )
    assert cls in (  # noqa: S101 (exhaustiveness check over _ELEMENT_TYPES' known values)
        ToolIdOptical1D,
        ToolIdOptical2DRs,
        ToolIdOptical2DRt,
        ToolIdOptical3D,
    )
    return cls(
        id=tool_id,
        basic_functions=basic_functions,
        cnc_axes=cnc_axes,
        supports_optimization_mode=supports_optimization_mode,
        align_mode=align_mode,
        move_on_acquisition=move_on_acquisition,
        supports_livestream=_parse_bool(text_of("supports_livestream")),
        supports_binary_socket=_parse_bool(text_of("supports_binary_socket")),
        daq_functions=_parse_daq_functions(child),
        roi=_parse_roi(child),
    )


def _parse_align_mode(parent: ET.Element) -> AlignMode:
    if parent.find("alignmode_fixed") is not None:
        return FixedAlignMode()
    indexed = parent.find("alignmode_indexed")
    if indexed is not None:
        aligncaa = _parse_bool(indexed.findtext("aligncaa"))
        step = float(indexed.findtext("step") or "0")
        return IndexedAlignMode(aligncaa, step)
    continuous = parent.find("alignmode_continuous")
    if continuous is not None:
        return ContinuousAlignMode(_parse_bool(continuous.findtext("aligncaa")))
    raise IppDmeProtocolError("Missing alignmode_fixed/indexed/continuous element")


def _parse_daq_function(parent: ET.Element, tag: str) -> DataAcquisitionFunction | None:
    el = parent.find(tag)
    if el is None:
        return None
    size_x = el.findtext("SizeX")
    size_y = el.findtext("SizeY")
    size_z = el.findtext("SizeZ")
    return DataAcquisitionFunction(
        float(size_x) if size_x is not None else None,
        float(size_y) if size_y is not None else None,
        float(size_z) if size_z is not None else None,
    )


def _parse_daq_functions(parent: ET.Element) -> DataAcquisitionFunctions:
    el = parent.find("dataAcquisitionFunctions")
    if el is None:
        return DataAcquisitionFunctions()
    return DataAcquisitionFunctions(
        single_shot=_parse_daq_function(el, "daqfunctionSingleShot"),
        multi_shot=_parse_daq_function(el, "daqfunctionMultiShot"),
        sweep=_parse_daq_function(el, "daqfunctionSweep"),
    )


def _parse_roi(parent: ET.Element) -> Roi | None:
    roi_el = parent.find("roi")
    if roi_el is None or len(roi_el) == 0:
        return None
    shape = roi_el[0]
    return Roi(
        kind=shape.tag,
        name=shape.findtext("name") or "",
        roi_type=shape.findtext("roiType") or "",
        include=_parse_bool(shape.findtext("include")),
    )
