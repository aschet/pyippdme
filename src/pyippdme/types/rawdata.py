# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""``Tool.AdvDataStruct()`` XML payload (Annex F, referenced by 6.17.2).

Describes how a server will hand over raw sensor data: through which
technology (``SocBin`` a binary socket, ``ShaMem`` shared memory, ``File`` a
file on disk - one or more may be offered at once) and in which format,
either a generic structured sample (``Gen``) or one of the simple, opaque
formats (``BitMap``/``Jpeg``/``SBF``/``ESBF``/``XML``).
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET  # nosec B405

from defusedxml.ElementTree import fromstring as defused_fromstring

from pyippdme.exceptions import IppDmeProtocolError

#: The technologies through which raw data may be retrieved (6.17.2).
RETURN_TECHNOLOGIES = ("SocBin", "ShaMem", "File")
#: The format kinds that carry no further structure of their own.
SIMPLE_FORMATS = ("BitMap", "Jpeg", "SBF", "ESBF", "XML")


@dataclass(frozen=True, slots=True)
class Vector:
    """A 3D vector as used in ``Tool.AdvDataStruct()`` (Annex F)."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class PosInformation:
    """Which position/orientation vectors accompany each sample."""

    tcp: Vector
    ijk: Vector | None = None
    lmn: Vector | None = None


@dataclass(frozen=True, slots=True)
class PixelStructure:
    """How each sample's color/intensity channels are encoded."""

    bits_per_color_channel: int
    num_color_channels: int


@dataclass(frozen=True, slots=True)
class DimensionSample:
    """One dimension's ``SpanWP``/``NoPixelOfDim`` pair of a :class:`SampleStructure`."""

    span_wp: float
    no_pixel_of_dim: int


@dataclass(frozen=True, slots=True)
class SampleStructure:
    """The dimensionality and per-dimension extent of a generic sample."""

    dimensions: tuple[DimensionSample, ...]

    @property
    def no_of_dims(self) -> int:
        return len(self.dimensions)


@dataclass(frozen=True, slots=True)
class GenFormat:
    """The generic, structured sample format (as opposed to an opaque image/binary format)."""

    pos_information: PosInformation
    pixel_structure: PixelStructure
    sample_structure: SampleStructure


#: Either the structured ``Gen`` format, or one of :data:`SIMPLE_FORMATS` by name.
Format = GenFormat | str


@dataclass(frozen=True, slots=True)
class AdvDataStruct:
    """How a server will hand over raw sensor data (``Tool.AdvDataStruct()``)."""

    return_technologies: frozenset[str]
    format: Format

    def __post_init__(self) -> None:
        if not self.return_technologies or not self.return_technologies <= set(RETURN_TECHNOLOGIES):
            raise ValueError(
                f"return_technologies must be a non-empty subset of {RETURN_TECHNOLOGIES}"
            )
        if isinstance(self.format, str) and self.format not in SIMPLE_FORMATS:
            raise ValueError(
                f"format {self.format!r} must be one of {SIMPLE_FORMATS} or a GenFormat"
            )


def to_xml(data: AdvDataStruct) -> str:
    root = ET.Element("Tool")
    struct_el = ET.SubElement(root, "AdvDataStruct")
    tech_el = ET.SubElement(struct_el, "ReturnTechnology")
    for tech in RETURN_TECHNOLOGIES:
        if tech in data.return_technologies:
            ET.SubElement(tech_el, tech)
    format_el = ET.SubElement(struct_el, "Format")
    if isinstance(data.format, str):
        ET.SubElement(format_el, data.format)
    else:
        _write_gen_format(format_el, data.format)
    return ET.tostring(root, encoding="unicode")


def _write_vector(parent: ET.Element, tag: str, vector: Vector) -> None:
    el = ET.SubElement(parent, tag)
    ET.SubElement(el, "X").text = repr(vector.x)
    ET.SubElement(el, "Y").text = repr(vector.y)
    ET.SubElement(el, "Z").text = repr(vector.z)


def _write_gen_format(parent: ET.Element, gen: GenFormat) -> None:
    gen_el = ET.SubElement(parent, "Gen")
    pos_el = ET.SubElement(gen_el, "PosInformation")
    _write_vector(pos_el, "TCP", gen.pos_information.tcp)
    if gen.pos_information.ijk is not None:
        _write_vector(pos_el, "IJK", gen.pos_information.ijk)
    if gen.pos_information.lmn is not None:
        _write_vector(pos_el, "LMN", gen.pos_information.lmn)
    pixel_el = ET.SubElement(gen_el, "PixelStructure")
    ET.SubElement(pixel_el, "BitsPerColCha").text = str(gen.pixel_structure.bits_per_color_channel)
    ET.SubElement(pixel_el, "NoColCha").text = str(gen.pixel_structure.num_color_channels)
    sample_el = ET.SubElement(gen_el, "SampleStructure")
    sample_el.set("NoOfDims", str(gen.sample_structure.no_of_dims))
    for dim in gen.sample_structure.dimensions:
        ET.SubElement(sample_el, "SpanWP").text = repr(dim.span_wp)
        ET.SubElement(sample_el, "NoPixelOfDim").text = str(dim.no_pixel_of_dim)


def from_xml(text: str) -> AdvDataStruct:
    """Parse ``text``; may carry data from an untrusted remote peer, so uses ``defusedxml``."""
    try:
        root = defused_fromstring(text)
    except (ET.ParseError, ValueError) as exc:
        raise IppDmeProtocolError(f"Invalid AdvDataStruct XML: {exc}") from exc
    struct_el = root.find("AdvDataStruct")
    if root.tag != "Tool" or struct_el is None:
        raise IppDmeProtocolError("Expected <Tool><AdvDataStruct>...")
    tech_el = struct_el.find("ReturnTechnology")
    technologies = frozenset(child.tag for child in tech_el) if tech_el is not None else frozenset()
    format_el = struct_el.find("Format")
    if format_el is None or len(format_el) == 0:
        raise IppDmeProtocolError("Missing <Format> element")
    format_child = format_el[0]
    fmt: Format = _read_gen_format(format_child) if format_child.tag == "Gen" else format_child.tag
    return AdvDataStruct(technologies, fmt)


def _read_vector(element: ET.Element) -> Vector:
    return Vector(
        float(element.findtext("X") or "0"),
        float(element.findtext("Y") or "0"),
        float(element.findtext("Z") or "0"),
    )


def _read_gen_format(gen_el: ET.Element) -> GenFormat:
    pos_el = gen_el.find("PosInformation")
    if pos_el is None:
        raise IppDmeProtocolError("Missing <PosInformation> element")
    tcp_el = pos_el.find("TCP")
    if tcp_el is None:
        raise IppDmeProtocolError("Missing <TCP> element")
    ijk_el = pos_el.find("IJK")
    lmn_el = pos_el.find("LMN")
    pos_information = PosInformation(
        tcp=_read_vector(tcp_el),
        ijk=_read_vector(ijk_el) if ijk_el is not None else None,
        lmn=_read_vector(lmn_el) if lmn_el is not None else None,
    )
    pixel_el = gen_el.find("PixelStructure")
    if pixel_el is None:
        raise IppDmeProtocolError("Missing <PixelStructure> element")
    pixel_structure = PixelStructure(
        int(pixel_el.findtext("BitsPerColCha") or "0"),
        int(pixel_el.findtext("NoColCha") or "0"),
    )
    sample_el = gen_el.find("SampleStructure")
    if sample_el is None:
        raise IppDmeProtocolError("Missing <SampleStructure> element")
    span_values = [float(el.text or "0") for el in sample_el.findall("SpanWP")]
    pixel_values = [int(el.text or "0") for el in sample_el.findall("NoPixelOfDim")]
    paired = zip(span_values, pixel_values, strict=True)
    dimensions = tuple(DimensionSample(span, pixels) for span, pixels in paired)
    return GenFormat(pos_information, pixel_structure, SampleStructure(dimensions))
