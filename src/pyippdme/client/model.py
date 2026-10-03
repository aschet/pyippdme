# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A typed, high-level client on top of :class:`~pyippdme.client.IppDmeClient`.

:class:`IppDmeClient` is a thin protocol client: ``call("GoTo", NamedValue(...))``
returns a raw tuple of :class:`~pyippdme.protocol.ast.DataPayload` nodes, and
the caller is responsible for knowing the wire shape of every command by
hand. :class:`IppDmeMachine` wraps it with typed Python methods and
dataclass/tuple return values instead, one namespace per top-level object of
the standard's own object model (the same one ``GetMachineClass()``'s
``Server_Tool_Alignment_Mover_ToolChanger_Part_RotaryTable`` naming scheme
enumerates, 6.4.1) - :attr:`IppDmeMachine.server`, ``.dme``, ``.cart_cmm``,
``.tool``, ``.found_tool``, ``.tool_changer``, ``.scanning``, ``.form_tester``, ``.mover``,
``.rotary_table``, ``.part``, ``.raw_data``, mirroring how the server side
already exposes that same object model as a set of command classes
(:mod:`pyippdme.server.classes`).

Anything the namespaces do not wrap stays reachable with
``machine.client.call(...)``, for example proprietary commands.

Every method here is a thin encode/decode layer over ``IppDmeClient.call()``;
none of it changes wire behaviour.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from pyippdme.client import IppDmeClient, LineHook, builders, features
from pyippdme.client.builders import AcquisitionPoint as AcquisitionPoint
from pyippdme.client.builders import CurvePoint as CurvePoint
from pyippdme.client.builders import PartAlignment as PartAlignment
from pyippdme.client.builders import ToolAlignment as ToolAlignment
from pyippdme.client.call import call_handle, stream_handle, unrecorded
from pyippdme.client.features import Shape as Shape
from pyippdme.client.report import Report, ReportValue, report_from_payload
from pyippdme.client.transaction import Transaction
from pyippdme.exceptions import IppDmeError
from pyippdme.protocol.ast import (
    Argument,
    BasicName,
    DataPayload,
    DataResponse,
    EventTag,
    Items,
    NamedValue,
    NameValue,
    Number,
    NumericData,
    PropertyData,
    String,
    StringValue,
    Xml,
)
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.network import TCP_NETWORK, Network
from pyippdme.protocol.parameters import ParameterName
from pyippdme.types.csy import CoordinateTransform
from pyippdme.types.rawdata import AdvDataStruct
from pyippdme.types.rawdata import from_xml as adv_data_struct_from_xml
from pyippdme.types.tool_id import ToolId
from pyippdme.types.tool_id import from_xml as tool_id_from_xml
from pyippdme.types.vec3 import Vec3


def _only(data: tuple[DataPayload, ...]) -> DataPayload:
    (item,) = data
    return item


def _items(payload: DataPayload) -> tuple[NamedValue, ...]:
    if not isinstance(payload, Items):
        raise TypeError(f"Expected an Items response, got {type(payload).__name__}")
    return payload.values


def _numeric(payload: DataPayload) -> tuple[float, ...]:
    if not isinstance(payload, NumericData):
        raise TypeError(f"Expected a NumericData response, got {type(payload).__name__}")
    return tuple(n.value for n in payload.values)


def _string_value(payload: DataPayload) -> str:
    if not isinstance(payload, StringValue):
        raise TypeError(f"Expected a StringValue response, got {type(payload).__name__}")
    return payload.value.value


def _name_value(payload: DataPayload) -> str:
    if not isinstance(payload, NameValue):
        raise TypeError(f"Expected a NameValue response, got {type(payload).__name__}")
    return payload.value


def _string_values(data: tuple[DataPayload, ...]) -> tuple[str, ...]:
    """Unwrap the "one StringValue per response line" shape (e.g. GetSupportedCommands)."""
    return tuple(_string_value(item) for item in data)


def _property_pairs(data: tuple[DataPayload, ...]) -> tuple[tuple[str, str], ...]:
    """Unwrap the "one PropertyData per response line" shape (EnumProp/EnumAllProp/...)."""
    pairs = []
    for item in data:
        if not isinstance(item, PropertyData):
            raise TypeError(f"Expected PropertyData, got {type(item).__name__}")
        pairs.append((item.first.value, item.second.value))
    return tuple(pairs)


def _num(nv: NamedValue) -> float:
    (value,) = nv.args
    if not isinstance(value, Number):
        raise TypeError(f"Expected a numeric value for {nv.name!r}, got {type(value).__name__}")
    return value.value


def _property_value(payload: DataPayload) -> float | str:
    (nv,) = _items(payload)
    (value,) = nv.args
    if isinstance(value, Number):
        return value.value
    if isinstance(value, String):
        return value.value
    raise TypeError(f"Expected a number or string property, got {type(value).__name__}")


def _named_numbers(payload: DataPayload) -> dict[str, float]:
    return {nv.name: _num(nv) for nv in _items(payload)}


def _field(fields: tuple[NamedValue, ...], name: str) -> Argument | None:
    """Return field ``name``'s single argument from an ``Items`` line's values, if present."""
    for nv in fields:
        if nv.name == name:
            (value,) = nv.args
            return value
    return None


def _named_strings(payload: DataPayload) -> dict[str, str]:
    result = {}
    for nv in _items(payload):
        (value,) = nv.args
        if not isinstance(value, String):
            raise TypeError(f"Expected a string value for {nv.name!r}, got {type(value).__name__}")
        result[nv.name] = value.value
    return result


def _transform_from_items(payload: DataPayload) -> CoordinateTransform:
    values = _named_numbers(payload)
    return CoordinateTransform(
        values["X0"], values["Y0"], values["Z0"], values["Theta"], values["Psi"], values["Phi"]
    )


class Server:
    """The ``Server`` class (6.3): session/error status.

    Beyond what :class:`~pyippdme.client.IppDmeClient` already covers directly.
    """

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    @call_handle
    async def abort_e(self) -> None:
        """Abort everything currently executing or queued (5.7)."""
        await self._client.call(CommandName.ABORT_E)

    @call_handle
    async def get_prop(self, name: str) -> ReportValue:
        """Read one property by its dotted name, e.g. ``"Part.Temperature"`` (6.3.1.1).

        The value is a number, a string, or a tuple of numbers for a property such
        as ``"Tool.Alignment"``.
        """
        report = await self._get_props(CommandName.GET_PROP, (name,))
        return report[name]

    @call_handle
    async def get_props(self, *names: str) -> Report:
        """Read several properties with one ``GetProp`` (6.3.1.1)."""
        return await self._get_props(CommandName.GET_PROP, names)

    @call_handle
    async def get_prop_e(self, name: str) -> ReportValue:
        """Like :meth:`get_prop`, but as a prioritized command that bypasses the queue (5.7)."""
        report = await self._get_props(CommandName.GET_PROP_E, (name,))
        return report[name]

    @call_handle
    async def get_props_e(self, *names: str) -> Report:
        """Like :meth:`get_props`, but as a prioritized command that bypasses the queue (5.7)."""
        return await self._get_props(CommandName.GET_PROP_E, names)

    async def _get_props(self, command: str, names: Sequence[str]) -> Report:
        payload = _only(await self._client.call(command, *builders.get_prop(*names)))
        return report_from_payload(payload)

    @call_handle
    async def set_prop(self, name: str, value: float | str | Sequence[float]) -> None:
        """Write one property by its dotted name (6.3.1.1)."""
        await self._client.call(CommandName.SET_PROP, *builders.set_prop(name, value))

    @call_handle
    async def set_props(self, values: Mapping[str, float | str | Sequence[float]]) -> None:
        """Write several properties with one ``SetProp`` (6.3.1.1)."""
        arguments = tuple(
            argument
            for name, value in values.items()
            for argument in builders.set_prop(name, value)
        )
        await self._client.call(CommandName.SET_PROP, *arguments)

    def unsolicited_events(self) -> AsyncIterator[DataResponse]:
        """Iterate the events the server sends on its own with tag ``E0000`` (5.5.1)."""
        return self._client.unsolicited_events()

    @call_handle
    async def start_session(self) -> None:
        await self._client.call(CommandName.START_SESSION)

    @call_handle
    async def end_session(self) -> None:
        await self._client.call(CommandName.END_SESSION)

    @call_handle
    async def clear_all_errors(self) -> None:
        """Let the server recover from an error (5.6)."""
        await self._client.call(CommandName.CLEAR_ALL_ERRORS)

    @call_handle
    async def get_xtd_err_status(self) -> Report:
        """Query the extended error status (6.3.1): all status lines merged by name.

        For example ``IsHomed``, plus ``ActiveError`` and ``Severity`` while an
        error is active.
        """
        status: dict[str, ReportValue] = {}
        for payload in await self._client.call(CommandName.GET_XTD_ERR_STATUS):
            status.update(report_from_payload(payload))
        return Report(status)

    @call_handle
    async def enum_name_spaces(self) -> tuple[str, ...]:
        """List the proprietary namespaces the server uses, if any (6.3.1)."""
        return _string_values(await self._client.call(CommandName.ENUM_NAME_SPACES))

    @call_handle
    async def stop_daemon(self, event_tag: EventTag) -> None:
        """Stop the daemon that was started with ``event_tag`` (5.5.2)."""
        await self._client.stop_daemon(event_tag)

    @call_handle
    async def stop_all_daemons(self) -> None:
        await self._client.call(CommandName.STOP_ALL_DAEMONS)

    @call_handle
    async def get_error_info(self, error_number: int) -> str:
        data = await self._client.call(CommandName.GET_ERROR_INFO, Number.of(error_number))
        return _string_value(_only(data))

    @call_handle
    async def get_err_status(self) -> bool:
        (value,) = _numeric(_only(await self._client.call(CommandName.GET_ERR_STATUS_E)))
        return value != 0.0

    @call_handle
    async def enum_prop(self, reference: str) -> tuple[tuple[str, str], ...]:
        """List the direct children of ``reference`` (e.g. ``"Tool.GoToPar"``, ``"Part"``).

        Returns ``((child_name, kind), ...)`` in server order (a plain
        ``dict`` would silently collapse the repeated names
        :meth:`enum_all_prop` can return at different tree positions, e.g.
        every parameter's own ``Min``); ``kind`` is one of ``"Number"``,
        ``"String"``, or ``"Property"`` (Table 20).
        """
        data = await self._client.call(CommandName.ENUM_PROP, *builders.enum_prop(reference))
        return _property_pairs(data)

    @call_handle
    async def enum_all_prop(self, reference: str) -> tuple[tuple[str, str], ...]:
        """Like :meth:`enum_prop`, but recursing into every ``"Property"``-typed child."""
        data = await self._client.call(
            CommandName.ENUM_ALL_PROP, *builders.enum_all_prop(reference)
        )
        return _property_pairs(data)


class Dme:
    """The ``DME`` class (6.4), mandatory for every conformant server."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    @call_handle
    async def get_dme_version(self) -> str:
        payload = _only(await self._client.call(CommandName.GET_DME_VERSION))
        (value,) = _items(payload)
        (version,) = value.args
        if not isinstance(version, String):
            raise TypeError(f"Expected a string DMEVersion, got {type(version).__name__}")
        return version.value

    @call_handle
    async def get_supported_commands(self) -> tuple[str, ...]:
        return _string_values(await self._client.call(CommandName.GET_SUPPORTED_COMMANDS))

    @call_handle
    async def get_supported_arguments(self, command_name: str) -> dict[str, str]:
        data = await self._client.call(CommandName.GET_SUPPORTED_ARGUMENTS, String(command_name))
        return dict(_property_pairs(data))

    @call_handle
    async def get_machine_class(self) -> str:
        """Return the machine class; use :meth:`get_machine_classes` if the server has several."""
        machine_classes = await self._machine_classes()
        if len(machine_classes) != 1:
            raise IppDmeError(
                f"The server reported {len(machine_classes)} machine classes; "
                "use get_machine_classes()"
            )
        return machine_classes[0]

    @call_handle
    async def get_machine_classes(self) -> tuple[str, ...]:
        """Return every machine class the server implements (Table 25)."""
        return await self._machine_classes()

    async def _machine_classes(self) -> tuple[str, ...]:
        return _string_values(await self._client.call(CommandName.GET_MACHINE_CLASS))

    @call_handle
    async def home(self) -> None:
        await self._client.call(CommandName.HOME)

    @call_handle
    async def is_homed(self) -> bool:
        payload = _only(await self._client.call(CommandName.IS_HOMED))
        return _named_numbers(payload)[CommandName.IS_HOMED] != 0.0


@dataclass(frozen=True, slots=True)
class TemperatureSensorInfo:
    """One ``GetTemperatureSensors()`` entry (Table 43)."""

    name: str
    #: One of ``"Mover"``, ``"Part"``, ``"CMM"``.
    kind: str
    cmm_temp_correction: bool
    #: The axis this sensor measures, for ``"Mover"``-kind sensors only.
    scale_axis: str | None


def _temperature_sensor_info(payload: DataPayload) -> TemperatureSensorInfo:
    fields = _items(payload)
    name = _field(fields, "Name")
    kind = _field(fields, "Kind")
    cmm_temp_correction = _field(fields, "CMMTempCorrection")
    scale = _field(fields, "Scale")
    if (
        not isinstance(name, String)
        or not isinstance(kind, BasicName)
        or not isinstance(cmm_temp_correction, Number)
    ):
        raise TypeError("Expected a string Name, a BasicName Kind, and a numeric CMMTempCorrection")
    if scale is not None and not isinstance(scale, String):
        raise TypeError(f"Expected a string Scale, got {type(scale).__name__}")
    return TemperatureSensorInfo(
        name=name.value,
        kind=kind.value,
        cmm_temp_correction=cmm_temp_correction.value != 0.0,
        scale_axis=scale.value if scale is not None else None,
    )


class CartCmm:
    """A subset of the ``CartCMM``/``Cartesian``/``TouchTrigger`` classes (6.5, 6.8, 6.12)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    @call_handle
    async def on_pt_meas_report(self, *names: str) -> None:
        """Choose which values ``pt_meas`` reports, e.g. ``"X", "Y", "Z", "IJK"`` (6.12.1)."""
        await self._client.call(CommandName.ON_PT_MEAS_REPORT, *builders.on_pt_meas_report(*names))

    @call_handle
    async def pt_meas_self_center(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ijk: Vec3 | None = None,
    ) -> Report:
        """Measure a point with self-centering (6.14)."""
        arguments = builders.pt_meas_self_center(x, y, z, ijk)
        payload = _only(await self._client.call(CommandName.PT_MEAS_SELF_CENTER, *arguments))
        return report_from_payload(payload)

    @call_handle
    async def pt_meas_self_center_locked(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ijk: Vec3 | None = None,
        lmn: Vec3 | None = None,
    ) -> Report:
        """Measure a point with self-centering inside the plane normal to ``lmn`` (6.14)."""
        arguments = builders.pt_meas_self_center(x, y, z, ijk, lmn)
        payload = _only(await self._client.call(CommandName.PT_MEAS_SELF_CENTER_LOCKED, *arguments))
        return report_from_payload(payload)

    @call_handle
    async def set_coord_system(self, csy: str) -> None:
        await self._client.call(CommandName.SET_COORD_SYSTEM, BasicName(csy))

    @call_handle
    async def get_coord_system(self) -> str:
        return _name_value(_only(await self._client.call(CommandName.GET_COORD_SYSTEM)))

    @call_handle
    async def get(self, *axes: str) -> Report:
        """Query one or more axes or properties, e.g. ``await cart_cmm.get("X", "IJK", "R")``."""
        return await self._get(axes)

    async def _get(self, axes: Sequence[str]) -> Report:
        payload = _only(await self._client.call(CommandName.GET, *builders.get(*axes)))
        return report_from_payload(payload)

    @call_handle
    async def get_position(self) -> Vec3:
        values = await self._get(("X", "Y", "Z"))
        return (values.number("X"), values.number("Y"), values.number("Z"))

    @call_handle
    async def go_to(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        *,
        r: float | None = None,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        alignment: ToolAlignment | None = None,
        sync: bool | None = None,
    ) -> None:
        """Move to an absolute position (6.8.1).

        ``r`` is the rotary table angle, ``a``, ``b`` and ``c`` are tool angles and
        ``alignment`` the tool orientation. With ``sync=True`` all axes start and
        end together.
        """
        await self._client.call(
            CommandName.GO_TO,
            *builders.go_to(x, y, z, r=r, a=a, b=b, c=c, alignment=alignment, sync=sync),
        )

    @call_handle
    async def step(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        *,
        r: float | None = None,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        sync: bool | None = None,
    ) -> None:
        """Perform a relative move (6.8.1): add to the current position, not replace it."""
        await self._client.call(
            CommandName.STEP, *builders.step(x, y, z, r=r, a=a, b=b, c=c, sync=sync)
        )

    @call_handle
    async def go_to_on_circle(self, center: Vec3, ijk: Vec3, target: Vec3) -> None:
        await self._client.call(
            CommandName.GO_TO_ON_CIRCLE, *builders.go_to_on_circle(center, ijk, target)
        )

    @call_handle
    async def go_to_on_spiral(self, center: Vec3, ijk: Vec3, target: Vec3) -> None:
        await self._client.call(
            CommandName.GO_TO_ON_SPIRAL, *builders.go_to_on_spiral(center, ijk, target)
        )

    @call_handle
    async def pt_meas(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        *,
        ijk: Vec3 | None = None,
        r: float | None = None,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        alignment: ToolAlignment | None = None,
        align_part: PartAlignment | None = None,
    ) -> Report:
        """Measure a point (6.12.1); ``ijk`` is the probing direction.

        The result holds what :meth:`on_pt_meas_report` asked for.
        """
        arguments = builders.pt_meas(
            x, y, z, ijk=ijk, r=r, a=a, b=b, c=c, alignment=alignment, align_part=align_part
        )
        payload = _only(await self._client.call(CommandName.PT_MEAS, *arguments))
        return report_from_payload(payload)

    @call_handle
    async def set_csy_transformation(self, csy: str, transform: CoordinateTransform) -> None:
        await self._client.call(
            CommandName.SET_CSY_TRANSFORMATION,
            BasicName(csy),
            *builders.set_csy_transformation(transform),
        )

    @call_handle
    async def get_csy_transformation(self, csy: str) -> CoordinateTransform:
        payload = _only(await self._client.call(CommandName.GET_CSY_TRANSFORMATION, BasicName(csy)))
        return _transform_from_items(payload)

    @call_handle
    async def save_named_csy_transformation(
        self, name: str, transform: CoordinateTransform
    ) -> None:
        await self._client.call(
            CommandName.SAVE_NAMED_CSY_TRANSFORMATION,
            String(name),
            *builders.set_csy_transformation(transform),
        )

    @call_handle
    async def get_named_csy_transformation(self, name: str) -> CoordinateTransform:
        payload = _only(
            await self._client.call(CommandName.GET_NAMED_CSY_TRANSFORMATION, String(name))
        )
        return _transform_from_items(payload)

    @call_handle
    async def save_active_coord_system(self, name: str) -> None:
        await self._client.call(CommandName.SAVE_ACTIVE_COORD_SYSTEM, String(name))

    @call_handle
    async def load_coord_system(self, name: str) -> None:
        await self._client.call(CommandName.LOAD_COORD_SYSTEM, String(name))

    @call_handle
    async def delete_coord_system(self, name: str) -> None:
        await self._client.call(CommandName.DELETE_COORD_SYSTEM, String(name))

    @call_handle
    async def enum_coord_systems(self) -> tuple[str, ...]:
        return _string_values(await self._client.call(CommandName.ENUM_COORD_SYSTEMS))

    @call_handle
    async def get_temperature_sensors(self) -> tuple[TemperatureSensorInfo, ...]:
        data = await self._client.call(CommandName.GET_TEMPERATURE_SENSORS)
        return tuple(_temperature_sensor_info(payload) for payload in data)

    @call_handle
    async def read_temperature_sensor(self, name: str) -> float:
        payload = _only(await self._client.call(CommandName.READ_TEMPERATURE_SENSOR, String(name)))
        (value,) = _numeric(payload)
        return value

    @call_handle
    async def read_all_temperatures(self) -> dict[str, float]:
        """Every currently-connected sensor's reading; a disconnected sensor is omitted."""
        data = await self._client.call(CommandName.READ_ALL_TEMPERATURES)
        result = {}
        for payload in data:
            fields = _items(payload)
            name = _field(fields, "Name")
            temperature = _field(fields, "Temperature")
            if not isinstance(name, String) or not isinstance(temperature, Number):
                raise TypeError("Expected a string Name and a numeric Temperature")
            result[name.value] = temperature.value
        return result


@dataclass(frozen=True, slots=True)
class ToolParameterValue:
    """One parameter's Min/Max/Def/Act quadruplet (6.10.4), as read over the wire."""

    minimum: float
    maximum: float
    default: float
    act: float


class _ToolProperties:
    """The read-only properties ``Tool`` and ``FoundTool`` have in common (6.10.3)."""

    def __init__(self, client: IppDmeClient, namespace: str) -> None:
        self._client = client
        self._namespace = namespace

    async def _property(self, leaf: str) -> NamedValue:
        name = f"{self._namespace}.{leaf}"
        payload = _only(await self._client.call(CommandName.GET_PROP, *builders.get_prop(name)))
        (field,) = _items(payload)
        return field

    async def _string_property(self, leaf: str) -> str:
        (value,) = (await self._property(leaf)).args
        if not isinstance(value, String):
            raise TypeError(f"Expected a string {leaf}, got {type(value).__name__}")
        return value.value

    @call_handle
    async def get_name(self) -> str:
        """Return the tool's name, or ``"NoTool"`` if there is none."""
        return await self._string_property("Name")

    @call_handle
    async def get_id(self) -> ToolId:
        """Return the tool description (Annex G)."""
        field = await self._property("Id")
        if field.xml is None:
            raise TypeError(f"{self._namespace}.Id() did not return an XML payload")
        return tool_id_from_xml(field.xml.raw)

    @call_handle
    async def get_collection(self) -> str:
        """Return the tool collection the tool belongs to."""
        return await self._string_property("Collection")

    @call_handle
    async def get_last_qualified(self) -> str:
        """Return when the tool was last qualified, as the server's ISO 8601 UTC string."""
        return await self._string_property("LastQualified")

    @call_handle
    async def get_alignment(self) -> tuple[float, ...]:
        """Return the alignment vector(s) of an alignable tool (6.20)."""
        numbers = []
        for value in (await self._property("Alignment")).args:
            if not isinstance(value, Number):
                raise TypeError(f"Expected numbers, got {type(value).__name__}")
            numbers.append(value.value)
        return tuple(numbers)


class FoundTool(_ToolProperties):
    """The ``FoundTool`` object (6.22): the tool the last ``find_tool`` located."""

    def __init__(self, client: IppDmeClient) -> None:
        super().__init__(client, "FoundTool")


class Tool(_ToolProperties):
    """A subset of the ``Tool`` class (6.10): its properties, parameter blocks and alignment."""

    def __init__(self, client: IppDmeClient) -> None:
        super().__init__(client, "Tool")

    @call_handle
    async def re_qualify(self) -> None:
        """Requalify the active tool (6.10.1)."""
        await self._client.call(CommandName.RE_QUALIFY)

    @call_handle
    async def align_tool(
        self,
        primary: Vec3,
        alpha: float,
        secondary: Vec3 | None = None,
        beta: float | None = None,
    ) -> None:
        """Align the tool (6.20); ``secondary`` and ``beta`` must be given together."""
        await self._client.call(
            CommandName.ALIGN_TOOL, *builders.align_tool(primary, alpha, secondary, beta)
        )

    @call_handle
    async def use_smallest_angle_to_align_tool(self, flag: bool) -> None:
        await self._client.call(
            CommandName.USE_SMALLEST_ANGLE_TO_ALIGN_TOOL, Number.of(1 if flag else 0)
        )

    @call_handle
    async def enable_optimize(self) -> None:
        await self._client.call(CommandName.ENABLE_OPTIMIZE)

    @call_handle
    async def disable_optimize(self) -> None:
        await self._client.call(CommandName.DISABLE_OPTIMIZE)

    @call_handle
    async def is_optimize_enabled(self) -> bool:
        (value,) = _numeric(_only(await self._client.call(CommandName.IS_OPTIMIZE_ENABLED)))
        return value != 0.0

    @call_handle
    async def calc_tool_alignment(
        self, axis: str = "A", *, namespace: str = "Tool"
    ) -> tuple[float, ...]:
        """Calculate the alignment vector(s) for the current tool angles (6.20)."""
        payload = _only(
            await self._client.call(
                CommandName.CALC_TOOL_ALIGNMENT, *builders.alignment_axis(namespace, axis)
            )
        )
        (field,) = _items(payload)
        numbers = []
        for value in field.args:
            if not isinstance(value, Number):
                raise TypeError(f"Expected numbers, got {type(value).__name__}")
            numbers.append(value.value)
        return tuple(numbers)

    @call_handle
    async def calc_tool_angles(
        self, primary: Vec3, secondary: Vec3 | None = None, *, namespace: str = "Tool"
    ) -> dict[str, float]:
        """Calculate the tool angles (``A``, ``B``, maybe ``C``) for alignment vector(s) (6.20)."""
        payload = _only(
            await self._client.call(
                CommandName.CALC_TOOL_ANGLES,
                *builders.alignment_vectors(namespace, primary, secondary),
            )
        )
        prefix = f"{namespace}."
        return {name.removeprefix(prefix): value for name, value in _named_numbers(payload).items()}

    @call_handle
    async def avr_radius(self) -> float:
        """Average effective radius of the tool tip (6.20)."""
        (value,) = _numeric(_only(await self._client.call(CommandName.AVR_RADIUS)))
        return value

    @call_handle
    async def avr_offsets(self) -> Vec3:
        """Average offsets of the tool tip (6.20)."""
        values = {
            name.upper(): value
            for name, value in _named_numbers(
                _only(await self._client.call(CommandName.AVR_OFFSETS))
            ).items()
        }
        return (values["X"], values["Y"], values["Z"])

    @call_handle
    async def collision_volume(self) -> tuple[DataPayload, ...]:
        """Return the tool's collision volume as raw response lines (6.20)."""
        return await self._client.call(CommandName.COLLISION_VOLUME)

    @call_handle
    async def alignment_volume(self) -> tuple[DataPayload, ...]:
        """Return the tool's alignment volume as raw response lines (6.20)."""
        return await self._client.call(CommandName.ALIGNMENT_VOLUME)

    @call_handle
    async def is_alignable(self) -> bool:
        (value,) = _numeric(_only(await self._client.call(CommandName.IS_ALIGNABLE)))
        return value != 0.0

    @call_handle
    async def get_parameter(self, block: str, name: str) -> ToolParameterValue:
        """E.g. ``await tool.get_parameter("PtMeasPar", "Retract")``."""
        path = f"Tool.{block}.{name}"
        payload = _only(
            await self._client.call(
                CommandName.GET_PROP,
                *builders.get_prop(f"{path}.Min", f"{path}.Max", f"{path}.Def", f"{path}.Act"),
            )
        )
        values = _named_numbers(payload)
        return ToolParameterValue(
            values[f"{path}.Min"],
            values[f"{path}.Max"],
            values[f"{path}.Def"],
            values[f"{path}.Act"],
        )

    @call_handle
    async def set_parameter(self, block: str, name: str, value: float) -> None:
        await self._client.call(
            CommandName.SET_PROP, *builders.set_prop(f"Tool.{block}.{name}", value)
        )


PointT = TypeVar("PointT")

#: How many numbers each name of an ``OnScanReport`` contributes to one point (6.13.2).
_REPORT_WIDTHS: Mapping[str, int] = {
    "X": 1,
    "Y": 1,
    "Z": 1,
    "R": 1,
    "Q": 1,
    "ER": 1,
    "IJK": 3,
    "IJKAct": 1,
    "Tool.A": 1,
    "Tool.B": 1,
    "Tool.C": 1,
}


@dataclass(frozen=True, slots=True)
class _PointFormat(Generic[PointT]):
    """The names reported per scanned point and how to turn one point's numbers into a value."""

    names: tuple[str, ...]
    width: int
    decode: Callable[[Sequence[float]], PointT]


def _xyz(values: Sequence[float]) -> Vec3:
    return (values[0], values[1], values[2])


_XYZ_FORMAT = _PointFormat(("X", "Y", "Z"), 3, _xyz)


def _report_format(names: Sequence[str], widths: Mapping[str, int]) -> _PointFormat[Report]:
    sizes = []
    for name in names:
        if name not in widths:
            raise ValueError(
                f"The number of values {name!r} reports is not known; give it with widths={{...}}"
            )
        sizes.append(widths[name])

    def decode(values: Sequence[float]) -> Report:
        report: dict[str, ReportValue] = {}
        offset = 0
        for name, size in zip(names, sizes, strict=True):
            part = tuple(values[offset : offset + size])
            report[name] = part[0] if size == 1 else part
            offset += size
        return Report(report)

    return _PointFormat(tuple(names), sum(sizes), decode)


class Scanning(Generic[PointT]):
    """The ``Scanning`` class (6.13).

    ``machine.scanning`` yields each scanned point as ``(x, y, z)``.
    :meth:`reporting` gives the same commands with other reported values per point.
    """

    def __init__(self, client: IppDmeClient, point_format: _PointFormat[PointT]) -> None:
        self._client = client
        self._format = point_format

    def reporting(self, *names: str, widths: Mapping[str, int] | None = None) -> Scanning[Report]:
        """Return the scan commands with each point as a :class:`Report` of ``names``.

        ``names`` are what ``OnScanReport`` asks for, for example ``"X", "Y", "Z", "Q"``
        or ``"IJK"``. ``widths`` gives how many numbers a name contributes to a point
        when the standard does not fix it, for example ``{"Tool.Alignment": 6}``
        or a proprietary value.
        """
        return Scanning(self._client, _report_format(names, {**_REPORT_WIDTHS, **(widths or {})}))

    @call_handle
    async def on_scan_report(self, *names: str) -> None:
        """Choose which values a scan reports (6.13.2).

        The scan methods set their own format before each scan; this is for
        scans started some other way.
        """
        await self._client.call(CommandName.ON_SCAN_REPORT, *builders.on_scan_report(*names))

    async def _scan(self, command: str, arguments: tuple[Argument, ...]) -> AsyncIterator[PointT]:
        """Stream a scan's points in the format of this object.

        Each streamed response is a *bare* ``NumericData``: positional values
        in the order ``OnScanReport(...)`` established (6.13.2), possibly with
        several points blocked into one response. This sets that order
        itself instead of trusting a previously configured one.
        """
        with unrecorded():
            await self._client.call(
                CommandName.ON_SCAN_REPORT, *builders.on_scan_report(*self._format.names)
            )
        width = self._format.width
        async for payload in self._client.call_streaming(command, *arguments):
            values = _numeric(payload)
            if len(values) % width:
                raise TypeError(f"Expected a multiple of {width} values per response")
            for start in range(0, len(values), width):
                yield self._format.decode(values[start : start + width])

    @stream_handle
    async def scan_on_line(
        self,
        start: Vec3,
        end: Vec3,
        direction: Vec3,
        step_width: float,
        rotary_table: bool | None = None,
    ) -> AsyncIterator[PointT]:
        """Scan a straight line, yielding each measured point as it arrives (6.13.2.1)."""
        arguments = builders.positional(start, end, direction, step_width, rotary_table)
        async for point in self._scan(CommandName.SCAN_ON_LINE, arguments):
            yield point

    @stream_handle
    async def scan_on_circle(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta: float,
        surface_angle: float,
        step_width: float,
        rotary_table: bool | None = None,
    ) -> AsyncIterator[PointT]:
        """Scan a circular arc of ``delta`` degrees around ``center`` (6.13.2.1)."""
        arguments = builders.scan_on_circle(
            center, start, normal, delta, surface_angle, step_width, rotary_table
        )
        async for point in self._scan(CommandName.SCAN_ON_CIRCLE, arguments):
            yield point

    @stream_handle
    async def scan_on_helix(
        self,
        center: Vec3,
        start: Vec3,
        normal: Vec3,
        delta: float,
        surface_angle: float,
        step_width: float,
        pitch: float,
        rotary_table: bool | None = None,
    ) -> AsyncIterator[PointT]:
        """Scan a helix: a circular arc that also rises by ``pitch`` per turn (6.13.2.1)."""
        arguments = builders.scan_on_circle(
            center, start, normal, delta, surface_angle, step_width, rotary_table, pitch=pitch
        )
        async for point in self._scan(CommandName.SCAN_ON_HELIX, arguments):
            yield point

    @stream_handle
    async def scan_on_curve(
        self,
        points: Sequence[CurvePoint],
        *,
        closed: bool = False,
        rotary_table: bool | None = None,
    ) -> AsyncIterator[PointT]:
        """Scan a client-supplied nominal curve, yielding each measured point (6.13.2.1).

        The optional tool directions and rotary-table angle of
        :class:`~pyippdme.client.builders.CurvePoint` are sent as extra columns
        when the points have them.
        """
        arguments = builders.scan_on_curve(points, closed=closed, rotary_table=rotary_table)
        async for point in self._scan(CommandName.SCAN_ON_CURVE, arguments):
            yield point

    @stream_handle
    async def scan_in_plane_end_is_sphere(
        self,
        start: Vec3,
        start_ijk: Vec3,
        plane_normal: Vec3,
        direction_point: Vec3,
        step_width: float,
        end_center: Vec3,
        diameter: float,
        n: int,
        end_ijk: Vec3,
    ) -> AsyncIterator[PointT]:
        """Scan an unknown contour in a plane until ``n`` hits of the end sphere (6.13.2.2)."""
        arguments = builders.positional(
            start,
            start_ijk,
            plane_normal,
            direction_point,
            step_width,
            end_center,
            diameter,
            n,
            end_ijk,
        )
        async for point in self._scan(CommandName.SCAN_IN_PLANE_END_IS_SPHERE, arguments):
            yield point

    @stream_handle
    async def scan_in_plane_end_is_plane(
        self,
        start: Vec3,
        start_ijk: Vec3,
        plane_normal: Vec3,
        direction_point: Vec3,
        step_width: float,
        end_plane_point: Vec3,
        end_plane_normal: Vec3,
        n: int,
        end_ijk: Vec3,
    ) -> AsyncIterator[PointT]:
        """Scan an unknown contour in a plane until ``n`` crossings of the end plane (6.13.2.2)."""
        arguments = builders.positional(
            start,
            start_ijk,
            plane_normal,
            direction_point,
            step_width,
            end_plane_point,
            end_plane_normal,
            n,
            end_ijk,
        )
        async for point in self._scan(CommandName.SCAN_IN_PLANE_END_IS_PLANE, arguments):
            yield point

    @stream_handle
    async def scan_in_plane_end_is_cyl(
        self,
        start: Vec3,
        start_ijk: Vec3,
        plane_normal: Vec3,
        direction_point: Vec3,
        step_width: float,
        axis_point: Vec3,
        axis_direction: Vec3,
        diameter: float,
        n: int,
        end_ijk: Vec3,
    ) -> AsyncIterator[PointT]:
        """Scan an unknown contour in a plane until ``n`` hits of the end cylinder (6.13.2.2)."""
        arguments = builders.positional(
            start,
            start_ijk,
            plane_normal,
            direction_point,
            step_width,
            axis_point,
            axis_direction,
            diameter,
            n,
            end_ijk,
        )
        async for point in self._scan(CommandName.SCAN_IN_PLANE_END_IS_CYL, arguments):
            yield point

    @stream_handle
    async def scan_in_cyl_end_is_sphere(
        self,
        axis_point: Vec3,
        axis_direction: Vec3,
        start: Vec3,
        start_ijk: Vec3,
        direction_point: Vec3,
        step_width: float,
        end_center: Vec3,
        diameter: float,
        n: int,
        end_ijk: Vec3,
    ) -> AsyncIterator[PointT]:
        """Scan an unknown contour on a cylinder until ``n`` hits of the end sphere (6.13.2.2)."""
        arguments = builders.positional(
            axis_point,
            axis_direction,
            start,
            start_ijk,
            direction_point,
            step_width,
            end_center,
            diameter,
            n,
            end_ijk,
        )
        async for point in self._scan(CommandName.SCAN_IN_CYL_END_IS_SPHERE, arguments):
            yield point

    @stream_handle
    async def scan_in_cyl_end_is_plane(
        self,
        axis_point: Vec3,
        axis_direction: Vec3,
        start: Vec3,
        start_ijk: Vec3,
        direction_point: Vec3,
        step_width: float,
        end_plane_point: Vec3,
        end_plane_normal: Vec3,
        n: int,
        end_ijk: Vec3,
    ) -> AsyncIterator[PointT]:
        """Scan an unknown contour on a cylinder until ``n`` end-plane crossings (6.13.2.2)."""
        arguments = builders.positional(
            axis_point,
            axis_direction,
            start,
            start_ijk,
            direction_point,
            step_width,
            end_plane_point,
            end_plane_normal,
            n,
            end_ijk,
        )
        async for point in self._scan(CommandName.SCAN_IN_CYL_END_IS_PLANE, arguments):
            yield point

    @call_handle
    async def scan_on_line_hint(self, angle: float, form: float) -> None:
        await self._client.call(CommandName.SCAN_ON_LINE_HINT, *builders.positional(angle, form))

    @call_handle
    async def scan_on_circle_hint(self, displacement: float, form: float) -> None:
        await self._client.call(
            CommandName.SCAN_ON_CIRCLE_HINT, *builders.positional(displacement, form)
        )

    @call_handle
    async def scan_on_curve_hint(self, deviation: float, min_radius_of_curvature: float) -> None:
        await self._client.call(
            CommandName.SCAN_ON_CURVE_HINT,
            *builders.positional(deviation, min_radius_of_curvature),
        )

    @call_handle
    async def scan_on_curve_density(
        self,
        dis: float | None = None,
        angle: float | None = None,
        angle_base_length: float | None = None,
        at_nominals: bool | None = None,
    ) -> None:
        await self._client.call(
            CommandName.SCAN_ON_CURVE_DENSITY,
            *builders.density(dis, angle, angle_base_length, at_nominals),
        )

    @call_handle
    async def scan_unknown_hint(self, min_radius_of_curvature: float) -> None:
        await self._client.call(
            CommandName.SCAN_UNKNOWN_HINT, *builders.positional(min_radius_of_curvature)
        )

    @call_handle
    async def scan_unknown_density(
        self,
        dis: float | None = None,
        angle: float | None = None,
        angle_base_length: float | None = None,
    ) -> None:
        await self._client.call(
            CommandName.SCAN_UNKNOWN_DENSITY, *builders.density(dis, angle, angle_base_length)
        )


class FormTester:
    """A subset of the ``FormTester`` class (6.6.1)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    async def _status(self, command: str, *values: float) -> bool:
        data = await self._client.call(command, *(Number.of(v) for v in values))
        (achieved,) = _numeric(_only(data))
        return achieved != 0.0

    @call_handle
    async def center_part(self, x: float, y: float, z: float, limit: float) -> bool:
        return await self._status(CommandName.CENTER_PART, x, y, z, limit)

    @call_handle
    async def tilt_part(self, i: float, j: float, k: float, limit: float) -> bool:
        return await self._status(CommandName.TILT_PART, i, j, k, limit)

    @call_handle
    async def tilt_center_part(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        limit: float,
    ) -> bool:
        return await self._status(CommandName.TILT_CENTER_PART, x1, y1, z1, x2, y2, z2, limit)

    @call_handle
    async def lock_axis(self, *axes: str) -> None:
        await self._client.call(CommandName.LOCK_AXIS, *builders.lock_axis(*axes))

    @call_handle
    async def lock_position(self, *positions: str) -> None:
        await self._client.call(CommandName.LOCK_POSITION, *builders.lock_position(*positions))


class Mover:
    """The ``Mover`` class (6.7.1)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    @call_handle
    async def update_scale_temperatures(self) -> None:
        await self._client.call(CommandName.UPDATE_SCALE_TEMPERATURES)

    @call_handle
    async def set_temperature_compensation_origin(
        self, x: float | None = None, y: float | None = None, z: float | None = None
    ) -> None:
        await self._client.call(
            CommandName.SET_TEMPERATURE_COMPENSATION_ORIGIN,
            *builders.named_numbers(
                (ParameterName.X, x), (ParameterName.Y, y), (ParameterName.Z, z)
            ),
        )

    @call_handle
    async def on_move_report(
        self, time: float, dis: float, *axes: str, prioritized: bool = False
    ) -> Transaction:
        """Start the move-report daemon (6.7.1); reports arrive via ``Transaction.events()``.

        Stop it with ``machine.server.stop_daemon(transaction.tag)``.
        """
        name = CommandName.ON_MOVE_REPORT_E if prioritized else CommandName.ON_MOVE_REPORT
        return await self._client.start_daemon(name, *builders.on_move_report(time, dis, axes))

    @call_handle
    async def enable_user(self) -> None:
        await self._client.call(CommandName.ENABLE_USER)

    @call_handle
    async def disable_user(self) -> None:
        await self._client.call(CommandName.DISABLE_USER)

    @call_handle
    async def is_user_enabled(self) -> bool:
        payload = _only(await self._client.call(CommandName.IS_USER_ENABLED))
        return _named_numbers(payload)[CommandName.IS_USER_ENABLED] != 0.0

    @call_handle
    async def enumerate_mover_axes(self) -> tuple[str, ...]:
        payload = _only(await self._client.call(CommandName.ENUMERATE_MOVER_AXES))
        return tuple(nv.name for nv in _items(payload))

    @call_handle
    async def set_scale_temperatures(self, **temperatures: float) -> None:
        await self._client.call(
            CommandName.SET_SCALE_TEMPERATURES, *builders.set_scale_temperatures(**temperatures)
        )

    @call_handle
    async def get_scale_temperatures(self, *axes: str) -> dict[str, float]:
        payload = _only(
            await self._client.call(
                CommandName.GET_SCALE_TEMPERATURES, *builders.get_scale_temperatures(*axes)
            )
        )
        return _named_numbers(payload)


class ToolChanger:
    """The ``ToolChanger`` class (6.22), over a small fixed tool catalog.

    See :data:`pyippdme.simulation.classes.tool_class.TOOL_CATALOG` for which tool
    names the built-in/``VirtualCMM`` server accepts.
    """

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    @call_handle
    async def enum_tool_collection(self, node_name: str) -> tuple[tuple[str, str], ...]:
        """List the direct children of a tool collection as ``(name, kind)`` pairs (6.22)."""
        data = await self._client.call(CommandName.ENUM_TOOL_COLLECTION, String(node_name))
        return _property_pairs(data)

    @call_handle
    async def enum_all_tool_collections(self, node_name: str) -> tuple[tuple[str, str], ...]:
        """Like :meth:`enum_tool_collection`, but recursing into sub-collections (6.22)."""
        data = await self._client.call(CommandName.ENUM_ALL_TOOL_COLLECTIONS, String(node_name))
        return _property_pairs(data)

    @call_handle
    async def open_tool_collection(self, node_name: str) -> None:
        await self._client.call(CommandName.OPEN_TOOL_COLLECTION, String(node_name))

    @call_handle
    async def enum_tools(self) -> tuple[str, ...]:
        data = await self._client.call(CommandName.ENUM_TOOLS)
        return tuple(_name_value(item) for item in data)

    @call_handle
    async def change_tool(self, tool_name: str) -> None:
        await self._client.call(CommandName.CHANGE_TOOL, String(tool_name))

    @call_handle
    async def find_tool(self, tool_name: str) -> None:
        await self._client.call(CommandName.FIND_TOOL, String(tool_name))

    @call_handle
    async def found_tool(self) -> str:
        return _name_value(_only(await self._client.call(CommandName.FOUND_TOOL)))

    @call_handle
    async def set_tool(self, tool_name: str) -> None:
        await self._client.call(CommandName.SET_TOOL, String(tool_name))

    @call_handle
    async def get_change_tool_action(self, tool_name: str) -> tuple[str, Vec3]:
        """Return ``(action, (dx, dy, dz))``.

        ``action`` is one of "Switch"/"Rotate"/"MoveAuto"/"MoveMan"
        (6.22.1's ``GetChangeToolAction`` table).
        """
        payload = _only(
            await self._client.call(CommandName.GET_CHANGE_TOOL_ACTION, String(tool_name))
        )
        action: str | None = None
        numbers: dict[str, float] = {}
        for nv in _items(payload):
            if nv.name == "Argument":
                (value,) = nv.args
                if not isinstance(value, BasicName):
                    raise TypeError(f"Expected a BasicName Argument, got {type(value).__name__}")
                action = value.value
            else:
                numbers[nv.name] = _num(nv)
        if action is None:
            raise TypeError("GetChangeToolAction response is missing Argument")
        return action, (numbers["X"], numbers["Y"], numbers["Z"])


class Part:
    """The ``Part`` class (6.24.2): temperature/approach/search/retract properties."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    async def _get_float(self, name: str) -> float:
        payload = _only(await self._client.call(CommandName.GET_PROP, *builders.get_prop(name)))
        return _named_numbers(payload)[name]

    async def _set_float(self, name: str, value: float) -> None:
        await self._client.call(CommandName.SET_PROP, *builders.set_prop(name, value))

    @call_handle
    async def get_temperature(self) -> float:
        return await self._get_float("Part.Temperature")

    @call_handle
    async def set_temperature(self, value: float) -> None:
        await self._set_float("Part.Temperature", value)

    @call_handle
    async def get_xpan_coefficient(self) -> float:
        return await self._get_float("Part.XpanCoefficient")

    @call_handle
    async def set_xpan_coefficient(self, value: float) -> None:
        await self._set_float("Part.XpanCoefficient", value)

    @call_handle
    async def get_approach(self) -> float:
        return await self._get_float("Part.Approach")

    @call_handle
    async def set_approach(self, value: float) -> None:
        await self._set_float("Part.Approach", value)

    @call_handle
    async def get_search(self) -> float:
        return await self._get_float("Part.Search")

    @call_handle
    async def set_search(self, value: float) -> None:
        await self._set_float("Part.Search", value)

    @call_handle
    async def get_retract(self) -> float:
        return await self._get_float("Part.Retract")

    @call_handle
    async def set_retract(self, value: float) -> None:
        await self._set_float("Part.Retract", value)


class RotaryTable:
    """A subset of the ``RotaryTable`` class (6.23)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    @call_handle
    async def enable_rotary_table_var_csy(self, enabled: bool) -> None:
        await self._client.call(
            CommandName.ENABLE_ROTARY_TABLE_VAR_CSY, Number.of(1 if enabled else 0)
        )

    @call_handle
    async def align_part(
        self,
        part_vector: Vec3,
        machine_vector: Vec3,
        alpha: float,
        *,
        second_part_vector: Vec3 | None = None,
        second_machine_vector: Vec3 | None = None,
        beta: float | None = None,
    ) -> tuple[Vec3, ...]:
        """Orient the part (6.23.1); the second vectors and ``beta`` are for a second table.

        Returns the vectors that were reached: two per rotary table.
        """
        alignment = PartAlignment(
            part_vector,
            machine_vector,
            alpha,
            second_part_vector,
            second_machine_vector,
            beta,
        )
        numbers = _numeric(
            _only(await self._client.call(CommandName.ALIGN_PART, *builders.align_part(alignment)))
        )
        return tuple(
            (numbers[i], numbers[i + 1], numbers[i + 2]) for i in range(0, len(numbers), 3)
        )


class RawDataHandling:
    """A subset of the ``Optical``/``RawDataHandling`` classes (6.15, 6.17)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    @call_handle
    async def raw_data_bin_setup(
        self, data_format: str, port: int, live_mode: bool = False
    ) -> None:
        """Choose the sample format (``"float"``/``"double"``) and port to send to (6.17.2.1)."""
        await self._client.call(
            CommandName.RAW_DATA_BIN_SETUP,
            *builders.raw_data_bin_setup(data_format, port, live_mode),
        )

    @call_handle
    async def get_raw_data_bin(self, acq_name: str) -> None:
        """Send an acquisition to the port chosen with :meth:`raw_data_bin_setup` (6.17.2.1).

        The samples do not come back through this connection: read them from the
        port, for example with :func:`pyippdme.rawdata.transfer.read_samples`.
        """
        await self._client.call(CommandName.GET_RAW_DATA_BIN, String(acq_name))

    @call_handle
    async def get_adv_data_struct(self) -> AdvDataStruct:
        payload = _only(await self._client.call(CommandName.ADV_DATA_STRUCT))
        if not isinstance(payload, Xml):
            raise TypeError(f"Expected an Xml response, got {type(payload).__name__}")
        return adv_data_struct_from_xml(payload.raw)

    @call_handle
    async def data_acquire(
        self,
        acq_name: str,
        acquisition_type: str,
        settings_name: str,
        points: Sequence[AcquisitionPoint] = (),
        *,
        step_width: float | None = None,
        rotary_table: bool | None = None,
    ) -> None:
        """Trigger an acquisition (6.15.1).

        Without ``points`` this is a single shot at the current position. Otherwise
        ``points`` are the positions and tool orientations the sensor moves along;
        ``step_width`` is the maximum step of a multi-shot or sweep, and
        ``rotary_table`` lets the server move the rotary table (it needs ``step_width``).
        """
        await self._client.call(
            CommandName.DATA_ACQUIRE,
            *builders.data_acquire(
                acq_name,
                acquisition_type,
                settings_name,
                points,
                step_width=step_width,
                rotary_table=rotary_table,
            ),
        )

    @call_handle
    async def get_raw_data_file(self, acq_name: str) -> str:
        """Return the ``file://`` URL the acquisition's data was written to."""
        payload = _only(await self._client.call(CommandName.GET_RAW_DATA_FILE, String(acq_name)))
        return _named_strings(payload)["FileURL"]

    @call_handle
    async def del_raw_data_file(self, acq_name: str) -> None:
        await self._client.call(CommandName.DEL_RAW_DATA_FILE, String(acq_name))

    @call_handle
    async def delete_acquisition(self, acq_name: str) -> None:
        await self._client.call(CommandName.DELETE_ACQUISITION, String(acq_name))

    @call_handle
    async def delete_all_acquisitions(self) -> None:
        await self._client.call(CommandName.DELETE_ALL_ACQUISITIONS)

    @call_handle
    async def get_raw_data_sha_mem(self, acq_name: str) -> tuple[str, int, int]:
        """Return ``(shared_memory_name, offset, size_bytes)``."""
        payload = _only(await self._client.call(CommandName.GET_RAW_DATA_SHA_MEM, String(acq_name)))
        fields = {nv.name: nv.args[0] for nv in _items(payload)}
        name, offset, size = fields["ShaMemName"], fields["DataSegment"], fields["Size"]
        if not isinstance(name, String):
            raise TypeError(f"Expected a string ShaMemName, got {type(name).__name__}")
        if not isinstance(offset, Number) or not isinstance(size, Number):
            raise TypeError("Expected numeric DataSegment/Size")
        return name.value, int(offset.value), int(size.value)

    @call_handle
    async def release_sha_mem(self, acq_name: str) -> None:
        await self._client.call(CommandName.RELEASE_SHA_MEM, String(acq_name))


class FeatureExtraction:
    """The deprecated ``FeatureExtraction`` class (Annex J.2)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    @call_handle
    async def roi(self, name: str, shape: Shape, *, include: bool = True) -> None:
        """Define a region of interest that ``feature_extract`` can refer to by ``name``."""
        await self._client.call(CommandName.ROI, *features.roi(name, shape, include))

    @call_handle
    async def feature_extract(
        self,
        acquisitions: Sequence[str],
        feature: Shape,
        rois: Sequence[str],
        settings_name: str,
    ) -> tuple[Report, ...]:
        """Extract the geometric element ``feature`` from the acquisitions."""
        arguments = features.feature_extract(acquisitions, feature, rois, settings_name, None)
        return tuple(
            report_from_payload(payload)
            for payload in await self._client.call(CommandName.FEATURE_EXTRACT, *arguments)
        )

    @stream_handle
    async def feature_extract_points(
        self,
        acquisitions: Sequence[str],
        feature: Shape,
        rois: Sequence[str],
        settings_name: str,
        step_width: float,
    ) -> AsyncIterator[tuple[float, ...]]:
        """Extract the qualified edge points at ``step_width``, in the ``OnScanReport`` order."""
        arguments = features.feature_extract(acquisitions, feature, rois, settings_name, step_width)
        async for payload in self._client.call_streaming(CommandName.FEATURE_EXTRACT, *arguments):
            yield _numeric(payload)


class IppDmeMachine:
    """A typed facade over one :class:`~pyippdme.client.IppDmeClient` connection.

    Groups commands the same way the standard's own object model does (one
    namespace per top-level object), so callers work with typed methods and
    values instead of building/decoding raw :mod:`pyippdme.protocol.ast`
    nodes by hand::

        machine = await IppDmeMachine.connect(host, port)
        await machine.start_session()
        await machine.cart_cmm.go_to(x=10, y=20)
        position = await machine.cart_cmm.get_position()

    Anything not wrapped by a namespace here remains reachable via
    :attr:`client` directly (``machine.client.call(...)``).
    """

    def __init__(self, client: IppDmeClient) -> None:
        self.client = client
        self.server = Server(client)
        self.dme = Dme(client)
        self.cart_cmm = CartCmm(client)
        self.tool = Tool(client)
        self.found_tool = FoundTool(client)
        self.tool_changer = ToolChanger(client)
        self.scanning: Scanning[Vec3] = Scanning(client, _XYZ_FORMAT)
        self.form_tester = FormTester(client)
        self.mover = Mover(client)
        self.rotary_table = RotaryTable(client)
        self.part = Part(client)
        self.raw_data = RawDataHandling(client)
        self.feature_extraction = FeatureExtraction(client)

    @classmethod
    async def connect(
        cls,
        host: str,
        port: int,
        *,
        network: Network = TCP_NETWORK,
        on_line_sent: LineHook | None = None,
        on_line_received: LineHook | None = None,
    ) -> IppDmeMachine:
        """Connect and wrap the result.

        ``network``, ``on_line_sent`` and ``on_line_received`` are as in
        :meth:`~pyippdme.client.IppDmeClient.connect`.
        """
        return cls(
            await IppDmeClient.connect(
                host,
                port,
                network=network,
                on_line_sent=on_line_sent,
                on_line_received=on_line_received,
            )
        )

    async def start_session(self) -> None:
        """Shortcut for ``machine.server.start_session()``."""
        await self.server.start_session()

    async def end_session(self) -> None:
        """Shortcut for ``machine.server.end_session()``."""
        await self.server.end_session()

    async def close(self) -> None:
        await self.client.close()
