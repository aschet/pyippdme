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
``.tool``, ``.tool_changer``, ``.scanning``, ``.form_tester``, ``.mover``,
``.rotary_table``, ``.part``, ``.raw_data``, mirroring how the server side
already exposes that same object model as a set of command classes
(:mod:`pyippdme.server.classes`).

Only the subset of each class this library's own reference/simulated server
implements (see each ``pyippdme.server.classes.*_class`` module) is covered;
talking to a different, more complete server, a caller can always fall back
to ``machine.client.call(...)`` for anything not wrapped here.

Every method here is a thin encode/decode layer over ``IppDmeClient.call()``;
none of it changes wire behaviour.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from pyippdme.client import IppDmeClient, LineHook, builders
from pyippdme.client.builders import CurvePoint as CurvePoint
from pyippdme.protocol.ast import (
    Argument,
    BasicName,
    DataPayload,
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

    async def get_error_info(self, error_number: int) -> str:
        data = await self._client.call(CommandName.GET_ERROR_INFO, Number.of(error_number))
        return _string_value(_only(data))

    async def get_err_status(self) -> bool:
        (value,) = _numeric(_only(await self._client.call(CommandName.GET_ERR_STATUS_E)))
        return value != 0.0

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

    async def get_dme_version(self) -> str:
        payload = _only(await self._client.call(CommandName.GET_DME_VERSION))
        (value,) = _items(payload)
        (version,) = value.args
        if not isinstance(version, String):
            raise TypeError(f"Expected a string DMEVersion, got {type(version).__name__}")
        return version.value

    async def get_supported_commands(self) -> tuple[str, ...]:
        return _string_values(await self._client.call(CommandName.GET_SUPPORTED_COMMANDS))

    async def get_supported_arguments(self, command_name: str) -> dict[str, str]:
        data = await self._client.call(CommandName.GET_SUPPORTED_ARGUMENTS, String(command_name))
        return dict(_property_pairs(data))

    async def get_machine_class(self) -> str:
        return _string_value(_only(await self._client.call(CommandName.GET_MACHINE_CLASS)))

    async def home(self) -> None:
        await self._client.call(CommandName.HOME)

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

    async def set_coord_system(self, csy: str) -> None:
        await self._client.call(CommandName.SET_COORD_SYSTEM, BasicName(csy))

    async def get_coord_system(self) -> str:
        return _name_value(_only(await self._client.call(CommandName.GET_COORD_SYSTEM)))

    async def get(self, *axes: str) -> dict[str, float]:
        """Query one or more axes, e.g. ``await cart_cmm.get("X", "Y", "R")``."""
        payload = _only(await self._client.call(CommandName.GET, *builders.get(*axes)))
        return _named_numbers(payload)

    async def get_position(self) -> Vec3:
        values = await self.get("X", "Y", "Z")
        return (values["X"], values["Y"], values["Z"])

    async def go_to(
        self, x: float | None = None, y: float | None = None, z: float | None = None
    ) -> None:
        await self._client.call(CommandName.GO_TO, *builders.go_to(x, y, z))

    async def step(
        self, x: float | None = None, y: float | None = None, z: float | None = None
    ) -> None:
        """Perform a relative move (6.8.1): add to the current position, not replace it."""
        await self._client.call(CommandName.STEP, *builders.step(x, y, z))

    async def go_to_on_circle(self, center: Vec3, ijk: Vec3, target: Vec3) -> None:
        await self._client.call(
            CommandName.GO_TO_ON_CIRCLE, *builders.go_to_on_circle(center, ijk, target)
        )

    async def go_to_on_spiral(self, center: Vec3, ijk: Vec3, target: Vec3) -> None:
        await self._client.call(
            CommandName.GO_TO_ON_SPIRAL, *builders.go_to_on_spiral(center, ijk, target)
        )

    async def pt_meas(
        self, x: float | None = None, y: float | None = None, z: float | None = None
    ) -> dict[str, float]:
        payload = _only(await self._client.call(CommandName.PT_MEAS, *builders.pt_meas(x, y, z)))
        return _named_numbers(payload)

    async def set_csy_transformation(self, csy: str, transform: CoordinateTransform) -> None:
        await self._client.call(
            CommandName.SET_CSY_TRANSFORMATION,
            BasicName(csy),
            *builders.set_csy_transformation(transform),
        )

    async def get_csy_transformation(self, csy: str) -> CoordinateTransform:
        payload = _only(await self._client.call(CommandName.GET_CSY_TRANSFORMATION, BasicName(csy)))
        return _transform_from_items(payload)

    async def save_named_csy_transformation(
        self, name: str, transform: CoordinateTransform
    ) -> None:
        await self._client.call(
            CommandName.SAVE_NAMED_CSY_TRANSFORMATION,
            String(name),
            *builders.set_csy_transformation(transform),
        )

    async def get_named_csy_transformation(self, name: str) -> CoordinateTransform:
        payload = _only(
            await self._client.call(CommandName.GET_NAMED_CSY_TRANSFORMATION, String(name))
        )
        return _transform_from_items(payload)

    async def save_active_coord_system(self, name: str) -> None:
        await self._client.call(CommandName.SAVE_ACTIVE_COORD_SYSTEM, String(name))

    async def load_coord_system(self, name: str) -> None:
        await self._client.call(CommandName.LOAD_COORD_SYSTEM, String(name))

    async def delete_coord_system(self, name: str) -> None:
        await self._client.call(CommandName.DELETE_COORD_SYSTEM, String(name))

    async def enum_coord_systems(self) -> tuple[str, ...]:
        return _string_values(await self._client.call(CommandName.ENUM_COORD_SYSTEMS))

    async def get_temperature_sensors(self) -> tuple[TemperatureSensorInfo, ...]:
        data = await self._client.call(CommandName.GET_TEMPERATURE_SENSORS)
        return tuple(_temperature_sensor_info(payload) for payload in data)

    async def read_temperature_sensor(self, name: str) -> float:
        payload = _only(await self._client.call(CommandName.READ_TEMPERATURE_SENSOR, String(name)))
        (value,) = _numeric(payload)
        return value

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


class Tool:
    """A subset of the ``Tool`` class (6.10): the parameter blocks and ``Id()``."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    async def get_name(self) -> str:
        return _name_value(_only(await self._client.call(CommandName.TOOL)))

    async def is_alignable(self) -> bool:
        (value,) = _numeric(_only(await self._client.call(CommandName.IS_ALIGNABLE)))
        return value != 0.0

    async def get_id(self) -> ToolId:
        payload = _only(
            await self._client.call(CommandName.GET_PROP, *builders.get_prop("Tool.Id"))
        )
        (value,) = _items(payload)
        if value.xml is None:
            raise TypeError("Tool.Id() did not return an XML payload")
        return tool_id_from_xml(value.xml.raw)

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

    async def set_parameter(self, block: str, name: str, value: float) -> None:
        await self._client.call(
            CommandName.SET_PROP, *builders.set_prop(f"Tool.{block}.{name}", value)
        )


class Scanning:
    """A subset of the ``Scanning`` class (6.13.2.1-6.13.2): known-contour scans."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    async def scan_on_line(
        self, start: Vec3, end: Vec3, direction: Vec3, step_width: float
    ) -> AsyncIterator[Vec3]:
        """Scan a straight line, yielding each measured point as it arrives.

        Each streamed ``ScanOnLine`` response is a *bare* ``NumericData`` -
        positional values in whatever order ``OnScanReport(...)`` last
        established (6.13.2), not named fields - so this fixes that order to
        ``X(), Y(), Z()`` itself, rather than accepting/trusting a
        previously-set ``OnScanReport`` the caller might have configured
        differently.
        """
        await self._client.call(CommandName.ON_SCAN_REPORT, *builders.on_scan_report("X", "Y", "Z"))
        args = tuple(Number.of(v) for v in (*start, *end, *direction, step_width))
        async for payload in self._client.call_streaming(CommandName.SCAN_ON_LINE, *args):
            x, y, z = _numeric(payload)
            yield (x, y, z)

    async def scan_on_curve(
        self, points: Sequence[CurvePoint], *, closed: bool = False
    ) -> AsyncIterator[Vec3]:
        """Scan a client-supplied nominal curve, yielding each measured point.

        See :mod:`pyippdme.simulation.classes.scanning_class`'s module docstring for
        the confirmed wire encoding and this implementation's scope (the
        mandatory position/orientation/tag ``Format`` columns only).
        """
        await self._client.call(CommandName.ON_SCAN_REPORT, *builders.on_scan_report("X", "Y", "Z"))
        args = builders.scan_on_curve(points, closed=closed)
        async for payload in self._client.call_streaming(CommandName.SCAN_ON_CURVE, *args):
            x, y, z = _numeric(payload)
            yield (x, y, z)


class FormTester:
    """A subset of the ``FormTester`` class (6.6.1)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    async def _status(self, command: str, *values: float) -> bool:
        data = await self._client.call(command, *(Number.of(v) for v in values))
        (achieved,) = _numeric(_only(data))
        return achieved != 0.0

    async def center_part(self, x: float, y: float, z: float, limit: float) -> bool:
        return await self._status(CommandName.CENTER_PART, x, y, z, limit)

    async def tilt_part(self, i: float, j: float, k: float, limit: float) -> bool:
        return await self._status(CommandName.TILT_PART, i, j, k, limit)

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

    async def lock_axis(self, *axes: str) -> None:
        await self._client.call(CommandName.LOCK_AXIS, *builders.lock_axis(*axes))

    async def lock_position(self, *positions: str) -> None:
        await self._client.call(CommandName.LOCK_POSITION, *builders.lock_position(*positions))


class Mover:
    """The ``Mover`` class (6.7.1)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    async def enable_user(self) -> None:
        await self._client.call(CommandName.ENABLE_USER)

    async def disable_user(self) -> None:
        await self._client.call(CommandName.DISABLE_USER)

    async def is_user_enabled(self) -> bool:
        payload = _only(await self._client.call(CommandName.IS_USER_ENABLED))
        return _named_numbers(payload)[CommandName.IS_USER_ENABLED] != 0.0

    async def enumerate_mover_axes(self) -> tuple[str, ...]:
        payload = _only(await self._client.call(CommandName.ENUMERATE_MOVER_AXES))
        return tuple(nv.name for nv in _items(payload))

    async def set_scale_temperatures(self, **temperatures: float) -> None:
        await self._client.call(
            CommandName.SET_SCALE_TEMPERATURES, *builders.set_scale_temperatures(**temperatures)
        )

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

    async def enum_tools(self) -> tuple[str, ...]:
        data = await self._client.call(CommandName.ENUM_TOOLS)
        return tuple(_name_value(item) for item in data)

    async def change_tool(self, tool_name: str) -> None:
        await self._client.call(CommandName.CHANGE_TOOL, String(tool_name))

    async def find_tool(self, tool_name: str) -> None:
        await self._client.call(CommandName.FIND_TOOL, String(tool_name))

    async def found_tool(self) -> str:
        return _name_value(_only(await self._client.call(CommandName.FOUND_TOOL)))

    async def set_tool(self, tool_name: str) -> None:
        await self._client.call(CommandName.SET_TOOL, String(tool_name))

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

    async def get_temperature(self) -> float:
        return await self._get_float("Part.Temperature")

    async def set_temperature(self, value: float) -> None:
        await self._set_float("Part.Temperature", value)

    async def get_xpan_coefficient(self) -> float:
        return await self._get_float("Part.XpanCoefficient")

    async def set_xpan_coefficient(self, value: float) -> None:
        await self._set_float("Part.XpanCoefficient", value)

    async def get_approach(self) -> float:
        return await self._get_float("Part.Approach")

    async def set_approach(self, value: float) -> None:
        await self._set_float("Part.Approach", value)

    async def get_search(self) -> float:
        return await self._get_float("Part.Search")

    async def set_search(self, value: float) -> None:
        await self._set_float("Part.Search", value)

    async def get_retract(self) -> float:
        return await self._get_float("Part.Retract")

    async def set_retract(self, value: float) -> None:
        await self._set_float("Part.Retract", value)


class RotaryTable:
    """A subset of the ``RotaryTable`` class (6.23)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    async def enable_rotary_table_var_csy(self, enabled: bool) -> None:
        await self._client.call(
            CommandName.ENABLE_ROTARY_TABLE_VAR_CSY, Number.of(1 if enabled else 0)
        )

    async def align_part(
        self, part_vector: Vec3, machine_vector: Vec3, alpha: float
    ) -> tuple[Vec3, Vec3]:
        args = tuple(Number.of(v) for v in (*part_vector, *machine_vector, alpha))
        n = _numeric(_only(await self._client.call(CommandName.ALIGN_PART, *args)))
        return (n[0], n[1], n[2]), (n[3], n[4], n[5])


class RawDataHandling:
    """A subset of the ``Optical``/``RawDataHandling`` classes (6.15, 6.17)."""

    def __init__(self, client: IppDmeClient) -> None:
        self._client = client

    async def get_adv_data_struct(self) -> AdvDataStruct:
        payload = _only(await self._client.call(CommandName.ADV_DATA_STRUCT))
        if not isinstance(payload, Xml):
            raise TypeError(f"Expected an Xml response, got {type(payload).__name__}")
        return adv_data_struct_from_xml(payload.raw)

    async def data_acquire(
        self,
        acq_name: str,
        acquisition_type: str,
        settings_name: str,
        points: Sequence[tuple[Vec3, Vec3]] | None = None,
    ) -> None:
        """Trigger an acquisition (6.15.1).

        With no ``points``, this is a single-shot acquisition at the current
        position (``n=0``). ``points`` gives the scan path as
        ``(position, direction)`` control points instead - for
        ``acquisition_type="Sweep"``, the server densifies the path between
        them into a line of points rather than reporting only these few
        (see :mod:`pyippdme.simulation.classes.rawdata_class`'s module docstring).
        """
        if points is None:
            await self._client.call(
                CommandName.DATA_ACQUIRE,
                String(acq_name),
                BasicName(acquisition_type),
                String(settings_name),
                Number.of(0),
            )
            return
        numbers = tuple(
            Number.of(v) for position, direction in points for v in (*position, *direction)
        )
        await self._client.call(
            CommandName.DATA_ACQUIRE,
            String(acq_name),
            BasicName(acquisition_type),
            String(settings_name),
            Number.of(len(points)),
            *numbers,
        )

    async def get_raw_data_file(self, acq_name: str) -> str:
        """Return the ``file://`` URL the acquisition's data was written to."""
        payload = _only(await self._client.call(CommandName.GET_RAW_DATA_FILE, String(acq_name)))
        return _named_strings(payload)["FileURL"]

    async def del_raw_data_file(self, acq_name: str) -> None:
        await self._client.call(CommandName.DEL_RAW_DATA_FILE, String(acq_name))

    async def delete_acquisition(self, acq_name: str) -> None:
        await self._client.call(CommandName.DELETE_ACQUISITION, String(acq_name))

    async def delete_all_acquisitions(self) -> None:
        await self._client.call(CommandName.DELETE_ALL_ACQUISITIONS)

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

    async def release_sha_mem(self, acq_name: str) -> None:
        await self._client.call(CommandName.RELEASE_SHA_MEM, String(acq_name))


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
        self.tool_changer = ToolChanger(client)
        self.scanning = Scanning(client)
        self.form_tester = FormTester(client)
        self.mover = Mover(client)
        self.rotary_table = RotaryTable(client)
        self.part = Part(client)
        self.raw_data = RawDataHandling(client)

    @classmethod
    async def connect(
        cls,
        host: str,
        port: int,
        *,
        on_line_sent: LineHook | None = None,
        on_line_received: LineHook | None = None,
    ) -> IppDmeMachine:
        """Connect and wrap the result.

        ``on_line_sent``/``on_line_received`` are as in
        :meth:`~pyippdme.client.IppDmeClient.connect`.
        """
        return cls(
            await IppDmeClient.connect(
                host, port, on_line_sent=on_line_sent, on_line_received=on_line_received
            )
        )

    async def start_session(self) -> None:
        await self.client.start_session()

    async def end_session(self) -> None:
        await self.client.end_session()

    async def close(self) -> None:
        await self.client.close()
