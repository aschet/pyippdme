# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Game controller input for the jog box.

The mapping from sticks and buttons to jog movements and teach-in events is plain Python
(:class:`PadMapping`) and does not need a controller. :class:`SdlGamepad` reads a real one
through SDL2 (``pip install pyippdme[gamepad]``); any other object with a ``poll`` method that
returns a :class:`PadState` can stand in, for example in tests.

Default layout: the left stick jogs X and Y, the right stick up and down jogs Z, the bumpers
make the step smaller or larger, and the buttons are the keys of the jog box: A picks a point,
B sends a clearance point, X sends ``Done``, Y sends ``F1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["GamepadSource", "PadMapping", "PadState", "SdlGamepad", "open_gamepad"]

#: Buttons, by the SDL game controller names.
BUTTONS = ("a", "b", "x", "y", "leftshoulder", "rightshoulder")


@dataclass(frozen=True)
class PadState:
    """What the controller shows at one moment: axes from -1 to 1 and the pressed buttons."""

    axes: dict[str, float] = field(default_factory=dict)
    buttons: frozenset[str] = frozenset()


class GamepadSource(Protocol):
    """Anything that can be asked for the state of a controller."""

    @property
    def name(self) -> str: ...

    def poll(self) -> PadState | None:
        """Return the current state, or ``None`` when no controller is connected."""
        ...


@dataclass
class PadMapping:
    """Turns pad states into jog movements and button presses (edge triggered)."""

    deadzone: float = 0.2
    #: Full stick deflection moves the tool this fast, in mm/s.
    speed: float = 20.0
    _previous: frozenset[str] = frozenset()

    def _stick(self, value: float) -> float:
        """Apply the dead zone and rescale so that movement starts gently."""
        if abs(value) < self.deadzone:
            return 0.0
        scaled = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        return scaled * scaled * (1 if value > 0 else -1)

    def jog(self, state: PadState, dt: float) -> tuple[float, float, float]:
        """Return the movement in mm for ``dt`` seconds of the current stick position."""
        axes = state.axes
        # Sticks report up as negative.
        dx = self._stick(axes.get("leftx", 0.0))
        dy = -self._stick(axes.get("lefty", 0.0))
        dz = -self._stick(axes.get("righty", 0.0))
        return (dx * self.speed * dt, dy * self.speed * dt, dz * self.speed * dt)

    def pressed(self, state: PadState) -> list[str]:
        """Return the buttons that went down since the last call."""
        new = sorted(state.buttons - self._previous)
        self._previous = state.buttons
        return new


class SdlGamepad:
    """The first game controller that SDL2 finds."""

    def __init__(self) -> None:
        import sdl2  # type: ignore[import-not-found,unused-ignore]

        self._sdl = sdl2
        if sdl2.SDL_Init(sdl2.SDL_INIT_GAMECONTROLLER) != 0:
            raise RuntimeError(sdl2.SDL_GetError().decode())
        self._pad: object | None = None
        self.name = "no controller"

    def _open(self) -> None:
        sdl = self._sdl
        for index in range(sdl.SDL_NumJoysticks()):
            if sdl.SDL_IsGameController(index):
                self._pad = sdl.SDL_GameControllerOpen(index)
                self.name = sdl.SDL_GameControllerName(self._pad).decode()
                return

    def poll(self) -> PadState | None:
        """Return the stick positions and pressed buttons, or ``None`` without a controller."""
        sdl = self._sdl
        sdl.SDL_PumpEvents()
        if self._pad is None:
            self._open()
        elif not sdl.SDL_GameControllerGetAttached(self._pad):
            sdl.SDL_GameControllerClose(self._pad)
            self._pad = None
            self.name = "no controller"
        if self._pad is None:
            return None
        axes = {}
        for axis in ("leftx", "lefty", "rightx", "righty"):
            code = sdl.SDL_GameControllerGetAxisFromString(axis.encode())
            axes[axis] = max(-1.0, sdl.SDL_GameControllerGetAxis(self._pad, code) / 32767.0)
        pressed = set()
        for button in BUTTONS:
            code = sdl.SDL_GameControllerGetButtonFromString(button.encode())
            if sdl.SDL_GameControllerGetButton(self._pad, code):
                pressed.add(button)
        return PadState(axes, frozenset(pressed))

    def close(self) -> None:
        """Release the controller and SDL."""
        if self._pad is not None:
            self._sdl.SDL_GameControllerClose(self._pad)
            self._pad = None
        self._sdl.SDL_QuitSubSystem(self._sdl.SDL_INIT_GAMECONTROLLER)


def open_gamepad() -> SdlGamepad | None:
    """Return an SDL controller reader, or ``None`` when SDL2 is not installed or fails."""
    try:
        return SdlGamepad()
    except (ImportError, RuntimeError, OSError):
        return None
