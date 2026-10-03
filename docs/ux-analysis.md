<!--
SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>

SPDX-License-Identifier: MPL-2.0
-->

# UX analysis of the two windows

Both windows were reviewed against the same questions: what is the machine doing right now, what
can I do next, and what happened to my last command? The findings below led to the current
design.

## Command client (`ippdme client-gui`)

| Finding | Effect on the user | Change |
| --- | --- | --- |
| The 110 commands were one alphabetical list that filled the left side. | Nobody knows the protocol by name; the list did not help to find a task. | Commands are grouped by task (session, move, measure a point, scan, tool, ...) in a tree; the search shows only matching groups. Double-click copies a command to the line. |
| Commands were echoed when the first reply arrived, and a status poll could land between them. | Log lines of different commands were mixed, and a repeated command was not echoed twice. | Each command has a number (`#7 > GoTo(...)`) and every reply carries it; the end of a command shows `done in 0.35 s` or the error. |
| Machine state was in a small status bar. | Homed, enabled and busy were not visible while working in a dialog or the log. | A strip above the log: lamps for connection, session, homed, user enabled and busy; the tool name; the position in large digits. |
| Abort was one toolbar icon among others. | The most urgent action was the least visible. | A red Abort button in the strip, and Esc anywhere in the window. Abort goes through while a move is still running. |
| Dialogs closed over, or left open, without telling whether the run worked. | The user looked at the log to find out. | The dialog shows Running ..., then Done with the time, or Failed with the server's error. Run is disabled while it runs. |
| Three docks on the right that were empty most of the time. | Wasted space and no hint what fills them. | One Results dock with tabs: points, point cloud (opens after an acquisition), last response. |
| An empty window after start. | No hint what to do first. | The log starts with the two ways to begin; sending while not connected explains how to connect instead of failing silently. |
| No keyboard use. | Repeated tasks need the mouse. | F5 connect, Ctrl+Shift+V virtual CMM, Ctrl+1..7 dialogs, Ctrl+L command line, Esc abort. |

Not changed on purpose: the command line stays, as the protocol is the product; the dialogs only
build lines for it, and the preview shows each line before it runs.

## Simulator (`ippdme gui`)

| Finding | Change |
| --- | --- |
| Settings, library, teaching and checking were one long form. | Tabs Machine, Tools, Teach-in, Check and Safety in a dock; the 3D view stays in the middle. |
| State was text drawn into the 3D view. | A strip above the view with lamps (server, client, homed, motion, safety), the tool and the position. |
| The camera could only orbit. | Views 1-7 (iso, top, front, side, follow the machine, the probe, the table), F fits. |
| Moving the machine needed a client. | Teach-in tab: jog buttons, go-to with a collision-free path, generate and run a probing program; the key, pick, clearance and tool-change events of the protocol (5.5.3). |
| Tools could only be picked. | A library of sensors, extensions and changers, and a tool creator with a live summary. |
| Repainting on every timer tick starved the window when a heavy part was shown. | The scene is repainted only when its state changes. |

## Still open

- CAD import of styli (parts and fixtures can be imported).
- Cameras that are attached to a chosen component, beyond follow, probe and table.
