# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Domain types shared by both :mod:`pyippdme.client` and :mod:`pyippdme.server`.

Coordinate-system math (:mod:`~pyippdme.types.csy`), the ``Tool.Id()``
Annex G payload (:mod:`~pyippdme.types.tool_id`), the ``AdvDataStruct``
Annex F payload (:mod:`~pyippdme.types.rawdata`), the raw point-cloud
Annex E payload (:mod:`~pyippdme.types.pointcloud` - a server writes it via
``GetRawDataFile``, 6.17.2.3, and a client reads the same file back with
:func:`~pyippdme.types.pointcloud.from_xml`), and 3D vector math
(:mod:`~pyippdme.types.vec3`) all cross the client/server boundary - a
server encodes them into responses, a client decodes the same shapes back
out of them - so they live here rather than under :mod:`pyippdme.server`
or :mod:`pyippdme.client`.
"""
