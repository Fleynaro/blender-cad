from typing import Any, Optional

import bpy
from mathutils import Vector

from .modifiers import Mode, add
from .part import BoxSetPart, Part


class BasePrimitive:
    """Base class for all 3D primitives."""

    def __init__(self, mode: Mode = Mode.ADD):
        self.part = self.create_part()
        add(self.part, mode=mode, _make_copy=False)

    def create_part(self) -> Part:
        """Implemented in subclasses (Box, Sphere, etc.). Returns a new Part object."""
        raise NotImplementedError


class Point(BasePrimitive):
    """A single vertex primitive at local coordinates (0, 0, 0)."""

    def __init__(self, mode: Mode = Mode.JOIN):
        super().__init__(mode)

    def create_part(self) -> Part:
        mesh = bpy.data.meshes.new(name="PointMesh")
        mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
        obj = bpy.data.objects.new("Point", mesh)
        return Part(obj)


class Box(BasePrimitive):
    """A rectangular cuboid primitive."""

    def __init__(
        self,
        length: float,
        width: float,
        height: float,
        custom_data: Optional[Any] = None,
        mode: Mode = Mode.ADD,
    ):
        self.length = length
        self.width = width
        self.height = height
        self.custom_data = custom_data
        super().__init__(mode)

    def create_part(self) -> Part:
        return BoxSetPart(
            size=Vector((self.length, self.width, self.height)),
            custom_data=self.custom_data,
        )


class Plane(BasePrimitive):
    """A UV-plane primitive."""

    def __init__(self, size: float = 1.0, mode: Mode = Mode.ADD):
        self.size = size
        super().__init__(mode)

    def create_part(self) -> Part:
        bpy.ops.mesh.primitive_plane_add(size=self.size, calc_uvs=True)
        obj = bpy.context.active_object
        obj.name = "Plane"
        return Part(obj)


class Sphere(BasePrimitive):
    """A UV-sphere primitive."""

    def __init__(self, radius: float, mode: Mode = Mode.ADD):
        self.radius = radius
        super().__init__(mode)

    def create_part(self) -> Part:
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=self.radius, segments=32, ring_count=16, calc_uvs=True
        )
        obj = bpy.context.active_object
        obj.name = "Sphere"
        return Part(obj)


class IcoSphere(BasePrimitive):
    """An IcoSphere primitive."""

    def __init__(
        self, radius: float = 1.0, subdivisions: int = 2, mode: Mode = Mode.ADD
    ):
        self.radius = radius
        self.subdivisions = subdivisions
        super().__init__(mode)

    def create_part(self) -> Part:
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=self.radius, subdivisions=self.subdivisions, calc_uvs=True
        )
        obj = bpy.context.active_object
        obj.name = "IcoSphere"
        return Part(obj)


class Cone(BasePrimitive):
    """A cone primitive with independent top and bottom radii."""

    def __init__(
        self,
        radius_bottom: float,
        radius_top: float,
        height: float,
        segments: int = 32,
        mode: Mode = Mode.ADD,
    ):
        self.radius_bottom = radius_bottom
        self.radius_top = radius_top
        self.height = height
        self.segments = segments
        super().__init__(mode)

    def create_part(self) -> Part:
        bpy.ops.mesh.primitive_cone_add(
            vertices=self.segments,
            radius1=self.radius_bottom,
            radius2=self.radius_top,
            depth=self.height,
            calc_uvs=True,
        )
        obj = bpy.context.active_object
        obj.name = "Cone"
        return Part(obj)


class Cylinder(Cone):
    """A cylinder primitive (special case of a cone)."""

    def __init__(
        self, radius: float, height: float, segments: int = 32, mode: Mode = Mode.ADD
    ):
        super().__init__(
            radius_bottom=radius,
            radius_top=radius,
            height=height,
            segments=segments,
            mode=mode,
        )

    def create_part(self) -> Part:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=self.segments,
            radius=self.radius_bottom,
            depth=self.height,
            calc_uvs=True,
        )
        obj = bpy.context.active_object
        obj.name = "Cylinder"
        return Part(obj)


class Torus(BasePrimitive):
    """A torus primitive."""

    def __init__(
        self,
        major_radius: float = 1.0,
        minor_radius: float = 0.25,
        mode: Mode = Mode.ADD,
    ):
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        super().__init__(mode)

    def create_part(self) -> Part:
        bpy.ops.mesh.primitive_torus_add(
            major_radius=self.major_radius,
            minor_radius=self.minor_radius,
            segments=48,
            pipe_segments=12,
            generate_uvs=True,
        )
        obj = bpy.context.active_object
        obj.name = "Torus"
        return Part(obj)


class Grid(BasePrimitive):
    """A grid primitive."""

    def __init__(
        self, size: float = 1.0, subdivisions: int = 10, mode: Mode = Mode.ADD
    ):
        self.size = size
        self.subdivisions = subdivisions
        super().__init__(mode)

    def create_part(self) -> Part:
        bpy.ops.mesh.primitive_grid_add(
            size=self.size,
            x_subdivisions=self.subdivisions,
            y_subdivisions=self.subdivisions,
            calc_uvs=True,
        )
        obj = bpy.context.active_object
        obj.name = "Grid"
        return Part(obj)


class Monkey(BasePrimitive):
    """A monkey primitive."""

    def __init__(self, size: float = 1.0, mode: Mode = Mode.ADD):
        self.size = size
        super().__init__(mode)

    def create_part(self) -> Part:
        bpy.ops.mesh.primitive_monkey_add(size=self.size, calc_uvs=True)
        obj = bpy.context.active_object
        obj.name = "Monkey"
        return Part(obj)
