import fnmatch
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypeAlias,
    Union,
)
from weakref import WeakSet, WeakValueDictionary

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .common import Axis, VectorLike, extract_part, extract_vector
from .location import Location, Locations, Pos, Scale, Transform, TransformExpr
from .material import build_material, mat

if TYPE_CHECKING:
    from .geometry import UVSelector
    from .joint import Joint
    from .part import BoxSetPart, Part

AttributeDomainItems: TypeAlias = Literal["POINT", "EDGE", "FACE", "CURVE"]


class CustomAttributeItem:
    """Emulates a single Blender attribute data element storing binary string data."""

    __slots__ = ("value",)

    def __init__(self, value: bytes = b""):
        self.value: bytes = value


class CustomAttribute:
    """Emulates Blender's Attribute object with a .data list interface."""

    def __init__(self, domain: AttributeDomainItems):
        self.domain = domain
        self.data: list[CustomAttributeItem] = []

    def ensure_size(self, size: int):
        """Ensures the internal storage array matches the requested geometry element count."""
        while len(self.data) < size:
            self.data.append(CustomAttributeItem())


@dataclass
class JointRegistration:
    """
    Named joint exposed by an object.
    """

    name: str
    joint: "Joint"
    propagate: bool = True


class Object(ABC):
    """An object representing a part. Manages its own mesh and Blender object."""

    @dataclass
    class BBox:
        min: Vector = field(default_factory=lambda: Vector((0.0, 0.0, 0.0)))
        max: Vector = field(default_factory=lambda: Vector((0.0, 0.0, 0.0)))

    FACE_TAG_ATTR = "_tag_face"
    EDGE_TAG_ATTR = "_tag_edge"
    VERT_TAG_ATTR = "_tag_vert"
    CURVE_TAG_ATTR = "_tag_curve"
    TAG_SEPARATOR = "|"

    TAG_OP_EXTRUDE = "op:extrude"
    TAG_ML_BACKGROUND = "ml:background"
    TAG_ML_BORDER = "ml:border"

    def __init__(self, obj: Optional[bpy.types.Object] = None):
        self.obj: Optional[bpy.types.Object] = obj or self._create_empty_object()
        self._auto_remove = True
        self._bbox_part: Optional["Part"] = None
        self._bbox_signature: Optional[int] = None
        self._convex_hull_part: Optional["Part"] = None
        self._convex_hull_signature: Optional[int] = None
        self._joints: WeakSet["Joint"] = WeakSet()
        self._joints_by_id: WeakValueDictionary[int, "Joint"] = WeakValueDictionary()
        self._joint_seq = 0
        self._joint_registry: list[JointRegistration] = []
        self._custom_attributes: dict[str, CustomAttribute] = {}

    def __del__(self):
        if self._auto_remove:
            self.remove()

    @property
    def is_valid(self):
        """Checks if the object is valid (not removed)."""
        return self.obj is not None

    @property
    def is_physically_valid(self):
        """Checks if the object is valid (not removed)."""
        if not self.is_valid:
            return False
        try:
            _ = self.obj.name
        except ReferenceError:
            return False
        return True

    @property
    def transform(self):
        """Access the transformation of the part."""
        # We decompose matrix_world but use self.obj.scale directly because
        # matrix_world is updated lazily by Blender and might hold stale scale
        # data until the next dependency graph update.
        loc, rot, _ = self.obj.matrix_world.decompose()
        scale = self.obj.scale
        mat = Matrix.LocRotScale(loc, rot, scale)
        return Transform(mat)

    @transform.setter
    def transform(self, value: "TransformExpr"):
        """Sets the transformation of the part."""
        self.obj.matrix_world = value.resolve(self).matrix

    @property
    def loc(self):
        """Access the location of the part."""
        return self.transform.loc

    @loc.setter
    def loc(self, loc: "TransformExpr"):
        """Sets the location of the part."""
        self.transform = loc * Scale(self.scale)

    @property
    def scale(self):
        """Access the scale of the part. It allows to change the size by x, y, z."""
        return self.obj.scale

    @scale.setter
    def scale(self, value: VectorLike):
        """
        Sets the scale of the part.
        """
        self.obj.scale = extract_vector(value)

    @property
    def size(self):
        """Access the size of the part. It allows to change the size by x, y, z."""
        return self.obj.dimensions

    @size.setter
    def size(self, value: VectorLike):
        """Sets the size of the part."""
        self.obj.dimensions = extract_vector(value)

    @property
    def orig_size(self):
        """Access the original size of the part."""
        return Vector([self.size[i] / self.scale[i] for i in range(3)])

    @property
    def bbox(self):
        """Access the bounding box of the part."""
        return self._bbox_from_matrix(self.transform.matrix)

    @property
    def local_bbox(self):
        """Access the local bounding box of the part."""
        return self._bbox_from_matrix()

    def get_bbox_set_part(self, custom_data: Optional[Any] = None) -> "BoxSetPart":
        """Box Set Part representing this object's bounding box."""
        from .build_part import BuildPart, Mode
        from .part import Part
        from .primitives import Box

        with BuildPart(part=Part.box_set_empty(), mode=Mode.PRIVATE) as bp:
            bb = self.local_bbox
            box_w = max(bb.max.x - bb.min.x, 1e-6)
            box_h = max(bb.max.y - bb.min.y, 1e-6)
            box_d = max(bb.max.z - bb.min.z, 1e-6)
            center_offset = Pos(
                X=(bb.min.x + bb.max.x) / 2.0,
                Y=(bb.min.y + bb.max.y) / 2.0,
                Z=(bb.min.z + bb.max.z) / 2.0,
            )
            with Locations(center_offset):
                Box(box_w, box_h, box_d, custom_data=custom_data)
            bp.transform = self.transform
        return bp.part

    @property
    def bbox_part(self) -> "Part":
        """Part representing this object's bounding box."""
        import bmesh

        from .part import Part

        if not self.is_valid:
            raise RuntimeError("Object is removed")

        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(dg)

        raw_verts = [list(v) for v in eval_obj.bound_box]

        xs = [v[0] for v in raw_verts]
        ys = [v[1] for v in raw_verts]
        zs = [v[2] for v in raw_verts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        eps = 1e-6
        half_eps = eps / 2.0

        if (max_x - min_x) < eps:
            min_x -= half_eps
            max_x += half_eps
        if (max_y - min_y) < eps:
            min_y -= half_eps
            max_y += half_eps
        if (max_z - min_z) < eps:
            min_z -= half_eps
            max_z += half_eps

        verts = [
            (min_x, min_y, min_z),
            (min_x, min_y, max_z),
            (min_x, max_y, max_z),
            (min_x, max_y, min_z),
            (max_x, min_y, min_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (max_x, max_y, min_z),
        ]

        signature = hash(tuple(round(c, 6) for v in verts for c in v))

        rebuild = (
            self._bbox_part is None
            or not self._bbox_part.is_valid
            or self._bbox_signature != signature
        )

        if rebuild:
            mesh = bpy.data.meshes.new(f"{self.obj.name}_bbox")
            faces = [
                (0, 1, 2, 3),
                (4, 5, 6, 7),
                (0, 4, 5, 1),
                (1, 5, 6, 2),
                (2, 6, 7, 3),
                (3, 7, 4, 0),
            ]
            mesh.from_pydata(verts, [], faces)

            # Recalculate normals
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(mesh)
            bm.free()

            mesh.update()

            obj = bpy.data.objects.new(f"{self.obj.name}_bbox", mesh)
            self._bbox_part = Part(obj)
            self._bbox_part.mat = mat.red + mat.PBR(alpha=0.5)
            self._bbox_signature = signature

        self._bbox_part.transform = self.transform
        return self._bbox_part

    @property
    def convex_hull_part(self) -> "Part":
        """Part representing this object's convex hull (more accurate alternative to bbox)."""
        import bmesh

        from .part import Part

        if not self.is_valid:
            raise RuntimeError("Object is removed")

        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(dg)
        mesh_data = eval_obj.to_mesh()

        bm = bmesh.new()
        bm.from_mesh(mesh_data)

        if len(bm.verts) >= 3:
            bmesh.ops.convex_hull(bm, input=bm.verts)

        bm.verts.ensure_lookup_table()

        hull_verts = sorted([tuple(v.co) for v in bm.verts])
        signature = hash(tuple(round(c, 6) for v in hull_verts for c in v))

        rebuild = (
            self._convex_hull_part is None
            or not self._convex_hull_part.is_valid
            or self._convex_hull_signature != signature
        )

        if rebuild:
            mesh = bpy.data.meshes.new(f"{self.obj.name}_convex_hull")
            bm.to_mesh(mesh)
            mesh.update()

            obj = bpy.data.objects.new(f"{self.obj.name}_convex_hull", mesh)
            self._convex_hull_part = Part(obj)
            self._convex_hull_part.mat = mat.red + mat.PBR(alpha=0.5)
            self._convex_hull_signature = signature

        bm.free()
        eval_obj.to_mesh_clear()

        self._convex_hull_part.transform = self.transform
        return self._convex_hull_part

    def project_2d(
        self,
        plane: Axis = Axis.Z,
        dissolve_limit: float = 0.001,
        remove_doubles: bool = True,
    ) -> "Part":
        """Creates a flat 2D silhouette projection of this object onto the selected plane."""
        import bmesh

        from .modifiers import _apply_transform
        from .part import Part

        if not self.is_valid:
            raise RuntimeError("Object is removed")

        # Get the evaluated mesh to account for any active modifiers
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(dg)
        rot_loc = self.transform.rotation_loc
        mesh_data = eval_obj.to_mesh()

        # Create a temporary BMesh from the evaluated mesh data
        bm = bmesh.new()
        bm.from_mesh(mesh_data)

        _apply_transform(bm, bm.verts, rot_loc)

        # Flatten the coordinates along the selected projection plane axis
        for v in bm.verts:
            if plane == Axis.X:
                v.co.x = 0.0
            elif plane == Axis.Y:
                v.co.y = 0.0
            else:
                v.co.z = 0.0

        # Delete degenerate faces that collapsed into flat lines or points
        bm.faces.ensure_lookup_table()
        degenerate_faces = [f for f in bm.faces if f.calc_area() < 1e-6]
        bmesh.ops.delete(bm, geom=degenerate_faces, context="FACES")

        # Merge overlapping vertices created by the projection flattening
        if remove_doubles:
            bm.verts.ensure_lookup_table()
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-5)

        # Dissolve inner edges between coplanar faces to form a clean silhouette outline
        if dissolve_limit > 0.0:
            bm.edges.ensure_lookup_table()
            bmesh.ops.dissolve_limit(
                bm,
                angle_limit=dissolve_limit,
                verts=list(bm.verts),
                edges=list(bm.edges),
            )

        # Create a new Blender mesh for the 2D shape
        new_mesh = bpy.data.meshes.new(f"{self.obj.name}_2d_mesh")
        bm.to_mesh(new_mesh)
        new_mesh.update()

        # Clean up temporary BMesh and evaluated mesh data references
        bm.free()
        eval_obj.to_mesh_clear()

        # Instantiate the new Blender object and wrap it into a Part
        new_obj = bpy.data.objects.new(f"{self.obj.name}_2d_{plane}", new_mesh)
        new_part = Part(new_obj)
        new_part.transform = self.transform * rot_loc.inverse
        return new_part

    def build_bvh(self, ensure_outwards_normals: bool = True):
        """
        Returns a BVH tree for the evaluated mesh of the object.
        Handles edge-only and vertex-only objects by generating degenerate polygons.
        """
        import bmesh

        if not self.is_valid:
            raise RuntimeError("Object is removed")

        # Get evaluated object to account for modifiers and curve-to-mesh conversion
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(dg)
        mesh_data = eval_obj.to_mesh()

        # Load evaluated mesh data into BMesh
        bm = bmesh.new()
        bm.from_mesh(mesh_data)

        if bm.faces:
            # Ensure all face normals look outwards
            if ensure_outwards_normals:
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

            # Extract flipped geometry
            bm.verts.ensure_lookup_table()
            vertices = [v.co.copy() for v in bm.verts]
            polygons = [[v.index for v in f.verts] for f in bm.faces]

        elif bm.edges:
            # Edge-only case: Degenerate polygons
            # TODO: ensure outwards normals
            bm.verts.ensure_lookup_table()
            vertices = [v.co.copy() for v in bm.verts]
            polygons = [
                (e.verts[0].index, e.verts[1].index, e.verts[0].index) for e in bm.edges
            ]

        elif bm.verts:
            # Vertex-only case: Degenerate polygons
            # TODO: ensure outwards normals
            bm.verts.ensure_lookup_table()
            vertices = [v.co.copy() for v in bm.verts]
            polygons = [(v.index, v.index, v.index) for v in bm.verts]

        else:
            bm.free()
            eval_obj.to_mesh_clear()
            raise RuntimeError("Object has no geometry")

        # Clean up memory
        bm.free()
        eval_obj.to_mesh_clear()

        # Build BVH tree with switched normals
        bvh = BVHTree.FromPolygons(vertices, polygons)
        return bvh

    @property
    def empty(self):
        return self.size == Vector()

    @property
    def _deformable_joints(self):
        return [j for j in list(self._joints) if j.deformable]

    @property
    def joints(self) -> list[JointRegistration]:
        """
        Returns registered joints.
        """
        return list(self._joint_registry)

    def joint(
        self, loc: Union["Location", Callable[[], "Location"]], deformable: bool = False
    ):
        from .joint import Joint

        if callable(loc):
            from .build_part import BuildPart, Mode

            with BuildPart(extract_part(self), mode=Mode.PRIVATE) as ctx:
                return Joint(loc(), self, deformable=deformable)
        return Joint(loc, self, deformable=deformable)

    def bbox_joint(
        self,
        axis: Axis | Vector,
        selector: Optional["UVSelector"] = None,
        deformable: bool = False,
    ):
        from .geometry import uv

        return self.joint(
            self.bbox_part.faces().group_by(axis)[-1][0].at(selector or uv),
            deformable=deformable,
        )

    def joints_by_name(self, *names: str) -> list["Joint"]:
        """
        Returns a list of registered joints matching the specified names or patterns.
        Results are returned in the exact order the names were provided in the arguments.
        """
        result: list["Joint"] = []

        for target in names:
            if "*" in target:
                for item in self._joint_registry:
                    if fnmatch.fnmatchcase(item.name, target):
                        result.append(item.joint)
            else:
                for item in self._joint_registry:
                    if item.name == target:
                        result.append(item.joint)
                        break

        return result

    def joint_by_name(self, name: str) -> "Joint":
        """
        Returns a registered joint by name.
        """
        joints = self.joints_by_name(name)
        if joints:
            return joints[0]
        raise KeyError(f"Joint '{name}' not found")

    def has_joint(self, name: str) -> bool:
        return any(x.name == name for x in self._joint_registry)

    def register_joint(
        self,
        name: str,
        joint: "Joint",
        propagate: bool = False,
    ):
        """
        Registers a named joint on this object.
        """
        if joint.object != self:
            raise ValueError(f"Joint '{name}' is registered on a different object")

        existing = next((x for x in self._joint_registry if x.name == name), None)
        if existing:
            self._joint_registry.remove(existing)

        self._joint_registry.append(
            JointRegistration(
                name=name,
                joint=joint,
                propagate=propagate,
            )
        )
        return joint

    def _transfer_registered_joints(
        self, target: "Object", propagate_only: bool = False
    ):
        """
        Transfers named joints preserving
        their world-space transforms.
        """
        for entry in self._joint_registry:
            if propagate_only and not entry.propagate:
                continue
            target.register_joint(
                name=entry.name,
                joint=target.joint(
                    entry.joint.loc,
                    deformable=entry.joint.deformable,
                ),
                propagate=entry.propagate,
            )

    def _bbox_from_matrix(self, matrix: Matrix = Matrix.Identity(4)) -> BBox:
        """Access the bounding box of the part."""
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(dg)
        world_corners = [matrix @ Vector(corner) for corner in eval_obj.bound_box]
        xs = [c.x for c in world_corners]
        ys = [c.y for c in world_corners]
        zs = [c.z for c in world_corners]
        return self.BBox(
            min=Vector((min(xs), min(ys), min(zs))),
            max=Vector((max(xs), max(ys), max(zs))),
        )

    @abstractmethod
    def _create_empty_object(self):
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> "Object":
        raise NotImplementedError

    def _after_copy(self, cloned: "Object"):
        self._transfer_registered_joints(cloned)

    def remove(self, physical=True):
        """Safely removes the object and its data from the Blender scene."""
        if self.is_valid:
            if physical and self.is_physically_valid:
                bpy.data.objects.remove(self.obj, do_unlink=True)
            self.obj = None

    def show(
        self, name: str | None = None, hide=False, collection_name: str | None = None
    ):
        """Displays the object in the Blender scene."""
        if not self.is_valid:
            raise RuntimeError("Object is removed")

        if name is not None:
            old_obj = bpy.data.objects.get(name)
            if old_obj:
                bpy.data.objects.remove(old_obj, do_unlink=True)

            self.obj.name = name

        target_col = bpy.context.collection
        if collection_name is not None:
            target_col = bpy.data.collections.get(collection_name)
            if target_col is None:
                target_col = bpy.data.collections.new(collection_name)
                bpy.context.scene.collection.children.link(target_col)

        if self.obj.name not in target_col.objects:
            target_col.objects.link(self.obj)

        self._auto_remove = False

        if hide:
            self.obj.hide_set(True)

    def shoot(
        self,
        camera_loc: Location,
        fov: float = 39.6,
        resolution: tuple[int, int] = (1024, 1024),
        offset: Location = Location(),
        label: str | None = None,
    ):
        """
        Temporarily isolates the object by hiding everything else in the scene,
        captures it with a camera, and returns a CameraTexture.
        """
        if not self.is_valid:
            raise RuntimeError("Part object is removed or invalid")

        scene_col = bpy.context.scene.collection

        # 1. Track original state of the target object
        is_linked_originally = self.obj.name in scene_col.objects
        if not is_linked_originally:
            scene_col.objects.link(self.obj)

        orig_loc = self.loc
        original_hide_render = self.obj.hide_render
        original_hide_viewport = self.obj.hide_get()

        # 2. ISOLATION LOGIC: Hide all other objects from rendering
        # We store original states to avoid unhiding objects that were already hidden
        other_objects_render_states: dict[bpy.types.Object, bool] = {}
        for obj in bpy.data.objects:
            if obj != self.obj:
                other_objects_render_states[obj] = obj.hide_render
                obj.hide_render = True

        # 3. Prepare target object for the shot
        self.loc = offset * self.loc
        self.obj.hide_render = False
        self.obj.hide_set(False)

        try:
            cam_tex = mat.CameraTex(
                location=offset * camera_loc,
                fov=fov,
                resolution=resolution,
                label=label or f"shoot_{self.obj.name}",
            )
            cam_tex.build_image()
            return cam_tex
        finally:
            # 4. RESTORATION LOGIC: Restore render visibility for all objects
            for obj, state in other_objects_render_states.items():
                # Check if object still exists to avoid reference errors
                try:
                    obj.hide_render = state
                except ReferenceError:
                    continue

            # Restore target object's state
            self.loc = orig_loc
            self.obj.hide_render = original_hide_render
            self.obj.hide_set(original_hide_viewport)

            if not is_linked_originally:
                try:
                    scene_col.objects.unlink(self.obj)
                except RuntimeError:
                    pass  # Already unlinked or handled elsewhere

    def _get_or_create_material_index(
        self, material: Optional["mat.Layer"], default: bool = False
    ) -> int:
        """Adds a material to the object and returns its index."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        if material is None:
            return 0
        bpy_mat = build_material(material)

        # Check if this material already exists in the object's slots
        for i, slot in enumerate(self.obj.material_slots):
            if slot.material == bpy_mat:
                return i

        # If not found, add it to a new slot
        if len(self.obj.data.materials) == 0:
            self.obj.data.materials.append(None)
        if default:
            self.obj.material_slots[0].material = bpy_mat
            return 0
        self.obj.data.materials.append(bpy_mat)
        return len(self.obj.data.materials) - 1

    def _alloc_joint_id(self) -> int:
        """Allocates a unique joint id inside this object."""
        self._joint_seq += 1
        return self._joint_seq

    def _register_joint(self, joint: "Joint"):
        """Registers a joint in both live and id-based registries."""
        self._joints.add(joint)
        self._joints_by_id[joint._joint_id] = joint

    def _get_joint_by_id(self, joint_id: int):
        """Returns a joint by its stored layer id."""
        return self._joints_by_id.get(joint_id)

    def _ensure_tag_attribute(
        self,
        domain: AttributeDomainItems,
    ):
        """
        Ensures that the geometry tag attribute exists on the underlying data.
        Falls back to custom internal storage if native attributes are unsupported (e.g., bpy.types.Curve).
        """
        data = self.obj.data
        if data is None:
            return None

        attr_name = {
            "FACE": self.FACE_TAG_ATTR,
            "EDGE": self.EDGE_TAG_ATTR,
            "POINT": self.VERT_TAG_ATTR,
            "CURVE": self.CURVE_TAG_ATTR,
        }[domain]

        # 1. Try native Blender attributes API
        if hasattr(data, "attributes"):
            try:
                attr = data.attributes.get(attr_name)
                if attr is None:
                    attr = data.attributes.new(
                        name=attr_name,
                        type="STRING",
                        domain=domain,
                    )
                if attr is not None:
                    return attr
            except (RuntimeError, TypeError):
                pass

        # 2. Fallback to custom storage if attributes API is unavailable on data type
        if attr_name not in self._custom_attributes:
            self._custom_attributes[attr_name] = CustomAttribute(domain)
        attr = self._custom_attributes[attr_name]
        attr.ensure_size(len(self._domain_indices(domain)))
        return attr

    @abstractmethod
    def _domain_indices(
        self,
        domain: AttributeDomainItems,
    ) -> list[int]:
        """
        Returns all indices for the specified geometry domain.
        """
        raise NotImplementedError

    def _get_tags(
        self,
        domain: AttributeDomainItems,
        indices: list[int],
    ) -> list[str]:
        """
        Returns all unique tags found on the specified geometry elements.
        """
        if not indices:
            return []

        attr = self._ensure_tag_attribute(domain)
        if attr is None:
            return []

        result = []
        seen = set()

        for index in indices:
            raw_value = attr.data[index].value
            value: str = (
                raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
            )
            if not value:
                continue

            for tag in value.split(self.TAG_SEPARATOR):
                tag = tag.strip()
                if not tag:
                    continue

                if tag not in seen:
                    seen.add(tag)
                    result.append(tag)

        return result

    def _set_tags(
        self,
        domain: AttributeDomainItems,
        indices: list[int],
        tags: Iterable[str],
    ):
        """
        Replaces tags on geometry elements.
        """
        if not indices:
            return

        attr = self._ensure_tag_attribute(domain)
        if attr is None:
            return

        unique_tags = []
        seen = set()

        for tag in tags:
            tag = str(tag).strip()
            if not tag:
                continue

            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        value = self.TAG_SEPARATOR.join(unique_tags)
        bytes_value = value.encode("utf-8")
        for index in indices:
            attr.data[index].value = bytes_value

        if hasattr(self.obj.data, "update"):
            self.obj.data.update()

    def _add_tags(
        self,
        domain: AttributeDomainItems,
        indices: list[int],
        tags: Iterable[str],
    ):
        """
        Adds tags to geometry elements. Existing tags are preserved.
        """
        if not indices:
            return

        tags_to_add = {str(tag).strip() for tag in tags if str(tag).strip()}

        if not tags_to_add:
            return

        attr = self._ensure_tag_attribute(domain)
        if attr is None:
            return

        for index in indices:
            raw_value = attr.data[index].value
            value: str = (
                raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
            )
            current_tags = {t for t in value.split(self.TAG_SEPARATOR) if t}
            current_tags.update(tags_to_add)

            attr.data[index].value = self.TAG_SEPARATOR.join(
                sorted(current_tags)
            ).encode("utf-8")
        if hasattr(self.obj.data, "update"):
            self.obj.data.update()

    def _remove_tags(
        self,
        domain: AttributeDomainItems,
        indices: list[int],
        tags: Iterable[str],
    ):
        """
        Removes tags from geometry elements.
        """
        if not indices:
            return

        attr = self._ensure_tag_attribute(domain)
        if attr is None:
            return

        tags_to_remove = {str(tag).strip() for tag in tags if str(tag).strip()}

        for index in indices:
            raw_value = attr.data[index].value
            value: str = (
                raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
            )
            current_tags = [
                tag
                for tag in value.split(self.TAG_SEPARATOR)
                if tag and tag not in tags_to_remove
            ]

            attr.data[index].value = self.TAG_SEPARATOR.join(current_tags).encode(
                "utf-8"
            )

        if hasattr(self.obj.data, "update"):
            self.obj.data.update()

    def get_tags(
        self,
        domain: Optional[AttributeDomainItems] = None,
    ) -> list[str]:
        """
        Returns all unique tags used in the specified domain (or all valid domains if None).
        """
        if domain is None:
            if isinstance(self.obj.data, bpy.types.Curve):
                return self.get_tags("CURVE") + self.get_tags("POINT")
            return (
                self.get_tags("FACE") + self.get_tags("EDGE") + self.get_tags("POINT")
            )
        return self._get_tags(
            domain,
            self._domain_indices(domain),
        )

    def set_tags(
        self,
        tags: Iterable[str],
        domain: Optional[AttributeDomainItems] = None,
    ):
        """
        Replaces tags on all geometry of the specified domain (or default domains if None).
        """
        if domain is None:
            if isinstance(self.obj.data, bpy.types.Curve):
                self.set_tags(tags, "CURVE")
                self.set_tags(tags, "POINT")
            else:
                self.set_tags(tags, "FACE")
                self.set_tags(tags, "EDGE")
                self.set_tags(tags, "POINT")
            return
        self._set_tags(
            domain,
            self._domain_indices(domain),
            tags,
        )

    def add_tags(
        self,
        tags: Iterable[str],
        domain: Optional[AttributeDomainItems] = None,
    ):
        """
        Adds tags to all geometry of the specified domain (or default domains if None).
        """
        if domain is None:
            if isinstance(self.obj.data, bpy.types.Curve):
                self.add_tags(tags, "CURVE")
                self.add_tags(tags, "POINT")
            else:
                self.add_tags(tags, "FACE")
                self.add_tags(tags, "EDGE")
                self.add_tags(tags, "POINT")
            return
        self._add_tags(
            domain,
            self._domain_indices(domain),
            tags,
        )

    def remove_tags(
        self,
        tags: Iterable[str],
        domain: Optional[AttributeDomainItems] = None,
    ):
        """
        Removes tags from all geometry of the specified domain (or default domains if None).
        """
        if domain is None:
            if isinstance(self.obj.data, bpy.types.Curve):
                self.remove_tags(tags, "CURVE")
                self.remove_tags(tags, "POINT")
            else:
                self.remove_tags(tags, "FACE")
                self.remove_tags(tags, "EDGE")
                self.remove_tags(tags, "POINT")
            return
        self._remove_tags(
            domain,
            self._domain_indices(domain),
            tags,
        )
