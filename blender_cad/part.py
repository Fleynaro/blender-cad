import hashlib
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

import bmesh
import bpy
from mathutils import Matrix, Vector
from typing_extensions import override

from .bmesh_wrapper import BMEdgeWrapper, BMeshWrapper, BMFaceWrapper, BMVertWrapper
from .common import VectorLike, extract_vector
from .location import Location, Pos, Scale, Transform, TransformExpr
from .material import bpy_material_hash, mat
from .object import AttributeDomainItems, Object
from .shape_list import ShapeList

if TYPE_CHECKING:
    from .geometry import (
        Edge,
        Face,
        GeometryCheckpoint,
        Topology,
        TopologyConfig,
        Vertex,
        Wire,
    )


_LEGACY_TAG_LABEL_OVERLAY_REGISTRY_KEY = "_blender_cad_tag_label_overlays_v1"
_legacy_overlay_registry = bpy.app.driver_namespace.pop(
    _LEGACY_TAG_LABEL_OVERLAY_REGISTRY_KEY, None
)
if _legacy_overlay_registry is not None:
    for _area, _handler in tuple(
        _legacy_overlay_registry.get("handlers", {}).values()
    ):
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handler, "WINDOW")
        except (ReferenceError, RuntimeError, ValueError):
            pass


@dataclass(frozen=True)
class TagDisplayConfig:
    """Controls mesh-tag plaque domains and longitudinal grouping."""

    faces: bool = True
    edges: bool = True
    vertices: bool = False
    system_tags: bool = False
    group_adjacent: bool = True


class Part(Object):
    """An object representing a part. Manages its own mesh and Blender object."""

    _TAG_LABEL_MARKER = "_blender_cad_tag_label"
    _TAG_LABEL_MATERIAL = "BlenderCAD_TagLabels"
    _SYSTEM_TAG_LABEL_MATERIAL = "BlenderCAD_SystemTagLabels"
    _TAG_LABEL_PLATE_MATERIAL = "BlenderCAD_TagLabelPlate"
    _TAG_ELEMENT_TYPE_MATERIAL = "BlenderCAD_TagElementType"
    _USER_TAG_LABEL_MATERIAL_PREFIX = "BlenderCAD_UserTag_"
    _TAG_LABEL_RELATIONSHIP_STATE_PREFIX = "_blender_cad_tag_label_relationships_"
    _USER_TAG_PALETTE = (
        (1.0, 0.28, 0.22, 1.0),  # coral
        (1.0, 0.68, 0.05, 1.0),  # amber
        (0.70, 1.0, 0.16, 1.0),  # lime
        (0.05, 0.90, 0.35, 1.0),  # emerald
        (0.15, 0.72, 1.0, 1.0),  # sky blue
        (0.68, 0.42, 1.0, 1.0),  # violet
        (1.0, 0.25, 0.68, 1.0),  # pink
        (1.0, 0.48, 0.08, 1.0),  # orange
        (0.40, 1.0, 0.75, 1.0),  # mint
        (1.0, 0.95, 0.78, 1.0),  # warm white
    )
    _SYSTEM_TAGS = frozenset(
        (
            Object.TAG_OP_EXTRUDE,
            Object.TAG_ML_BACKGROUND,
            Object.TAG_ML_BORDER,
        )
    )

    def __init__(
        self,
        obj: Optional[bpy.types.Object] = None,
        topology: Optional["TopologyConfig"] = None,
    ):
        super().__init__(obj)
        from .geometry import TopologyConfig

        self._bm_wrapper: Optional["BMeshWrapper"] = None
        self.topology_config = topology or TopologyConfig()
        self._topology: Optional["Topology"] = None
        self._last_checkpoint: Optional["GeometryCheckpoint"] = None
        self._last_op_checkpoint: Optional["GeometryCheckpoint"] = None

    @override
    def _create_empty_object(self):
        mesh = bpy.data.meshes.new("PartMesh")
        obj = bpy.data.objects.new("Part", mesh)
        return obj

    @override
    def copy(self) -> "Part":
        """Creates a copy of the Part and its underlying Blender object."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        new_obj = self.obj.copy()
        new_obj.data = self.obj.data.copy()
        part = Part(new_obj)
        part._bm_wrapper = self._bm_wrapper
        part._topology = self._topology
        self._after_copy(part)
        return part

    def show(
        self,
        name: str | None = None,
        hide=False,
        collection_name: str | None = None,
        show_tags: bool | TagDisplayConfig = False,
    ):
        """Display this part and optionally label its mesh tags.

        ``show_tags=True`` shows user face and edge tags. Pass a
        ``TagDisplayConfig`` to include vertex or system tags, choose domains,
        or disable grouping of adjacent tagged elements.
        """
        if isinstance(show_tags, bool):
            tag_config = TagDisplayConfig() if show_tags else None
        elif isinstance(show_tags, TagDisplayConfig):
            tag_config = show_tags
        else:
            raise TypeError("show_tags must be a bool or TagDisplayConfig")

        show_name = (
            None
            if name is not None
            and self.is_physically_valid
            and self.obj.name == name
            else name
        )
        super().show(name=show_name, hide=hide, collection_name=collection_name)
        self._clear_tag_labels()
        if tag_config is not None:
            collection = (
                bpy.data.collections.get(collection_name)
                if collection_name is not None
                else bpy.context.collection
            )
            if collection is None:
                collection = (
                    self.obj.users_collection[0]
                    if self.obj.users_collection
                    else bpy.context.scene.collection
                )
            self._show_tag_labels(
                collection,
                hide=hide,
                config=tag_config,
            )

    def _read_tag_attribute(
        self,
        attribute_name: str,
        domain: str,
        element_count: int,
    ) -> list[tuple[int, tuple[str, ...]]]:
        """Reads the stored tag strings without creating empty attributes."""
        data = self.obj.data
        attributes = getattr(data, "attributes", None)
        attribute = attributes.get(attribute_name) if attributes is not None else None

        if attribute is not None:
            if attribute.domain != domain or attribute.data_type != "STRING":
                return []
            values = attribute.data
        else:
            attribute = self._custom_attributes.get(attribute_name)
            if attribute is None or attribute.domain != domain:
                return []
            values = attribute.data

        records = []
        for index in range(min(element_count, len(values))):
            raw_value = values[index].value
            if isinstance(raw_value, bytes):
                raw_value = raw_value.decode("utf-8")
            tags = tuple(
                tag.strip()
                for tag in str(raw_value or "").split(self.TAG_SEPARATOR)
                if tag.strip()
            )
            if tags:
                records.append((index, tags))
        return records

    @staticmethod
    def _tag_edge_normals(mesh: bpy.types.Mesh) -> list[Vector]:
        """Builds an averaged adjacent-face normal for each mesh edge."""
        edge_lookup = {
            tuple(sorted(edge.vertices)): index for index, edge in enumerate(mesh.edges)
        }
        normals = [Vector((0.0, 0.0, 0.0)) for _ in mesh.edges]
        for polygon in mesh.polygons:
            vertices = polygon.vertices
            for index, vertex_a in enumerate(vertices):
                vertex_b = vertices[(index + 1) % len(vertices)]
                edge_index = edge_lookup.get(tuple(sorted((vertex_a, vertex_b))))
                if edge_index is not None:
                    normals[edge_index] += polygon.normal
        return normals

    def _clear_tag_labels(self):
        """Removes the text and backing plates created for mesh tags."""
        if not self.is_physically_valid:
            return

        for child in tuple(self.obj.children):
            if not child.get(self._TAG_LABEL_MARKER, False):
                continue
            label_data = child.data
            bpy.data.objects.remove(child, do_unlink=True)
            if label_data is None or label_data.users != 0:
                continue
            if isinstance(label_data, bpy.types.Curve):
                bpy.data.curves.remove(label_data)
            elif isinstance(label_data, bpy.types.Mesh):
                bpy.data.meshes.remove(label_data)

        if not any(obj.get(self._TAG_LABEL_MARKER, False) for obj in bpy.data.objects):
            self._restore_tag_label_relationship_lines()

        for label_material in tuple(bpy.data.materials):
            is_tag_material = label_material.name in (
                self._TAG_LABEL_MATERIAL,
                self._SYSTEM_TAG_LABEL_MATERIAL,
                self._TAG_LABEL_PLATE_MATERIAL,
                self._TAG_ELEMENT_TYPE_MATERIAL,
            ) or label_material.name.startswith(self._USER_TAG_LABEL_MATERIAL_PREFIX)
            if is_tag_material and label_material.users == 0:
                bpy.data.materials.remove(label_material)

    def _hide_tag_label_relationship_lines(self):
        """Hides Blender's parent rays while tag labels are visible."""
        scene = bpy.context.scene
        if scene is None:
            return

        for screen in bpy.data.screens:
            view_areas = [area for area in screen.areas if area.type == "VIEW_3D"]
            if not view_areas:
                continue

            state_key = f"{self._TAG_LABEL_RELATIONSHIP_STATE_PREFIX}{screen.name}"
            if state_key not in scene:
                scene[state_key] = "".join(
                    "1" if area.spaces.active.overlay.show_relationship_lines else "0"
                    for area in view_areas
                )
            for area in view_areas:
                area.spaces.active.overlay.show_relationship_lines = False

    def _restore_tag_label_relationship_lines(self):
        """Restores relationship-line visibility saved before adding labels."""
        scene = bpy.context.scene
        if scene is None:
            return

        for screen in bpy.data.screens:
            state_key = f"{self._TAG_LABEL_RELATIONSHIP_STATE_PREFIX}{screen.name}"
            if state_key not in scene:
                continue

            states = str(scene[state_key])
            view_areas = [area for area in screen.areas if area.type == "VIEW_3D"]
            for area, state in zip(view_areas, states):
                area.spaces.active.overlay.show_relationship_lines = state == "1"
            del scene[state_key]

    def _show_tag_labels(
        self,
        collection: bpy.types.Collection,
        hide=False,
        config: TagDisplayConfig = TagDisplayConfig(),
    ):
        """Creates spaced tag plaques and leader lines for tagged mesh elements."""
        mesh = self.obj.data
        if not isinstance(mesh, bpy.types.Mesh):
            return

        records = {
            "FACE": self._read_tag_attribute(
                self.FACE_TAG_ATTR, "FACE", len(mesh.polygons)
            ),
            "EDGE": self._read_tag_attribute(
                self.EDGE_TAG_ATTR, "EDGE", len(mesh.edges)
            ),
            "POINT": self._read_tag_attribute(
                self.VERT_TAG_ATTR, "POINT", len(mesh.vertices)
            ),
        }
        if not config.faces:
            records["FACE"] = []
        if not config.edges:
            records["EDGE"] = []
        if not config.vertices:
            records["POINT"] = []
        if not config.system_tags:
            records = {
                domain: [
                    (index, tuple(tag for tag in tags if tag not in self._SYSTEM_TAGS))
                    for index, tags in domain_records
                    if any(tag not in self._SYSTEM_TAGS for tag in tags)
                ]
                for domain, domain_records in records.items()
            }
        if not any(records.values()):
            return

        world_matrix = self.obj.matrix_world.copy()
        world_bbox = [world_matrix @ Vector(corner) for corner in self.obj.bound_box]
        bbox_min = Vector(
            tuple(min(corner[axis] for corner in world_bbox) for axis in range(3))
        )
        bbox_max = Vector(
            tuple(max(corner[axis] for corner in world_bbox) for axis in range(3))
        )
        model_size = max((float(value) for value in self.obj.dimensions), default=0.0)
        font_size = max(model_size * 0.016, 0.005)
        local_x_values = [float(vertex.co.x) for vertex in mesh.vertices]
        local_x_min = min(local_x_values, default=0.0)
        local_x_span = max(local_x_values, default=0.0) - local_x_min
        tag_group_x_span = max(local_x_span * 0.25, font_size * 6.0)

        def element_local_x(domain: str, index: int) -> float:
            if domain == "FACE":
                return float(mesh.polygons[index].center.x)
            if domain == "EDGE":
                edge = mesh.edges[index]
                return float(
                    (
                        mesh.vertices[edge.vertices[0]].co.x
                        + mesh.vertices[edge.vertices[1]].co.x
                    )
                    * 0.5
                )
            return float(mesh.vertices[index].co.x)

        def x_region(index: int, domain: str) -> int:
            region_index = int(
                (element_local_x(domain, index) - local_x_min) // tag_group_x_span
            )
            max_region_index = max(
                int(max(local_x_span - 1e-9, 0.0) / tag_group_x_span), 0
            )
            return min(region_index, max_region_index)

        def grouped_tag_records(domain: str, domain_records):
            if not config.group_adjacent:
                return [
                    (index, tags, (index,)) for index, tags in domain_records
                ]

            element_count = {
                "FACE": len(mesh.polygons),
                "EDGE": len(mesh.edges),
                "POINT": len(mesh.vertices),
            }[domain]
            adjacency = [set() for _ in range(element_count)]
            if domain == "FACE":
                edge_lookup = {
                    tuple(sorted(edge.vertices)): index
                    for index, edge in enumerate(mesh.edges)
                }
                faces_by_edge = {}
                for polygon in mesh.polygons:
                    vertices = polygon.vertices
                    for position, vertex_a in enumerate(vertices):
                        vertex_b = vertices[(position + 1) % len(vertices)]
                        edge_index = edge_lookup.get(
                            tuple(sorted((vertex_a, vertex_b)))
                        )
                        if edge_index is not None:
                            faces_by_edge.setdefault(edge_index, []).append(
                                polygon.index
                            )
                for face_indices in faces_by_edge.values():
                    for face_index in face_indices:
                        adjacency[face_index].update(
                            other for other in face_indices if other != face_index
                        )
            elif domain == "EDGE":
                edges_by_vertex = {}
                for edge_index, edge in enumerate(mesh.edges):
                    for vertex_index in edge.vertices:
                        edges_by_vertex.setdefault(vertex_index, []).append(edge_index)
                for edge_indices in edges_by_vertex.values():
                    for edge_index in edge_indices:
                        adjacency[edge_index].update(
                            other for other in edge_indices if other != edge_index
                        )
            else:
                for edge in mesh.edges:
                    vertex_a, vertex_b = edge.vertices
                    adjacency[vertex_a].add(vertex_b)
                    adjacency[vertex_b].add(vertex_a)

            tags_by_index = dict(domain_records)
            remaining = set(tags_by_index)
            grouped = []
            while remaining:
                seed = min(remaining)
                seed_tags = tags_by_index[seed]
                seed_key = tuple(sorted(seed_tags))
                component = []
                pending = [seed]
                remaining.remove(seed)
                while pending:
                    index = pending.pop()
                    component.append(index)
                    for neighbor in adjacency[index]:
                        if neighbor not in remaining:
                            continue
                        if tuple(sorted(tags_by_index[neighbor])) != seed_key:
                            continue
                        if x_region(neighbor, domain) != x_region(seed, domain):
                            continue
                        remaining.remove(neighbor)
                        pending.append(neighbor)
                grouped.append((min(component), seed_tags, tuple(sorted(component))))
            return grouped

        grouped_records = {
            domain: grouped_tag_records(domain, domain_records)
            for domain, domain_records in records.items()
        }

        user_tags = sorted(
            {
                tag
                for domain_records in grouped_records.values()
                for _, tags, _indices in domain_records
                for tag in tags
                if tag not in self._SYSTEM_TAGS
            }
        )
        user_tag_hashes = {
            tag: hashlib.sha256(tag.encode("utf-8")).digest() for tag in user_tags
        }
        user_tag_colors = {
            tag: self._USER_TAG_PALETTE[
                int.from_bytes(user_tag_hashes[tag][:4], "big")
                % len(self._USER_TAG_PALETTE)
            ]
            for tag in user_tags
        }

        edge_normals = self._tag_edge_normals(mesh) if records["EDGE"] else []
        normal_matrix = world_matrix.to_3x3().inverted_safe().transposed()
        normal_offset = max(font_size * 0.5, model_size * 0.001)
        label_up = Vector((0.0, 0.0, 1.0))
        # Local text X/Y/Z map to world Y/Z/X, so the plaque face is normal to +X.
        label_rotation = Matrix(
            (
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
            )
        ).transposed().to_quaternion()

        def get_label_material(name: str, color: tuple[float, float, float, float]):
            label_material = bpy.data.materials.get(name)
            if label_material is None:
                label_material = bpy.data.materials.new(name)
            label_material.diffuse_color = color
            if label_material.use_nodes and label_material.node_tree is not None:
                shader = label_material.node_tree.nodes.get("Principled BSDF")
                if shader is not None and "Base Color" in shader.inputs:
                    shader.inputs["Base Color"].default_value = color
            return label_material

        plate_material = get_label_material(
            self._TAG_LABEL_PLATE_MATERIAL, (0.012, 0.018, 0.03, 1.0)
        )
        system_color = (0.2, 0.95, 1.0, 1.0)
        type_color = (0.82, 0.88, 0.98, 1.0)
        system_material_name = self._SYSTEM_TAG_LABEL_MATERIAL
        type_material_name = self._TAG_ELEMENT_TYPE_MATERIAL

        def world_surface_normal(normal: Vector):
            world_normal = normal_matrix @ normal
            if world_normal.length_squared <= 1e-12:
                world_normal = normal_matrix @ Vector((0.0, 0.0, 1.0))
            if world_normal.length_squared <= 1e-12:
                world_normal = Vector((0.0, 0.0, 1.0))
            world_normal.normalize()
            return world_normal

        def link_label_object(
            label_obj: bpy.types.Object,
            label_matrix: Matrix,
        ):
            collection.objects.link(label_obj)
            label_obj.parent = self.obj
            label_obj.matrix_parent_inverse = world_matrix.inverted_safe()
            label_obj.matrix_world = label_matrix
            label_obj.hide_set(hide)
            label_obj[self._TAG_LABEL_MARKER] = True

        def create_leader_object(
            domain: str,
            index: int,
            anchor_position: Vector,
            position: Vector,
            plate_width: float,
            plate_height: float,
            side_sign: float,
            material_name: str,
            color: tuple[float, float, float, float],
        ):
            endpoint = Vector(
                (
                    position.x,
                    position.y - side_sign * plate_width / 2,
                    min(
                        max(anchor_position.z, position.z - plate_height / 2),
                        position.z + plate_height / 2,
                    ),
                )
            )
            inverse_world = world_matrix.inverted_safe()
            leader_data = bpy.data.curves.new(
                f"{self.obj.name}_Tag_{domain}_{index}_LeaderCurve", type="CURVE"
            )
            leader_data.dimensions = "3D"
            leader_data.resolution_u = 1
            leader_data.bevel_depth = max(font_size * 0.025, model_size * 0.0004, 0.002)
            leader_data.bevel_resolution = 0
            material = get_label_material(material_name, color)
            leader_data.materials.append(material)
            spline = leader_data.splines.new("POLY")
            spline.points.add(1)
            for point, world_position in zip(
                spline.points,
                (anchor_position, endpoint),
            ):
                local_position = inverse_world @ world_position
                point.co = (*local_position, 1.0)

            leader_obj = bpy.data.objects.new(
                f"{self.obj.name}_Tag_{domain}_{index}_Leader", leader_data
            )
            link_label_object(leader_obj, world_matrix.copy())
            leader_obj.show_in_front = True

        def create_plate_object(
            domain: str,
            index: int,
            lines: list[tuple[str, float, str, tuple]],
            anchor_position: Vector,
            position: Vector,
            side_sign: float,
            plate_width: float,
            plate_height: float,
            line_spacing: float,
            plate_depth: float,
            rotation,
        ):
            stripe_width = font_size * 0.12
            stripe_color = lines[1][3] if len(lines) > 1 else lines[0][3]
            vertices = [
                (-plate_width / 2, -plate_height / 2, -plate_depth / 2),
                (plate_width / 2, -plate_height / 2, -plate_depth / 2),
                (plate_width / 2, plate_height / 2, -plate_depth / 2),
                (-plate_width / 2, plate_height / 2, -plate_depth / 2),
                (-plate_width / 2, -plate_height / 2, plate_depth / 2),
                (plate_width / 2, -plate_height / 2, plate_depth / 2),
                (plate_width / 2, plate_height / 2, plate_depth / 2),
                (-plate_width / 2, plate_height / 2, plate_depth / 2),
            ]
            faces = [
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            ]
            stripe_left = -plate_width / 2 + font_size * 0.2
            stripe_right = stripe_left + stripe_width
            stripe_front_z = plate_depth / 2 + font_size * 0.01
            vertices.extend(
                (
                    (stripe_left, -plate_height / 2, stripe_front_z),
                    (stripe_right, -plate_height / 2, stripe_front_z),
                    (stripe_right, plate_height / 2, stripe_front_z),
                    (stripe_left, plate_height / 2, stripe_front_z),
                    (stripe_left, -plate_height / 2, -stripe_front_z),
                    (stripe_right, -plate_height / 2, -stripe_front_z),
                    (stripe_right, plate_height / 2, -stripe_front_z),
                    (stripe_left, plate_height / 2, -stripe_front_z),
                )
            )
            faces.extend(((8, 9, 10, 11), (12, 15, 14, 13)))

            plate_data = bpy.data.meshes.new(
                f"{self.obj.name}_Tag_{domain}_{index}_PlateMesh"
            )
            plate_data.from_pydata(vertices, [], faces)
            plate_data.materials.append(plate_material)
            stripe_hash = hashlib.sha256(
                str(stripe_color).encode("utf-8")
            ).hexdigest()[:12]
            stripe_material = get_label_material(
                f"{self._USER_TAG_LABEL_MATERIAL_PREFIX}stripe_{stripe_hash}",
                stripe_color,
            )
            plate_data.materials.append(stripe_material)
            plate_data.polygons[-2].material_index = 1
            plate_data.polygons[-1].material_index = 1
            plate_obj = bpy.data.objects.new(
                f"{self.obj.name}_Tag_{domain}_{index}_Plate", plate_data
            )
            plate_matrix = Matrix.LocRotScale(
                position, rotation, Vector((1.0, 1.0, 1.0))
            )
            link_label_object(plate_obj, plate_matrix)

            first_line_offset = (len(lines) - 1) * line_spacing / 2
            for line_index, (text, size, material_name, color) in enumerate(lines):
                line_offset = first_line_offset - line_index * line_spacing
                for text_side, z_offset, flip in (
                    ("Front", plate_depth / 2 + font_size * 0.04, False),
                    ("Back", -plate_depth / 2 - font_size * 0.04, True),
                ):
                    label_data = bpy.data.curves.new(
                        name=(
                            f"{self.obj.name}_Tag_{domain}_{index}_"
                            f"Line_{line_index}_{text_side}_Text"
                        ),
                        type="FONT",
                    )
                    label_data.body = text
                    label_data.size = size
                    label_data.align_x = "CENTER"
                    label_data.align_y = "CENTER"
                    material = get_label_material(material_name, color)
                    label_data.materials.append(material)
                    label_obj = bpy.data.objects.new(
                        f"{self.obj.name}_Tag_{domain}_{index}_Line_{line_index}_{text_side}",
                        label_data,
                    )
                    label_matrix = plate_matrix @ Matrix.Translation(
                        (0.0, line_offset, z_offset)
                    )
                    if flip:
                        label_matrix = label_matrix @ Matrix.Rotation(
                            3.141592653589793, 4, "Y"
                        )
                    link_label_object(label_obj, label_matrix)
            create_leader_object(
                domain,
                index,
                anchor_position,
                position,
                plate_width,
                plate_height,
                side_sign,
                lines[1][2],
                lines[1][3],
            )

        pending_plates = []
        def add_label(
            domain: str,
            index: int,
            tags: tuple[str, ...],
            center: Vector,
            normal: Vector,
        ):
            world_normal = world_surface_normal(normal)
            system_tags = tuple(tag for tag in tags if tag in self._SYSTEM_TAGS)
            custom_tags = tuple(tag for tag in tags if tag not in self._SYSTEM_TAGS)
            lines = [
                (
                    tag,
                    font_size,
                    f"{self._USER_TAG_LABEL_MATERIAL_PREFIX}"
                    f"{user_tag_hashes[tag].hex()[:12]}",
                    user_tag_colors[tag],
                )
                for tag in custom_tags
            ]
            if system_tags:
                lines.append(
                    (
                        ", ".join(system_tags),
                        font_size * 0.82,
                        system_material_name,
                        system_color,
                    )
                )
            if lines:
                element_type = {
                    "FACE": "Грань",
                    "EDGE": "Ребро",
                    "POINT": "Вершина",
                }[domain]
                lines.insert(
                    0,
                    (element_type, font_size * 0.72, type_material_name, type_color),
                )
                anchor_position = world_matrix @ center
                anchor_position += world_normal * normal_offset
                domain_offsets = {"FACE": 0.65, "EDGE": -0.65, "POINT": 0.0}
                anchor_position += label_up * font_size * domain_offsets[domain]
                line_spacing = font_size * 1.35
                plate_width = max(
                    max(font_size * 0.62 * len(text) for text, *_rest in lines)
                    + font_size * 1.05,
                    font_size * 1.8,
                )
                plate_height = max(
                    font_size * 1.7,
                    (len(lines) - 1) * line_spacing + font_size * 1.7,
                )
                pending_plates.append(
                    {
                        "domain": domain,
                        "index": index,
                        "lines": lines,
                        "anchor_position": anchor_position,
                        "side": None,
                        "line_spacing": line_spacing,
                        "plate_width": plate_width,
                        "plate_height": plate_height,
                        "plate_depth": font_size * 0.08,
                    }
                )

        for index, tags, group_indices in grouped_records["FACE"]:
            group_center = sum(
                (mesh.polygons[face_index].center for face_index in group_indices),
                Vector((0.0, 0.0, 0.0)),
            ) / len(group_indices)
            polygon = min(
                (mesh.polygons[face_index] for face_index in group_indices),
                key=lambda candidate: (candidate.center - group_center).length_squared,
            )
            add_label(
                "FACE", index, tags, polygon.center, polygon.normal
            )

        for index, tags, group_indices in grouped_records["EDGE"]:
            edge_centers = {
                edge_index: (
                    mesh.vertices[mesh.edges[edge_index].vertices[0]].co
                    + mesh.vertices[mesh.edges[edge_index].vertices[1]].co
                )
                * 0.5
                for edge_index in group_indices
            }
            group_center = sum(
                edge_centers.values(), Vector((0.0, 0.0, 0.0))
            ) / len(group_indices)
            index = min(
                group_indices,
                key=lambda edge_index: (
                    edge_centers[edge_index] - group_center
                ).length_squared,
            )
            edge = mesh.edges[index]
            vertex_a, vertex_b = edge.vertices
            point_a = mesh.vertices[vertex_a].co
            point_b = mesh.vertices[vertex_b].co
            center = (point_a + point_b) * 0.5
            normal = edge_normals[index]
            if normal.length_squared <= 1e-12:
                normal = (
                    mesh.vertices[vertex_a].normal + mesh.vertices[vertex_b].normal
                )
            add_label("EDGE", index, tags, center, normal)

        for index, tags, group_indices in grouped_records["POINT"]:
            group_center = sum(
                (mesh.vertices[vertex_index].co for vertex_index in group_indices),
                Vector((0.0, 0.0, 0.0)),
            ) / len(group_indices)
            index = min(
                group_indices,
                key=lambda vertex_index: (
                    mesh.vertices[vertex_index].co - group_center
                ).length_squared,
            )
            vertex = mesh.vertices[index]
            add_label(
                "POINT",
                index,
                tags,
                vertex.co,
                vertex.normal,
            )

        callout_offset = max(model_size * 0.14, font_size * 3.0, 0.25)
        label_gap = max(model_size * 0.02, font_size * 0.75, 0.12)
        side_loads = {-1.0: 0.0, 1.0: 0.0}
        for item in sorted(
            pending_plates,
            key=lambda value: (value["domain"], value["index"]),
        ):
            anchor = item["anchor_position"]
            distance_to_negative = anchor.y - (bbox_min.y - callout_offset)
            distance_to_positive = bbox_max.y + callout_offset - anchor.y
            if abs(distance_to_positive - distance_to_negative) <= label_gap:
                side = min(side_loads, key=side_loads.get)
            else:
                side = 1.0 if distance_to_positive < distance_to_negative else -1.0
            item["side"] = side
            side_loads[side] += item["plate_height"] + label_gap

        for side in (-1.0, 1.0):
            side_items = [
                item for item in pending_plates if item["side"] == side
            ]
            side_items.sort(
                key=lambda item: (
                    item["anchor_position"].z,
                    item["anchor_position"].x,
                    item["domain"],
                    item["index"],
                )
            )
            total_height = sum(item["plate_height"] for item in side_items)
            total_height += label_gap * max(len(side_items) - 1, 0)
            z_cursor = (bbox_min.z + bbox_max.z - total_height) / 2
            y_boundary = bbox_max.y if side > 0 else bbox_min.y
            for item in side_items:
                plate_height = item["plate_height"]
                position = Vector(
                    (
                        item["anchor_position"].x,
                        y_boundary
                        + side * (callout_offset + item["plate_width"] / 2),
                        z_cursor + plate_height / 2,
                    )
                )
                create_plate_object(
                    item["domain"],
                    item["index"],
                    item["lines"],
                    item["anchor_position"],
                    position,
                    side,
                    item["plate_width"],
                    plate_height,
                    item["line_spacing"],
                    item["plate_depth"],
                    label_rotation,
                )
                z_cursor += plate_height + label_gap

        self._hide_tag_label_relationship_lines()

    @override
    def remove(self, physical=True):
        """Safely removes the object and its data from the Blender scene."""
        if physical and self.is_physically_valid:
            self._clear_tag_labels()
            mesh_data = self.obj.data
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)
        super().remove(physical)

    def hash(self, precision=4, use_materials=False):
        """Generates a SHA256 hash based on geometry, materials, and world matrix."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        hash_m = hashlib.sha256()
        mesh = self.obj.data

        # 1. Get coordinates and ROUND them
        # Rounding mitigates float precision differences (e.g., 0.0000001)
        verts_coords = [0.0] * (len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", verts_coords)

        # Group by (x,y,z) and round each component
        rounded_verts = [
            tuple(round(v, precision) for v in mesh.vertices[i].co)
            for i in range(len(mesh.vertices))
        ]

        # SORT the vertex list. This is CRUCIAL:
        # It makes the hash independent of the vertex order in the mesh storage
        rounded_verts.sort()

        # Pack sorted and rounded coordinates
        for v in rounded_verts:
            hash_m.update(struct.pack("3f", *v))

        # 2. Face materials (if enabled)
        if use_materials:
            # 1. Pre-calculate and cache hashes for all material slots
            material_cache = []
            for slot in self.obj.material_slots:
                if slot.material:
                    material_cache.append(bpy_material_hash(slot.material))
                else:
                    material_cache.append("None")

            # 2. Iterate through polygons using the cache
            num_slots = len(material_cache)
            for p in mesh.polygons:
                slot_index = p.material_index

                # Retrieve from cache or fallback if index is out of bounds
                if 0 <= slot_index < num_slots:
                    mat_hash = material_cache[slot_index]
                else:
                    mat_hash = "None"

                # Update hash
                hash_m.update(mat_hash.encode("utf-8"))

        # 3. Transformation matrix
        matrix_flat = [
            round(val, precision) for row in self.transform.matrix for val in row
        ]
        hash_m.update(struct.pack("16f", *matrix_flat))

        return hash_m.hexdigest()

    @property
    def has_polygons(self):
        """Checks if the part has any polygons."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        return len(self.obj.data.polygons) > 0

    @property
    def mat(self):
        """Access the material of the part."""
        pass

    @mat.setter
    def mat(self, material: Optional["mat.Layer"]):
        """Sets the material for the selected faces (or all faces if none selected)."""
        self._set_material(material)

    @property
    def default_mat(self):
        """Access the default material (index 0) of the part."""
        pass

    @default_mat.setter
    def default_mat(self, material: Optional["mat.Layer"]):
        """
        Sets the default material at index 0.
        If specific face materials are removed (set to None), this material will be used.
        """
        self._set_material(material, replace=False, default=True)

    def _set_material(
        self,
        material: Optional["mat.Layer"],
        faces: list["BMFaceWrapper"] | None = None,
        replace: bool = True,
        default: bool = False,
    ):
        self._ensure_bmesh(write=True)
        faces, all_faces = self._get_actual_bmesh_faces(faces)
        assert not default or len(faces) == len(all_faces)
        idx = self._get_or_create_material_index(material, default)
        for f in faces:
            if not replace and f.native.material_index > 0:
                continue
            f.native.material_index = idx
        self._write_bmesh()

    def _fix_topology(self, remove_double_verts: bool = True, min_vert_dist=1e-4):
        """Fixes the topology for the further correct topology analysis."""
        bm = self._ensure_bmesh(write=True)
        if remove_double_verts:
            bmesh.ops.remove_doubles(
                bm.native, verts=bm.native.verts, dist=min_vert_dist
            )
        loose_verts = [v for v in bm.native.verts if not v.link_edges]
        if loose_verts:
            bmesh.ops.delete(bm.native, geom=loose_verts, context="VERTS")
        bm.native.verts.index_update()
        self._write_bmesh()

    def _ensure_bmesh(self, write=False):
        """Creates or returns a BMesh wrapper."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        if write:
            self._flush_bmesh()
        if self._bm_wrapper is None:
            bm = bmesh.new()
            bm.from_mesh(self.obj.data)
            # Ensure index access via lookup tables
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            bm.edges.layers.float.new("bevel_weight_edge")
            self._bm_wrapper = BMeshWrapper(bm)
        return self._bm_wrapper

    def _get_actual_geometry(
        self, items: list, all_items: list, name: str, remove_duplicates: bool
    ) -> list:
        """Internal helper maintaining original mapping logic."""
        if not items:
            return all_items

        item_map = {i: i for i in all_items}
        try:
            actual_items = [item_map[i] for i in items]
            if remove_duplicates:
                actual_items = list(dict.fromkeys(actual_items))
            return actual_items
        except KeyError:
            raise RuntimeError(
                f"{name.capitalize()} are not valid for this object anymore. Recall {name}()."
            )

    def _get_actual_bmesh_faces(
        self, faces: list["BMEdgeWrapper"] | None = None, remove_duplicates=True
    ):
        bm = self._ensure_bmesh()
        actual: list["BMFaceWrapper"] = self._get_actual_geometry(
            faces, bm.faces, "faces", remove_duplicates
        )
        return actual, bm.faces

    def _get_actual_bmesh_edges(
        self, edges: list["BMEdgeWrapper"] | None = None, remove_duplicates=True
    ):
        bm = self._ensure_bmesh()
        actual: list["BMEdgeWrapper"] = self._get_actual_geometry(
            edges, bm.edges, "edges", remove_duplicates
        )
        return actual, bm.edges

    def _get_actual_bmesh_verts(
        self, verts: list["BMVertWrapper"] | None = None, remove_duplicates=True
    ):
        bm = self._ensure_bmesh()
        actual: list["BMVertWrapper"] = self._get_actual_geometry(
            verts, bm.verts, "vertices", remove_duplicates
        )
        return actual, bm.verts

    def _write_bmesh(self, flush=False):
        """Writes the current BMesh data back to the Blender mesh."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        if self._bm_wrapper is None:
            raise RuntimeError("BMesh is not created")
        self._bm_wrapper.native.to_mesh(self.obj.data)
        self.obj.data.update()
        if flush:
            self._flush_bmesh()

    def _flush_bmesh(self):
        """Bakes the BMesh into the object and clears the current wrapper."""
        if self.obj is None:
            raise RuntimeError("Object is removed")
        if self._bm_wrapper is not None:
            self._bm_wrapper = None
            self._topology = None
            self._last_checkpoint = None

    def make_checkpoint(self) -> "GeometryCheckpoint":
        """
        Fixes the current state as the last checkpoint.
        If it already exists, returns the current one.
        """
        if self._last_checkpoint is None:
            from .geometry import GeometryCheckpoint

            bm = self._ensure_bmesh()
            self._last_checkpoint = GeometryCheckpoint(bm)
        return self._last_checkpoint

    def _make_op_checkpoint(self):
        self._last_op_checkpoint = self.make_checkpoint()
        return self._last_op_checkpoint

    def _add_vertices(
        self,
        coords: list[Vector],
        write: bool = False,
    ) -> list[bmesh.types.BMVert]:
        """Adds a list of coordinates as loose vertices to the BMesh."""
        bm = self._ensure_bmesh(write)
        vertices: list[bmesh.types.BMVert] = []

        for co in coords:
            v = bm.native.verts.new(co)
            vertices.append(v)

        if write:
            self._write_bmesh()

        return vertices

    def get_vertices(self) -> list[Vector]:
        return [v.co for v in self._ensure_bmesh().verts]

    def _ensure_joint_layers(self):
        """Creates or reuses custom vertex layers used by deformable joints."""
        bm = self._ensure_bmesh()

        joint_id_layer = bm.native.verts.layers.int.get("bp_joint_id")
        if joint_id_layer is None:
            joint_id_layer = bm.native.verts.layers.int.new("bp_joint_id")

        joint_role_layer = bm.native.verts.layers.int.get("bp_joint_role")
        if joint_role_layer is None:
            joint_role_layer = bm.native.verts.layers.int.new("bp_joint_role")

        return joint_id_layer, joint_role_layer

    def _inject_joint_markers(self, write=False) -> list[bmesh.types.BMVert]:
        """Adds temporary loose vertices for all deformable joints."""
        if not self._deformable_joints:
            return []
        from .joint import JointRole

        bm = self._ensure_bmesh(write)
        joint_id_layer, joint_role_layer = self._ensure_joint_layers()

        markers: list[bmesh.types.BMVert] = []
        eps = 0.001

        for joint in self._deformable_joints:
            # Use local-space coordinates, not world-space coordinates.
            loc = joint._rel_loc.matrix
            o = loc.to_translation()
            x = (
                loc.col[0].to_3d().normalized()
                if loc.col[0].to_3d().length > 1e-9
                else Vector((1, 0, 0))
            ) * eps
            y = (
                loc.col[1].to_3d().normalized()
                if loc.col[1].to_3d().length > 1e-9
                else Vector((0, 1, 0))
            ) * eps

            # Three points encode an origin plus two independent axes. Blender
            # deforms them with the mesh, allowing sync to rebuild the frame.
            payload = [
                (JointRole.ORIGIN, o),
                (JointRole.X_AXIS, o + x),
                (JointRole.Y_AXIS, o + y),
            ]

            for role, co in payload:
                v = bm.native.verts.new(co)
                v[joint_id_layer] = joint._joint_id
                v[joint_role_layer] = int(role)
                markers.append(v)

        bm.native.verts.ensure_lookup_table()
        if write:
            self._write_bmesh()
        return markers

    def _sync_joint_markers(self, write=False):
        """Reads temporary marker vertices back and updates attached joints."""
        if not self._deformable_joints:
            return
        from .joint import JointRole

        bm = self._ensure_bmesh(write)
        joint_id_layer, joint_role_layer = self._ensure_joint_layers()

        frames = {}
        to_delete: list[bmesh.types.BMVert] = []

        for v in bm.native.verts:
            joint_id = int(v[joint_id_layer])
            role = int(v[joint_role_layer])
            # Skip regular mesh vertices.
            if joint_id <= 0:
                continue

            frames.setdefault(joint_id, {})[role] = v.co.copy()
            to_delete.append(v)

        for joint_id, data in frames.items():
            joint = self._get_joint_by_id(joint_id)
            if joint is None:
                continue

            if (
                JointRole.ORIGIN in data
                and JointRole.X_AXIS in data
                and JointRole.Y_AXIS in data
            ):
                # Joint frame reconstruction also rejects collapsed marker axes
                # so a severe deformation cannot leave a silently invalid port.
                joint.sync_from_frame(
                    data[JointRole.ORIGIN],
                    data[JointRole.X_AXIS],
                    data[JointRole.Y_AXIS],
                )

        # remove markers
        bmesh.ops.delete(bm.native, geom=to_delete, context="VERTS")

        if write:
            self._write_bmesh()

    @override
    def _domain_indices(
        self,
        domain: AttributeDomainItems,
    ) -> list[int]:
        """
        Returns all indices for the specified geometry domain.
        """
        data = self.obj.data
        assert isinstance(data, bpy.types.Mesh)
        if domain == "FACE":
            return list(range(len(data.polygons)))
        if domain == "EDGE":
            return list(range(len(data.edges)))
        if domain == "POINT":
            return list(range(len(data.vertices)))
        return []

    def _get_topology(self):
        """Returns a cached or new topology graph."""
        from .geometry import Topology

        if self._topology is None or self.topology_config != self._topology.config:
            self._topology = Topology(self._ensure_bmesh(), self, self.topology_config)
        return self._topology

    # Selectors return specialized wrappers
    def faces(self) -> ShapeList["Face"]:
        """Returns a ShapeList of Face objects."""
        return ShapeList(self._get_topology().faces)

    def wires(self) -> ShapeList["Wire"]:
        """Returns a ShapeList of Wire objects."""
        return ShapeList(self._get_topology().wires)

    def edges(self) -> ShapeList["Edge"]:
        """Returns a ShapeList of Edge objects."""
        return ShapeList(self._get_topology().edges)

    def vertices(self) -> ShapeList["Vertex"]:
        """Returns a ShapeList of Vertex objects."""
        return ShapeList(self._get_topology().vertices)

    @staticmethod
    def from_any_mesh(mesh: bpy.types.Mesh, name: str = "unknown"):
        """Creates a Part from a Blender mesh (including evaluated meshes)."""
        physical_mesh = bpy.data.meshes.new(f"{name}_mesh")
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.to_mesh(physical_mesh)
        bm.free()
        return Part(obj=bpy.data.objects.new(name, physical_mesh))

    @staticmethod
    def from_any_object(obj: bpy.types.Object):
        """Creates a Part from a Blender object (including evaluated objects)."""
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(dg)
        return Part.from_any_mesh(eval_obj.to_mesh(), obj.name)

    @staticmethod
    def box_set_empty(topology: Optional["TopologyConfig"] = None):
        return BoxSetPart(topology=topology)


T = TypeVar("T", bound="Any")


class BoxSetPart(Part):
    """
    An optimized Part subclass representing a set of box primitives (primitive LOD).
    Does not hold a Blender mesh by default; maintains a list of BoxData items.
    The Blender mesh is constructed lazily upon accessing `self.obj`.
    """

    @dataclass
    class Box(Generic[T]):
        """Stores geometric metadata for a single box primitive."""

        transform: Transform
        size: Vector
        custom_data: Optional[T] = None

        @property
        def part(self):
            from .build_part import Mode
            from .primitives import Box

            return Box(
                self.size.x, self.size.y, self.size.z, mode=Mode.PRIVATE
            ).create_part()

    def __init__(
        self,
        boxes: Optional[list[Box]] = None,
        size: Optional[Vector] = None,
        topology: Optional["TopologyConfig"] = None,
        custom_data: Optional[T] = None,
        transform: Transform = Transform(),
    ):
        self.boxes: list[BoxSetPart.Box] = boxes or []
        if size is not None:
            self.boxes.append(
                BoxSetPart.Box(
                    transform=Transform(), size=Vector(size), custom_data=custom_data
                )
            )
        self._transform: Transform = transform

        self._obj_cache: Optional[bpy.types.Object] = None
        self._vertices_cache: Optional[list[Vector]] = None
        self._bbox_cache: Optional[Part.BBox] = None
        super().__init__(obj=None, topology=topology)

    @property
    @override
    def transform(self) -> Transform:
        """Access stored world transformation without touching self.obj."""
        return self._transform

    @transform.setter
    @override
    def transform(self, value: "TransformExpr"):
        """Sets transformation matrix directly in Python state."""
        self._transform = value.resolve(self)
        if self._obj_cache is not None:
            self._obj_cache.matrix_world = self._transform.matrix

    @property
    @override
    def scale(self) -> Vector:
        """Access scale without touching self.obj."""
        return self.transform.scale

    @scale.setter
    @override
    def scale(self, value: VectorLike):
        """Sets scale without touching self.obj."""
        self.transform = self.loc * Scale(value)

    @property
    @override
    def size(self) -> Vector:
        """
        Calculates total dimensions from local bounding box scaled by current object scale.
        """
        bbox = self.local_bbox
        local_dims = bbox.max - bbox.min
        return Vector(
            (
                local_dims.x * self.scale.x,
                local_dims.y * self.scale.y,
                local_dims.z * self.scale.z,
            )
        )

    @size.setter
    @override
    def size(self, value: VectorLike):
        """Adjusts scale based on desired overall dimensions."""
        target_size = extract_vector(value)
        orig = self.orig_size
        self.scale = Vector(
            (
                target_size.x / orig.x if orig.x > 1e-9 else 1.0,
                target_size.y / orig.y if orig.y > 1e-9 else 1.0,
                target_size.z / orig.z if orig.z > 1e-9 else 1.0,
            )
        )

    @override
    def _create_empty_object(self) -> Optional[bpy.types.Object]:
        # Defer creation of the Blender object until explicitly needed
        return None

    @property
    @override
    def is_valid(self) -> bool:
        return True

    @property
    @override
    def is_physically_valid(self) -> bool:
        if self._obj_cache is None:
            return True  # Valid logically even if mesh has not been built
        try:
            _ = self._obj_cache.name
            return True
        except ReferenceError:
            return False

    @property
    @override
    def obj(self) -> Optional[bpy.types.Object]:
        """
        Lazily generates and returns the Blender object when requested by client code.
        """
        if self._obj_cache is None:
            self._rebuild_mesh()
        return self._obj_cache

    @obj.setter
    def obj(self, value: Optional[bpy.types.Object]):
        self.invalidate_mesh_cache()
        self._obj_cache = value

    # Unit cube corner vectors centered at the origin [-0.5, 0.5]
    _UNIT_CORNERS = tuple(
        Vector((x, y, z)) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)
    )

    @override
    def get_vertices(self) -> list[Vector]:
        """
        Calculates and caches all 8 local corner vertices for every box in the set.
        """
        if self._vertices_cache is not None:
            return self._vertices_cache

        vertices = []
        for box in self.boxes:
            # Scale matrix for box dimensions combined with box local transform matrix
            box_scale_mat = Matrix.Diagonal((*box.size, 1.0))
            box_matrix = box.transform.matrix @ box_scale_mat

            # Transform unit cube corners to box local coordinates
            for corner in self._UNIT_CORNERS:
                vertices.append(box_matrix @ corner)

        self._vertices_cache = vertices
        return self._vertices_cache

    @override
    def _bbox_from_matrix(self, matrix: Matrix = Matrix.Identity(4)) -> Part.BBox:
        """
        Fast mathematical evaluation of the bounding box using cached box corner points.
        Bypasses dependency graph evaluation and self.obj entirely.
        """
        if self._bbox_cache is not None:
            return self._bbox_cache

        vertices = self.get_vertices()
        if not vertices:
            return self.BBox()

        # Transform cached local vertices by target matrix
        transformed_verts = [matrix @ v for v in vertices]

        min_v = Vector(
            (
                min(v.x for v in transformed_verts),
                min(v.y for v in transformed_verts),
                min(v.z for v in transformed_verts),
            )
        )
        max_v = Vector(
            (
                max(v.x for v in transformed_verts),
                max(v.y for v in transformed_verts),
                max(v.z for v in transformed_verts),
            )
        )

        bbox = self.BBox(min=min_v, max=max_v)
        self._bbox_cache = bbox
        return bbox

    def add_part(self, part: Part):
        """
        Adds another Part to this box set.
        If `part` is a BoxSetPart, its boxes are absorbed.
        Otherwise, the part is converted to a box via its bounding box.
        """
        if isinstance(part, BoxSetPart):
            for box in part.boxes:
                combined_transform = part.transform * box.transform
                self.boxes.append(
                    BoxSetPart.Box(
                        transform=combined_transform,
                        size=box.size.copy(),
                        custom_data=box.custom_data,
                    )
                )
        else:
            # Convert regular Part to a single box using local_bbox
            bbox = part.local_bbox
            box_size = bbox.max - bbox.min
            center = (bbox.max + bbox.min) / 2.0

            box_transform = part.transform * Pos(center)
            self.boxes.append(BoxSetPart.Box(transform=box_transform, size=box_size))

        self.invalidate_mesh_cache()

    def _rebuild_mesh(self):
        """Builds a combined BMesh from all stored box primitives."""
        if self._obj_cache is None:
            mesh = bpy.data.meshes.new("BoxSetMesh")
            self._obj_cache = bpy.data.objects.new("BoxSetPart", mesh)
        else:
            mesh: bpy.types.Mesh = self._obj_cache.data

        self._obj_cache.matrix_world = self.transform.matrix

        bm = bmesh.new()
        for box in self.boxes:
            # Apply box dimensions scale matrix combined with box world transform
            world_mat = (box.transform * Scale(box.size)).matrix
            bmesh.ops.create_cube(bm, size=1.0, matrix=world_mat)

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    def apply_transform(
        self,
        op: TransformExpr = Transform(),
        space: Location = Location(),
    ):
        """
        Applies transformation directly to internal BoxData primitives
        without building or touching BMesh / Blender object.
        """
        if not self.boxes:
            return
        op_tr = op.resolve(self)
        delta_tr = space * op_tr * space.inverse
        for box in self.boxes:
            box.transform = delta_tr * box.transform

    @override
    def copy(self) -> "BoxSetPart":
        """Deep copies box descriptors without immediately duplicating Blender mesh data."""
        boxes_copy = [
            BoxSetPart.Box(
                transform=b.transform.copy(),
                size=b.size.copy(),
                custom_data=b.custom_data,
            )
            for b in self.boxes
        ]
        part = BoxSetPart(
            boxes=boxes_copy,
            topology=self.topology_config,
            transform=self.transform.copy(),
        )
        part._obj_cache = self._obj_cache
        part._vertices_cache = self._vertices_cache
        return part

    @override
    def remove(self, physical=True):
        """Safely cleans up Blender object and mesh references."""
        if physical and self._obj_cache and self.is_physically_valid:
            mesh_data = self._obj_cache.data
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)
            if self._obj_cache.name in bpy.data.objects:
                bpy.data.objects.remove(self._obj_cache, do_unlink=True)
            self._obj_cache = None
        self.boxes.clear()

    def invalidate_mesh_cache(self):
        self._obj_cache = None
        self._vertices_cache = None
        self._bbox_cache = None
