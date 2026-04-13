from abc import ABC, abstractmethod
import bpy
from typing import Dict, Literal, NamedTuple, Optional, Union
from mathutils import Vector, Matrix

from .common import Axis, VectorLike, extract_vector
from .material import build_material, mat
from .location import Location, Pos, Scale, Transform

class Object(ABC):
    """An object representing a part. Manages its own mesh and Blender object."""
    def __init__(self, obj: Optional[bpy.types.Object] = None):
        self.obj: Optional[bpy.types.Object] = obj or self._create_empty_object()
        self._auto_remove = True

    def __del__(self):
        if self._auto_remove:
            self.remove()

    @property
    def is_valid(self):
        """Checks if the object is valid (not removed)."""
        return self.obj is not None

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
    def transform(self, value: 'Transform'):
        """Sets the transformation of the part."""
        self.obj.matrix_world = value.matrix

    @property
    def loc(self):
        """Access the location of the part."""
        return self.transform.loc
    
    @loc.setter
    def loc(self, loc: 'Location'):
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
    def bbox(self):
        """Access the bounding box of the part."""
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = self.obj.evaluated_get(dg)
        matrix_world = self.transform.matrix
        world_corners = [matrix_world @ Vector(corner) for corner in eval_obj.bound_box]
        xs = [c.x for c in world_corners]
        ys = [c.y for c in world_corners]
        zs = [c.z for c in world_corners]
        class BBox(NamedTuple):
            min = Vector((min(xs), min(ys), min(zs)))
            max = Vector((max(xs), max(ys), max(zs)))
        return BBox()
    
    @abstractmethod
    def _create_empty_object(self):
        raise NotImplementedError
    
    @abstractmethod
    def copy(self) -> 'Object':
        raise NotImplementedError
    
    def remove(self, physical=True):
        """Safely removes the object and its data from the Blender scene."""
        if self.obj:
            if physical:
                bpy.data.objects.remove(self.obj, do_unlink=True)
            self.obj = None

    def show(self, name: str | None = None, hide = False, collection_name: str | None = None):
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

    def _apply_size_change(self, axis_input: Union[Axis, Vector], value: float, mode: Literal['SCALE', 'SIZE']):
        """
        Resizes the object while keeping the opposite side fixed based on axis direction.
        :param axis_input: Axis enum or negated Axis (e.g., Axis.X or -Axis.X)
        :param mode: 'SCALE' or 'SIZE'
        """
        # Convert to vector to handle both Axis and negated Axis (-Axis.X)
        vec = Vector(axis_input.value if isinstance(axis_input, Axis) else axis_input)
        
        # Identify axis index (0, 1, or 2) and the side (-1 or 1)
        # If vec.x is 1.0 -> side is 1 (anchor max). If -1.0 -> side is -1 (anchor min).
        axis_idx = next(i for i, v in enumerate(vec) if v != 0)
        side = int(vec[axis_idx])

        # Reset location
        old_loc = self.loc
        self.loc = Location()

        # Store the anchor point in world space before transformation
        old_bbox = self.bbox
        anchor_point = old_bbox.min[axis_idx] if side == 1 else old_bbox.max[axis_idx]

        # Apply change to scale or size
        if mode == 'SCALE':
            self.scale[axis_idx] = value
        else:
            self.size[axis_idx] = value

        # Calculate drift and compensate position
        new_bbox = self.bbox
        current_side_pos = new_bbox.min[axis_idx] if side == 1 else new_bbox.max[axis_idx]
        
        # Shift the object's world translation to keep the anchor point stationary
        move_vec = Vector((0, 0, 0))
        move_vec[axis_idx] = anchor_point - current_side_pos
        self.transform = old_loc * Pos(move_vec) * self.transform

    def set_scale(self, value: float, axis: Union[Axis, Vector]):
        """Set scale along axis. Use -Axis to anchor the opposite side."""
        self._apply_size_change(axis, value, 'SCALE')

    def set_size(self, value: float, axis: Union[Axis, Vector]):
        """Set size along axis. Use -Axis to anchor the opposite side."""
        self._apply_size_change(axis, value, 'SIZE')

    def shoot(
        self,
        camera_loc: Location,
        fov: float = 39.6,
        resolution: tuple[int, int] = (1024, 1024),
        offset: Location = Location(),
        label: str | None = None
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
        other_objects_render_states: Dict[bpy.types.Object, bool] = {}
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
                label=label or f"shoot_{self.obj.name}"
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
                    pass # Already unlinked or handled elsewhere

    def _get_or_create_material_index(self, material: Optional['mat.Layer'], default: bool = False) -> int:
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