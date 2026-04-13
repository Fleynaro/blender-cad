from typing import TYPE_CHECKING, List, Optional, Tuple, Union, overload
from typing_extensions import override
import math
from mathutils import Vector, Matrix, Quaternion, Euler

from .common import CurveLike, VectorLike, extract_vector

if TYPE_CHECKING:
    from .geometry import GeometryEntity, UVSelector, Face

class SVector(Vector):
    """A wrapper for mathutils.Vector that supports the Solver interface."""
    def copy(self) -> 'SVector':
        return SVector(self)

    @property
    def values(self) -> List[float]:
        return list(self)

    @values.setter
    def values(self, vals: List[float]):
        for i, v in enumerate(vals):
            self[i] = v

    @property
    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return [(None, None)] * len(self)

class Transform:
    """Base class for all matrix-based transformations, implementing build123d-style algebra."""
    def __init__(self, mat: Optional[Matrix] = None):
        self.matrix = mat if mat is not None else Matrix.Identity(4)

    def copy(self) -> 'Transform':
        return Transform(self.matrix.copy())

    @property
    def values(self) -> List[float]:
        """9 Degrees of Freedom: Translation (3), Rotation Euler (3), Scale (3)."""
        return list(self.position) + list(self.rotation) + list(self.scale)

    @values.setter
    def values(self, vals: List[float]):
        loc = Vector(vals[0:3])
        rot = Euler([math.radians(a) for a in vals[3:6]], 'XYZ')
        scale = Vector(vals[6:9])
        self.matrix = Matrix.LocRotScale(loc, rot, scale)

    @property
    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Translation bounds (None), Rotation bounds (None), Scale bounds (None)."""
        return [(None, None)] * 9

    @overload
    def __mul__(self, other: 'Transform') -> 'Transform': ...

    @overload
    def __mul__(self, other: VectorLike) -> Vector: ...
    
    def __mul__(self, other: Union['Transform', VectorLike]) -> Union['Transform', Vector]:
        if isinstance(other, Transform):
            return Transform(self.matrix @ other.matrix)
        return self.matrix @ extract_vector(other)
    
    @property
    def loc(self) -> 'Location':
        """Returns the location component as a Location."""
        return Location(self.matrix)
    
    @property
    def position_loc(self) -> 'Location':
        """Returns the position component as a Location."""
        return Pos(self.position)
    
    @property
    def rotation_loc(self) -> 'Location':
        """Returns the rotation component as a Location."""
        return Location(self.quaternion.to_matrix().to_4x4())

    @property
    def position(self) -> Vector:
        """Returns the translation component of the transformation."""
        return self.matrix.to_translation()
    
    @property
    def quaternion(self) -> Quaternion:
        """Returns the rotation component as a Quaternion."""
        return self.matrix.to_quaternion()
    
    @property
    def rotation(self) -> Vector:
        """Returns the rotation component as an Euler in degrees."""
        return Vector([math.degrees(a) for a in self.euler_rad])
    
    @property
    def euler_rad(self) -> Euler:
        """Returns the rotation component as an Euler in radians."""
        return self.matrix.to_euler()

    @property
    def scale(self) -> Vector:
        """Returns the scale component of the transformation."""
        return self.matrix.to_scale()

    @property
    def inverse(self) -> 'Transform':
        """Returns the inverse transformation."""
        return Transform(self.matrix.inverted())
    
    @property
    def x(self):
        """Returns the x component of the position."""
        return self.position.x
    
    @property
    def y(self):
        """Returns the y component of the position."""
        return self.position.y
    
    @property
    def z(self):
        """Returns the z component of the position."""
        return self.position.z
    
    @property
    def rx(self):
        """Returns the x component of the rotation."""
        return self.rotation.x
    
    @property
    def ry(self):
        """Returns the y component of the rotation."""
        return self.rotation.y
    
    @property
    def rz(self):
        """Returns the z component of the rotation."""
        return self.rotation.z
    
    @property
    def sx(self):
        """Returns the x component of the scale."""
        return self.scale.x
    
    @property
    def sy(self):
        """Returns the y component of the scale."""
        return self.scale.y
    
    @property
    def sz(self):
        """Returns the z component of the scale."""
        return self.scale.z

class Scale(Transform):
    """A transformation representing uniform scaling."""
    @overload
    def __init__(self, factor: VectorLike | 'Scale' = 1): ...

    @overload
    def __init__(self, *, X: float = 1, Y: float = 1, Z: float = 1, 
                 XY: float = 1, YZ: float = 1, XZ: float = 1, XYZ: float = 1): ...

    def __init__(self, *args, **kwargs):
        if len(args) == 3:
            final_vec = args
        elif len(args) == 1:
            if isinstance(args[0], Scale):
                super().__init__(args[0].matrix)
                return
            final_vec = args[0]
        elif kwargs:
            base = kwargs.get('XYZ', 1.0)

            x_val = base * kwargs.get('X', 1.0) * kwargs.get('XY', 1.0) * kwargs.get('XZ', 1.0)
            y_val = base * kwargs.get('Y', 1.0) * kwargs.get('XY', 1.0) * kwargs.get('YZ', 1.0)
            z_val = base * kwargs.get('Z', 1.0) * kwargs.get('XZ', 1.0) * kwargs.get('YZ', 1.0)
            
            final_vec = (x_val, y_val, z_val)
        else:
            final_vec = 1.0
        mat = Matrix.Diagonal((*extract_vector(final_vec), 1.0))
        super().__init__(mat)

    def copy(self) -> 'Scale':
        return Scale(self.values)

    @property
    def values(self):
        return super().values[6:9]

    @values.setter
    def values(self, vals: List[float]):
        current = Transform.values.fget(self)
        current[6:9] = vals
        Transform.values.fset(self, current)

    @property
    def bounds(self):
        return super().bounds[6:9]

FlipX = Scale(X=-1)
FlipY = Scale(Y=-1)
FlipZ = Scale(Z=-1)

class Location(Transform):
    def __init__(self, mat: Optional[Matrix] = None, parent_loc: Optional['Location'] = None):
        if mat is not None:
            # Remove scale
            pos = mat.to_translation()
            rot = mat.to_quaternion()
            mat = Matrix.LocRotScale(pos, rot, Vector((1.0, 1.0, 1.0)))
            if parent_loc is not None:
                mat = parent_loc.matrix @ mat
        super().__init__(mat)
        self._parent_loc = parent_loc

    def copy(self) -> 'Location':
        return Location(self.matrix.copy(), self._parent_loc)

    @property
    def values(self) -> List[float]:
        return super().values[0:6]

    @values.setter
    def values(self, vals: List[float]):
        current = Transform.values.fget(self)
        current[0:6] = vals
        Transform.values.fset(self, current)

    @property
    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        return super().bounds[0:6]

    @property
    def local(self) -> 'Location':
        if not self._parent_loc:
            return self
        return self._parent_loc.inverse * self

    @property
    def inverse(self) -> 'Location':
        return Location(super().inverse.matrix)
    
    def look_at(self, target: Union['Location', VectorLike], flip_z = False) -> 'Location':
        """
        Returns a new Location at the current position, rotated to point at the target.
        Looks along local Z axis. If flip_z is True, looks along local -Z axis (suitable for cameras).
        """
        target_vec = target.position if isinstance(target, Location) else extract_vector(target)
        curr_pos = self.position
        direction = target_vec - curr_pos

        # Avoid zero-length vector issues
        if direction.length_squared < 1e-12:
            return self.copy()

        # track='Z' means that axis points to target
        # up='Y' means that axis points to world Up
        rot_quat = direction.to_track_quat('-Z' if flip_z else 'Z', 'Y')

        # Combine current translation with new look-at rotation
        new_mat = Matrix.LocRotScale(curr_pos, rot_quat, Vector((1.0, 1.0, 1.0)))
        return Location(new_mat)
    
    def reorder_axes(self, order: str = "XYZ") -> 'Location':
        """
        Returns a new Location with reordered basis vectors.
        Example: order="ZXY" means New_X = Old_Z, New_Y = Old_X, New_Z = Old_Y.
        """
        order = order.upper()
        if len(order) != 3 or set(order) != {"X", "Y", "Z"}:
            raise ValueError("Order must be a permutation of 'XYZ' (e.g., 'ZYX')")

        mapping = {"X": 0, "Y": 1, "Z": 2}
        old_mat = self.matrix.to_3x3()
        new_mat = Matrix.Identity(3)

        for i, axis in enumerate(order):
            # Column i of new matrix is Column mapping[axis] of old matrix
            new_mat.col[i] = old_mat.col[mapping[axis]]

        res_mat = Matrix.Translation(self.position) @ new_mat.to_4x4()
        return Location(res_mat)
    
    def x_as_z(self) -> 'Location':
        """Returns a new Location with the X axis aligned with the Z axis."""
        return self.reorder_axes("ZYX")
    
    def y_as_z(self) -> 'Location':
        """Returns a new Location with the Y axis aligned with the Z axis."""
        return self.reorder_axes("XZY")
    
    @overload
    def __mul__(self, other: 'Location') -> 'Location': ...

    @overload
    def __mul__(self, other: 'Transform') -> 'Transform': ...

    @overload
    def __mul__(self, other: VectorLike) -> Vector: ...
    
    @override
    def __mul__(self, other: Union['Transform', VectorLike]) -> Union['Transform', Vector]:
        # If the other operand is a dynamic SurfaceLocation, wrap it with 'self' as a parent
        # instead of immediate matrix multiplication. This preserves the Face context.
        if isinstance(other, SurfaceLocation):
            return other.copy(parent_loc=self)

        res = super().__mul__(other)
        # If result is a Transform and 'other' was a Location, keep it as a Location
        if isinstance(other, Location):
            return Location(res.matrix)
        return res

class Pos(Location):
    """A transformation representing translation (position) only."""
    @overload
    def __init__(self, vector: VectorLike = 0): ...

    @overload
    def __init__(self, X: float = 0, Y: float = 0, Z: float = 0,
                XY: float = 1, YZ: float = 1, XZ: float = 1, XYZ: float = 1): ...

    def __init__(self, *args, **kwargs):
        if len(args) == 3:
            final_vec = args
        elif len(args) == 1:
            final_vec = extract_vector(args[0])
        elif kwargs:
            base = kwargs.get('XYZ', 0.0)

            x_val = base + kwargs.get('X', 0.0) + kwargs.get('XY', 0.0) + kwargs.get('XZ', 0.0)
            y_val = base + kwargs.get('Y', 0.0) + kwargs.get('XY', 0.0) + kwargs.get('YZ', 0.0)
            z_val = base + kwargs.get('Z', 0.0) + kwargs.get('XZ', 0.0) + kwargs.get('YZ', 0.0)
            
            final_vec = (x_val, y_val, z_val)
        else:
            final_vec = (0.0, 0.0, 0.0)
        super().__init__(Matrix.Translation(final_vec))

    def copy(self) -> 'Pos':
        return Pos(self.values)

    @property
    def values(self):
        return super().values[0:3]

    @values.setter
    def values(self, vals: List[float]):
        current = Transform.values.fget(self)
        current[0:3] = vals
        Transform.values.fset(self, current)

    @property
    def bounds(self):
        return super().bounds[0:3]

class Rot(Location):
    """A transformation representing rotation only (angles in degrees)."""
    @overload
    def __init__(self, angles: VectorLike = 0): ...

    @overload
    def __init__(self, X: float = 0, Y: float = 0, Z: float = 0,
                 XY: float = 1, YZ: float = 1, XZ: float = 1, XYZ: float = 1): ...

    def __init__(self, *args, **kwargs):
        if len(args) == 3:
            angles = args
        elif len(args) == 1:
            angles = extract_vector(args[0])
        elif kwargs:
            base = kwargs.get('XYZ', 0.0)

            x_val = base + kwargs.get('X', 0.0) + kwargs.get('XY', 0.0) + kwargs.get('XZ', 0.0)
            y_val = base + kwargs.get('Y', 0.0) + kwargs.get('XY', 0.0) + kwargs.get('YZ', 0.0)
            z_val = base + kwargs.get('Z', 0.0) + kwargs.get('XZ', 0.0) + kwargs.get('YZ', 0.0)
            
            angles = (x_val, y_val, z_val)
        else:
            angles = (0.0, 0.0, 0.0)
            
        rad_angles = [math.radians(a) for a in angles]
        euler = Euler(rad_angles, 'XYZ')
        super().__init__(euler.to_matrix().to_4x4())

    def copy(self) -> 'Rot':
        return Rot(self.values)

    @property
    def values(self):
        return super().values[3:6]

    @values.setter
    def values(self, vals: List[float]):
        current = Transform.values.fget(self)
        current[3:6] = vals
        Transform.values.fset(self, current)

    @property
    def bounds(self):
        return super().bounds[3:6]

class SurfaceLocation(Location):
    """
    A specialized Location that represents a surface (Face).
    Maps X, Y coordinates to U, V surface parameters and Z to the normal offset.
    """
    def __init__(
        self, 
        face: 'Face', 
        uv: Optional['UVSelector'] = None, 
        u: float = 0.0, 
        v: float = 0.0, 
        z: float = 0.0,
        rotation: Optional[Location] = None,
        parent_loc: Optional['Location'] = None,
        intrinsic_matrix: Optional[Matrix] = None
    ):
        from .geometry import uv as uv_factory
        self.face = face
        self.uv_selector = uv or uv_factory.set(0, 0)
        self.u_offset = u
        self.v_offset = v
        self.z_offset = z
        self.local_rotation = rotation or Location()
        
        if intrinsic_matrix is None:
            # 1. Evaluate the physical matrix at the current UV + offsets
            # face.at returns a Location (matrix) with Z aligned to the normal
            base_eval = self.face.at(self.uv_selector.offset_m(self.u_offset, self.v_offset))
            
            # 2. Combine: Base Surface Point * Normal Offset * Local Rotation
            # This becomes the "intrinsic" matrix of this specific surface location
            intrinsic_matrix = (base_eval * Pos(0, 0, self.z_offset) * self.local_rotation).matrix
        
        # 3. Initialize parent Location with the calculated matrix
        super().__init__(mat=intrinsic_matrix, parent_loc=parent_loc)

    @override
    def __mul__(self, other: Union['Transform', VectorLike]) -> Union['Transform', Vector]:
        if isinstance(other, Location):
            # Combine current offsets with the incoming transform's position
            new_u = self.u_offset + other.x
            new_v = self.v_offset + other.y
            new_z = self.z_offset + other.z
            
            # Combine rotations: current_rotation * incoming_rotation
            new_rotation = self.local_rotation * other.rotation_loc
            
            return SurfaceLocation(
                face=self.face,
                uv=self.uv_selector,
                u=new_u,
                v=new_v,
                z=new_z,
                rotation=new_rotation,
                parent_loc=self._parent_loc
            )
        
        # If multiplying by a Vector, use the pre-calculated self.matrix
        return super().__mul__(other)

    def copy(self, parent_loc = Location()) -> 'SurfaceLocation':
        cur_parent_loc = self._parent_loc or Location()
        return SurfaceLocation(
            self.face, self.uv_selector, 
            self.u_offset, self.v_offset, self.z_offset,
            self.local_rotation, parent_loc * cur_parent_loc,
            intrinsic_matrix=cur_parent_loc.inverse.matrix @ self.matrix
        )

class Locations:
    """
    Context manager for managing insertion points.
    Supports nesting and matrix multiplication.
    """
    _stack: List[List[Location]] = []

    def __init__(self, *pts: Union[VectorLike, Location, 'GeometryEntity']):
        from .geometry import GeometryEntity, Face
        self.local_locations: List[Location] = []
        for p in pts:
            if isinstance(p, Location):
                self.local_locations.append(p)
            elif isinstance(p, (Vector, tuple, list)):
                self.local_locations.append(Pos(*p))
            elif isinstance(p, GeometryEntity):
                if isinstance(p, Face):
                    self.local_locations.append(p.at(0.5, 0.5))
                else:
                    self.local_locations.append(p.center())
        
        if not self.local_locations:
            self.local_locations = [Location()]

    def __enter__(self):
        from .build_part import BuildPart
        ctx = BuildPart._get_context()
        # Get the current top of the LOCAL stack
        parent_locs = ctx._locations_stack[-1]
        
        # Multiply current locations with new ones (nesting)
        combined = []
        for p_loc in parent_locs:
            for c_loc in self.local_locations:
                combined.append(p_loc * c_loc)
        
        # Push to the context-specific stack
        ctx._locations_stack.append(combined)
        return self

    def __exit__(self, *_):
        from .build_part import BuildPart
        ctx = BuildPart._get_context()
        ctx._locations_stack.pop()

    def __iter__(self):
        return iter(self.local_locations)
    
    def __len__(self):
        return len(self.local_locations)

    @classmethod
    def _get_active(cls) -> List[Location]:
        """Returns the active locations from the current BuildPart context."""
        from .build_part import BuildPart
        try:
            ctx = BuildPart._get_context()
            return ctx._locations_stack[-1]
        except RuntimeError:
            return [Location()]

class GridLocations(Locations):
    """Generates a grid of locations with specified spacing and count."""
    def __init__(self, x_spacing: float, y_spacing: float, x_count: int, y_count: int):
        locs = []
        offset_x = (x_count - 1) * x_spacing / 2
        offset_y = (y_count - 1) * y_spacing / 2
        for i in range(x_count):
            for j in range(y_count):
                locs.append(Pos(i * x_spacing - offset_x, j * y_spacing - offset_y, 0))
        super().__init__(*locs)

class PolarLocations(Locations):
    """Generates locations arranged in a circle."""
    def __init__(self, radius: float, count: int, start_angle: float = 0):
        locs = []
        for i in range(count):
            angle = math.radians(start_angle + (i * 360 / count))
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            # In build123d, polar points are typically oriented along the circle
            locs.append(Pos(x, y, 0) * Rot(0, 0, math.degrees(angle)))
        super().__init__(*locs)

class HexLocations(Locations):
    """Generates locations in a hexagonal pattern."""
    def __init__(self, apothem: float, x_count: int, y_count: int):
        locs = []
        s = apothem / math.cos(math.radians(30))
        for i in range(x_count):
            for j in range(y_count):
                x = i * (3/2 * s)
                y = j * (2 * apothem) + (i % 2) * apothem
                locs.append(Pos(x, y, 0))
        super().__init__(*locs)

class CurveLocations(Locations):
    """Generates locations along a CurveLike object based on count, spacing, or offsets."""
    
    def __init__(
        self, 
        curve: CurveLike, 
        count: Optional[int] = None, 
        spacing: Optional[float] = None, 
        offsets: Optional[List[float]] = None,
        offsets_m: Optional[List[float]] = None,
    ):
        locs = []

        # 1. Specific distance offsets in meters
        if offsets_m is not None:
            for dist in offsets_m:
                # Clamp distance within [0, length] to avoid evaluation errors
                safe_dist = max(0.0, min(curve.length(), dist))
                locs.append(curve.at(t_m=safe_dist))

        # 2. Specific 't' parameter offsets (0.0 to 1.0)
        elif offsets is not None:
            for t in offsets:
                locs.append(curve.at(t=t))
        
        # 3. Linear distribution by number of points
        elif count is not None:
            if count < 2:
                # Fallback to midpoint for single count
                locs.append(curve.at(t=0.5))
            else:
                for i in range(count):
                    t = i / (count - 1)
                    locs.append(curve.at(t=t))
        
        # 4. Distribution by fixed interval spacing
        elif spacing is not None:
            total_len = curve.length()
            # Small epsilon to catch the final point if it lands exactly on the spacing
            actual_count = int(total_len // spacing) + 1
            for i in range(actual_count):
                dist = i * spacing
                if dist > total_len + 1e-6: 
                    break
                locs.append(curve.at(t_m=dist))
        
        else:
            # Default fallback: Start and End
            locs = [curve.at(t=0.0), 
                    curve.at(t=1.0)]

        super().__init__(*locs)

def align(from_port: Location, to_port: Location, twist: Optional[float] = None, rot: Optional[Quaternion] = None) -> Location:
    """
    Calculates the transformation to align from_port with to_port.
    If twist is None, it uses the shortest arc rotation to minimize extra movement.
    If twist is a value, it aligns axes exactly and applies the specified rotation.
    """
    m_from = from_port.matrix
    m_to = to_port.matrix

    if rot is not None:
        # We want the joint to end up at pos_to.
        # Current world joint orientation = Part_Rotation @ Joint_Relative_Rotation
        final_rotation_q = rot @ m_from.to_quaternion()
    elif twist is None:
        # --- Mode 2: Natural alignment (Shortest Arc) ---
        # Extract world normal vectors (Z-axis)
        z_from = (m_from.to_3x3() @ Vector((0, 0, 1))).normalized()
        z_to = (m_to.to_3x3() @ Vector((0, 0, 1))).normalized()
        
        # Find the minimal rotation to make Z-axes point at each other
        rot_diff = z_from.rotation_difference(-z_to)
        
        # Apply this diff to the current rotation of the port
        final_rotation_q = rot_diff @ m_from.to_quaternion()
    else:
        # --- Mode 3: Absolute alignment (Strict axes match + Twist) ---
        # Get target orientation
        target_quat = m_to.to_quaternion()
        
        # 180-degree flip around X to face each other (Z -> -Z, Y -> -Y)
        flip_x = Quaternion((0, 1, 0, 0))
        
        # Apply user-defined twist around the new local Z axis
        twist_rad = math.radians(twist)
        twist_quat = Quaternion((0, 0, 1), twist_rad)
        
        # Combine: Target * Flip * Twist
        final_rotation_q = target_quat @ flip_x @ twist_quat

    # Construct the final matrix:
    # 1. Translation to to_port position
    # 2. Apply the calculated rotation
    # 3. Subtract the from_port's local offset relative to its own part
    result_matrix = (
        Matrix.Translation(m_to.to_translation()) @ 
        final_rotation_q.to_matrix().to_4x4() @ 
        m_from.inverted()
    )
    return Location(result_matrix)
