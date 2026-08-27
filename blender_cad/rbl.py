from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations
import math
import time
from typing import Any, Iterable, Iterator, Self, Union, List, Dict, Tuple, Optional, Callable
import uuid
from typing_extensions import override
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

from .common import AbstractCurve, CurveLike, PartLike, VectorLike, _flatten_items, extract_curve, extract_object, extract_part, extract_vector, match_tags, tag_to_list, try_extract_vector
from .location import Pos, Rot, Scale, Transform
from .solver import Solver, SolverLike
from .object import Object
from .part import Part
from .curve import BuildCurve, curve
from .modifiers import add
from .build_part import Mode

PartNodeLike = Union['RuleBasedLayout', 'RuleBasedLayout.Query', PartLike]
PosNodeLike = Union[VectorLike, Callable[[], VectorLike], PartNodeLike]
ScalarLike = Union[Union[int, float, Callable[[], int | float]]]

@dataclass
class RuleBinding:
    rule: 'Rule'
    element: 'RuleBasedLayout'
    targets: List['RuleBasedLayout']

class Dof:
    """Tracks a single degree of freedom optimization parameter constraint state."""
    def __init__(self, value: float, min: Optional[float] = None, max: Optional[float] = None, enabled: bool = True, inited: bool = False):
        self.value: float = value
        self.min: Optional[float] = min
        self.max: Optional[float] = max
        self.enabled: bool = enabled
        self.inited: bool = inited

@dataclass
class PartData:
    """Stores extracted geometry part alongside lazily-initialized BVH tree and local vertices."""
    part_like: PartLike
    _part: Optional[Part] = None
    _bvh: Optional[BVHTree] = None
    _vertices: Optional[List[Vector]] = None

    @property
    def part(self):
        """Lazily constructs and caches the extracted part geometry."""
        if self._part is None:
            self._part = extract_part(self.part_like)
        return self._part

    @property
    def bvh(self):
        """Lazily constructs and caches the BVH tree for the extracted part mesh."""
        if self._bvh is None:
            if isinstance(self.part_like, (AbstractCurve, BuildCurve, curve)):
                 self._bvh = extract_curve(self.part_like).build_bvh()
            else:
                self._bvh = self.part.build_bvh()
        return self._bvh

    @property
    def vertices(self) -> List[Vector]:
        """Lazily extracts and caches local mesh vertices as Vector instances."""
        if self._vertices is None:
            self._vertices = self.part.get_vertices()
        return self._vertices

class RuntimeContext:
    """Manages active DOFs, explicit transformation pipelines, and graph multi-parent lookups."""
    _current_context: ContextVar[Optional['RuntimeContext']] = ContextVar('current_context', default=None)

    def __init__(self, root: 'RuleBasedLayout'):
        # Map storing PartData instances per rl node
        self.root = root
        self.part_data: Dict[Tuple[PartLike, str], PartData] = {}
        self.rule_data: Dict[Rule, Dict[str, Any]] = {}
        self._current_rule: Optional[Rule] = None
        self._current_rule_element: Optional[Rule] = None
        self._current_target_element: Optional[RuleBasedLayout] = None
        self.force_soft_rules = False

    def __enter__(self):
        self._token = self._current_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            self._current_context.reset(self._token)

    @classmethod
    def get_current(cls) -> Self:
        ctx = cls._current_context.get()
        if ctx is None:
            raise RuntimeError("RuntimeContext not active")
        return ctx
    
    def get_rule_data(self, rule: 'Rule') -> Dict[str, Any]:
        return self.rule_data.setdefault(rule, {})
    
    def get_dofs(self, target: 'RuleBasedLayout') -> Dict[str, Dof]:
        return { k: Dof(0) for k in ['x', 'y', 'z', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz'] }

    def get_local_transform(self, target: 'RuleBasedLayout') -> Transform:
        obj = extract_object(target.part)
        return obj.transform if obj else Transform()

    def set_local_transform(
        self, 
        target: 'RuleBasedLayout', 
        x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None,
        rx: Optional[float] = None, ry: Optional[float] = None, rz: Optional[float] = None,
        sx: Optional[float] = None, sy: Optional[float] = None, sz: Optional[float] = None
    ) -> None:
        if isinstance(target.part, Object):
            l_tr = target.part.transform
            
            nx = x if x is not None else l_tr.x
            ny = y if y is not None else l_tr.y
            nz = z if z is not None else l_tr.z
            nrx = rx if rx is not None else l_tr.rx
            nry = ry if ry is not None else l_tr.ry
            nrz = rz if rz is not None else l_tr.rz
            nsx = sx if sx is not None else l_tr.sx
            nsy = sy if sy is not None else l_tr.sy
            nsz = sz if sz is not None else l_tr.sz

            target.part.transform = Pos(nx, ny, nz) * Rot(nrx, nry, nrz) * Scale(nsx, nsy, nsz)

    def get_global_transform(self, target: 'RuleBasedLayout') -> Transform:
        return self.get_local_transform(target)

    def set_global_transform(
        self, 
        target: 'RuleBasedLayout', 
        x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None,
        rx: Optional[float] = None, ry: Optional[float] = None, rz: Optional[float] = None,
        sx: Optional[float] = None, sy: Optional[float] = None, sz: Optional[float] = None
    ) -> None:
        self.set_local_transform(target, x, y, z, rx, ry, rz, sx, sy, sz)

    def resolve_target(self, target: PartNodeLike) -> 'RuleBasedLayout':
        if isinstance(target, RuleBasedLayout.Query):
            objects = target.execute()
            if not objects:
                raise ValueError('Could not resolve target')
            return objects[0]
        return RuleBasedLayout._from(target)
    
    def resolve_scalar(self, scalar: ScalarLike, current_target_element: 'RuleBasedLayout') -> float:
        if callable(scalar):
            self._current_target_element = current_target_element
            scalar = scalar()
            self._current_target_element = None
        return float(scalar)

    def resolve_position(self, target: PosNodeLike, current_target_element: 'RuleBasedLayout') -> Vector:
        if callable(target):
            self._current_target_element = current_target_element
            pos = extract_vector(target())
            self._current_target_element = None
            return pos
        
        pos = try_extract_vector(target)
        if pos is not None:
            return pos
        
        return self.resolve_node_position(RuleBasedLayout._from(target))

    def resolve_node_position(self, target: 'RuleBasedLayout'):
        phys_elements = RuleBasedLayout.get_physical_elements(target)
        return sum([self.get_global_transform(l).position for l in phys_elements], Vector()) / len(phys_elements)
    
    def get_part_data(self, target: 'RuleBasedLayout', target_shell_selector: Optional[str] = None) -> PartData:
        """Lazily retrieves or constructs cached PartData for a given LayoutObject."""
        if not target.is_physical:
            raise RuntimeError("Cannot get PartData for non-physical element")
        key = (target.part, "default")
        if key not in self.part_data:
            self.part_data[key] = PartData(part_like=target.part)
        return self.part_data[key]
    
    def initialize_bindings(self, bindings: List['RuleBinding']):
        for bind in bindings:
            self._current_rule = bind.rule
            self._current_rule_element = bind.element
            bind.rule.initialize(self, bind.targets)
        self._current_rule = None
        self._current_rule_element = None

    def evaluate_bindings(self, bindings: List['RuleBinding'], s: Solver.Session):
        for bind in bindings:
            self._current_rule = bind.rule
            self._current_rule_element = bind.element
            bind.rule.evaluate(self, bind.targets, s)
        self._current_rule = None
        self._current_rule_element = None

class DefaultRuntimeContext(RuntimeContext):
    """Manages active DOFs, explicit transformation pipelines, and graph multi-parent lookups."""
    def __init__(self, root: 'RuleBasedLayout'):
        super().__init__(root)
        self._dofs: Dict[str, Dict[str, Dof]] = {}
        self._local_transforms: Dict[str, Transform] = {}
        # Multi-parent tracking matrix topology to support elements sharing multiple groups
        self._parent_map: Dict['RuleBasedLayout', List['RuleBasedLayout']] = {}
        self._init([root])

    def _init(self, objects: List['RuleBasedLayout'], parent: Optional['RuleBasedLayout'] = None):
        """Flattens structural nodes while building graph multi-parent dependencies maps."""
        for obj in objects:
            # Initialize multi-parent map
            if obj not in self._parent_map:
                self._parent_map[obj] = []
            if parent is not None and parent not in self._parent_map[obj]:
                self._parent_map[obj].append(parent)

            # Initialize local transform
            self._local_transforms[obj.id] = super().get_local_transform(obj)

            # SOT Parameters Map: Positional active by default, rotational/scaling disabled
            self._dofs[obj.id] = {
                'x': Dof(0.0, enabled=obj.is_physical),
                'y': Dof(0.0, enabled=obj.is_physical),
                'z': Dof(0.0, enabled=obj.is_physical),
                'rx': Dof(0.0, enabled=False),
                'ry': Dof(0.0, enabled=False),
                'rz': Dof(0.0, enabled=False),
                'sx': Dof(1.0, enabled=False),
                'sy': Dof(1.0, enabled=False),
                'sz': Dof(1.0, enabled=False),
            }

            self._init(obj._get_children(ignore_queries=True), parent=obj)

    def sync_dofs_to_local_transforms(self) -> None:
        """Converts structural SOT solver tracking parameter metrics directly to local assets."""
        for target_id, dof_map in self._dofs.items():
            self.set_local_transform(
                target_id,
                **{ k: dof_map[k].value for k in ['x', 'y', 'z', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz'] if dof_map[k].enabled }
            ),
    
    def init_dofs_from_local_transforms(self) -> None:
        """Initializes SOT solver tracking parameter metrics directly from local assets."""
        for target_id, dof_map in self._dofs.items():
            for k in ['x', 'y', 'z', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']:
                if k not in dof_map or not dof_map[k].enabled or dof_map[k].inited:
                    continue
                dof_map[k].value = getattr(self.get_local_transform(target_id), k)

    @override
    def get_dofs(self, target: 'RuleBasedLayout') -> Dict[str, Dof]:
        return self._dofs[target.id]

    @override
    def get_local_transform(self, target: Union['RuleBasedLayout', 'str']) -> Transform:
        if isinstance(target, str):
            return self._local_transforms[target]
        if target.id not in self._local_transforms:
            return super().get_local_transform(target)
        return self._local_transforms[target.id]

    @override
    def set_local_transform(
        self, 
        target: Union['RuleBasedLayout', 'str'], 
        x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None,
        rx: Optional[float] = None, ry: Optional[float] = None, rz: Optional[float] = None,
        sx: Optional[float] = None, sy: Optional[float] = None, sz: Optional[float] = None
    ) -> None:
        """Directly updates the local transform and syncs changes back to the DOFs (SOT)."""
        target_id = target if isinstance(target, str) else target.id
        if target_id not in self._local_transforms:
            assert isinstance(target, RuleBasedLayout)
            return super().set_local_transform(target, x, y, z, rx, ry, rz, sx, sy, sz)
        
        l_tr = self.get_local_transform(target)
        
        nx = x if x is not None else l_tr.x
        ny = y if y is not None else l_tr.y
        nz = z if z is not None else l_tr.z
        nrx = rx if rx is not None else l_tr.rx
        nry = ry if ry is not None else l_tr.ry
        nrz = rz if rz is not None else l_tr.rz
        nsx = sx if sx is not None else l_tr.sx
        nsy = sy if sy is not None else l_tr.sy
        nsz = sz if sz is not None else l_tr.sz

        self._local_transforms[target_id] = Pos(nx, ny, nz) * Rot(nrx, nry, nrz) * Scale(nsx, nsy, nsz)

    @override
    def get_global_transform(self, target: 'RuleBasedLayout') -> Transform:
        """Recursively resolves aggregated world transformation spaces across multi-parent chains."""
        local_tr = self.get_local_transform(target)
        parents = self._parent_map.get(target, [])
        
        if not parents:
            return local_tr
            
        # Accumulate structural transformation configurations sequentially across graph paths
        combined_parents = Transform()  # Identity
        for p in parents:
            combined_parents = combined_parents * self.get_global_transform(p)
            
        return combined_parents * local_tr

    @override
    def set_global_transform(
        self, 
        target: 'RuleBasedLayout', 
        x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None,
        rx: Optional[float] = None, ry: Optional[float] = None, rz: Optional[float] = None,
        sx: Optional[float] = None, sy: Optional[float] = None, sz: Optional[float] = None
    ) -> None:
        """Applies spatial configurations across all 9 components directly to local spaces."""
        g_tr = self.get_global_transform(target)
        
        # Utilize current transform attributes as fallbacks if values are unspecified
        nx = x if x is not None else g_tr.x
        ny = y if y is not None else g_tr.y
        nz = z if z is not None else g_tr.z
        nrx = rx if rx is not None else g_tr.rx
        nry = ry if ry is not None else g_tr.ry
        nrz = rz if rz is not None else g_tr.rz
        nsx = sx if sx is not None else g_tr.sx
        nsy = sy if sy is not None else g_tr.sy
        nsz = sz if sz is not None else g_tr.sz

        # Recompose using procedural positional/rotational/scaling helper wrappers
        new_g_tr = Pos(nx, ny, nz) * Rot(nrx, nry, nrz) * Scale(nsx, nsy, nsz)

        parents = self._parent_map.get(target, [])
        combined_parents = Transform()
        for p in parents:
            combined_parents = combined_parents * self.get_global_transform(p)
            
        # Project world space updates back into localized relative positioning spaces
        self._local_transforms[target.id] = combined_parents.inverse * new_g_tr

class Rule:
    """Abstract structural constraint base specifying procedural evaluation routines."""
    
    class Scope(Enum):
        SELF = auto()         # Apply to the whole group (as a single object)
        EACH_CHILD = auto()   # Apply to immediate children (1st level)
        EACH_CHILD_WITH_SELF = auto()  # Apply to immediate children (1st level) including the root itself
        DEEP_PHYSICAL = auto()  # Apply recursively only to physical parts
        DEEP_PHYSICAL_WITH_SELF = auto()  # Apply recursively to all physical parts including the PHYSICAL root itself
        DEEP_ALL = auto()     # Apply recursively to all nodes
        DEEP_ALL_WITH_SELF = auto()  # Apply recursively to all nodes including the root itself

    def __init__(self, priority: float = 1.0, init_priority: int = 0, scope: Scope = Scope.DEEP_PHYSICAL_WITH_SELF):
        self.priority: float = priority
        self.init_priority: int = init_priority
        self.scope: Rule.Scope = scope

    def clone(self) -> Self:
        return copy(self)
    
    def on_self(self) -> Self:
        """Returns a new rule instance applied to the whole group."""
        new_rule = self.clone()
        new_rule.scope = Rule.Scope.SELF
        return new_rule

    def on_each(self, include_self: bool = False) -> Self:
        """Returns a new rule instance applied to all immediate children."""
        new_rule = self.clone()
        new_rule.scope = Rule.Scope.EACH_CHILD_WITH_SELF if include_self else Rule.Scope.EACH_CHILD
        return new_rule

    def on_deep_physical(self, include_self: bool = False) -> Self:
        """Returns a new rule instance applied recursively to all physical nodes."""
        new_rule = self.clone()
        new_rule.scope = Rule.Scope.DEEP_PHYSICAL_WITH_SELF if include_self else Rule.Scope.DEEP_PHYSICAL
        return new_rule
        
    def on_deep_all(self, include_self: bool = False) -> Self:
        """Returns a new rule instance applied recursively to all nodes."""
        new_rule = self.clone()
        new_rule.scope = Rule.Scope.DEEP_ALL_WITH_SELF if include_self else Rule.Scope.DEEP_ALL
        return new_rule

    def with_priority(self, priority: float) -> Self:
        """Returns a new rule instance with modified priority."""
        new_rule = self.clone()
        new_rule.priority = priority
        return new_rule
    
    def __or__(self, other: Any) -> 'RuleGroup':
        """Combines two rules into a group: rule1 | rule2"""
        if isinstance(other, Rule):
            return RuleGroup(rules=[self, other])
        raise NotImplementedError

    def __ror__(self, other: Union['RuleBasedLayout.Query', PartLike]) -> 'RuleBasedLayout':
        """Pipelines objects straight into evaluation layout instances: object | rule."""
        if isinstance(other, RuleBasedLayout.Query):
            return RuleBasedLayout(children=[other], rules=[self], tag=RuleBasedLayout.TAG_TAGGED) 
        return RuleBasedLayout(rules=[self], part=other)

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        raise NotImplementedError

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        raise NotImplementedError
    
    @staticmethod
    def group(*args: Union['RuleBasedLayout', PartLike]) -> 'RuleGroup':
        return RuleGroup(rules=list(_flatten_items(args)))
    

class RuleGroup(Rule):
    """A composite rule container that aggregates multiple rules and applies them together."""
    def __init__(self, rules: List[Rule], priority: float = 1.0):
        super().__init__(priority=priority)
        self.rules: List[Rule] = []
        for r in rules:
            if isinstance(r, RuleGroup):
                self.rules.extend(r.rules)
            else:
                self.rules.append(r)

    def __or__(self, other: Any) -> 'Rule':
        if isinstance(other, Rule):
            return RuleGroup(rules=self.rules + [other])
        return NotImplemented

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        for rule in self.rules:
            rule.initialize(context, elements)

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        for rule in self.rules:
            rule.evaluate(context, elements, s)


class TransformRule(Rule):
    """Overrides optimization DOFs, forcing targeted fields to explicit fixed properties."""
    def __init__(
        self, 
        x: Optional[Union[float, Callable[[], float]]] = None,
        y: Optional[Union[float, Callable[[], float]]] = None,
        z: Optional[Union[float, Callable[[], float]]] = None,
        rx: Optional[Union[float, Callable[[], float]]] = None,
        ry: Optional[Union[float, Callable[[], float]]] = None,
        rz: Optional[Union[float, Callable[[], float]]] = None,
        sx: Optional[Union[float, Callable[[], float]]] = None,
        sy: Optional[Union[float, Callable[[], float]]] = None,
        sz: Optional[Union[float, Callable[[], float]]] = None,
        local: bool = True,
        init_only: bool = False
    ):
        super().__init__(init_priority=1000000 - 1)
        # Filter out None values to build the active transform overrides map
        self.transforms: Dict[str, Union[float, Callable[[], float]]] = {
            k: v for k, v in {
                'x': x, 'y': y, 'z': z,
                'rx': rx, 'ry': ry, 'rz': rz,
                'sx': sx, 'sy': sy, 'sz': sz
            }.items() if v is not None
        }
        self.local: bool = local
        self.init_only: bool = init_only

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        if not self.init_only:
            for e in elements:
                for key in self.transforms.keys():
                    if key in context.get_dofs(e):
                        context.get_dofs(e)[key].enabled = False
        self._eval(context, elements)

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        if self.init_only:
            return
        self._eval(context, elements)

    def _eval(self, context: RuntimeContext, elements: List['RuleBasedLayout']):
        for e in elements:
            # Resolve properties (supporting static values and runtime lambda expressions)
            resolved = {k: (v() if callable(v) else v) for k, v in self.transforms.items()}
            if self.local:
                context.set_local_transform(e, **resolved)
            else:
                context.set_global_transform(e, **resolved)


class CollisionRule(Rule):
    """Evaluates surface collisions and spatial proximity using mesh vertices and BVH trees from RuntimeContext."""

    class Mode(Enum):
        OUTSIDE = auto()    # Object A is outside Object B (non-intersecting)
        INSIDE = auto()     # Object A is inside Object B (non-intersecting boundary)
        INTERSECT = auto()  # Objects must intersect or overlap

    def __init__(
        self,
        target: Optional[PartNodeLike] = None,
        target_shell_selector: Optional[str] = None,
        mode: Mode = Mode.OUTSIDE,
        margin: float = 0.0,
        full_check: bool = False,
        shell_override: Optional[PartNodeLike] = None
    ):
        super().__init__(priority=1000000.0)
        self._target = target
        self.target_shell_selector = target_shell_selector
        self.mode = mode
        self.margin = margin
        self.full_check = full_check
        self._shell_override = shell_override

    @property
    def target(self) -> Optional['RuleBasedLayout']:
        ctx = RuntimeContext.get_current()
        data = ctx.get_rule_data(self)
        if 'target' not in data:
            data['target'] = ctx.resolve_target(self._target) if self._target else None
        return data['target']
    
    @property
    def shell_override(self) -> Optional['RuleBasedLayout']:
        ctx = RuntimeContext.get_current()
        data = ctx.get_rule_data(self)
        if 'shell_override' not in data:
            data['shell_override'] = ctx.resolve_target(self._shell_override) if self._shell_override else None
        return data['shell_override']

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        if self.mode == self.Mode.INSIDE and self.target is not None:
            target_part_data = self._get_part_data(context, self.target)
            bbox = target_part_data.part.bbox
            for e in elements:
                for axis in ['x', 'y', 'z']:
                    axis_min = getattr(bbox.min, axis)
                    axis_max = getattr(bbox.max, axis)
                    
                    context.get_dofs(e)[axis].min = axis_min
                    context.get_dofs(e)[axis].max = axis_max
                    
                    # Initialize the DOF position to the center of the bounding box
                    context.get_dofs(e)[axis].value = (axis_min + axis_max) / 2.0
                    context.get_dofs(e)[axis].inited = True

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        objects: List[RuleBasedLayout] = []
        for e in elements:
            objects.extend(RuleBasedLayout.get_physical_elements(e))

        if self.target is not None:
            for phys_elem in objects:
                self._check_directed_collision(phys_elem, self.target, context, s)
                if self.full_check:
                    self._check_directed_collision(self.target, phys_elem, context, s)
        else:
            for elem_a, elem_b in combinations(objects, 2):
                self._check_directed_collision(elem_a, elem_b, context, s)
                if self.full_check:
                    self._check_directed_collision(elem_b, elem_a, context, s)

    def _get_part_data(self, context: RuntimeContext, obj: 'RuleBasedLayout') -> PartData:
        if obj is self.target:
            return context.get_part_data(obj, self.target_shell_selector)
        return context.get_part_data(target=self.shell_override or obj)

    def _check_directed_collision(
        self, 
        obj_a: 'RuleBasedLayout', 
        obj_b: 'RuleBasedLayout',
        context: RuntimeContext, 
        s: Solver.Session
    ) -> None:
        """Checks vertices of object A against the BVH tree of object B using RuntimeContext data."""
        part_data_a = self._get_part_data(context, obj_a)
        part_data_b = self._get_part_data(context, obj_b)

        bvh_b = part_data_b.bvh
        if not bvh_b:
            return

        verts_a = part_data_a.vertices
        if not verts_a:
            return

        tr_a = context.get_global_transform(obj_a)
        tr_b = context.get_global_transform(obj_b)
        inv_tr_b = tr_b.inverse

        for v in verts_a:
            # Transform local vertex of A to world space, then to local space of B
            world_v_a = tr_a * v
            local_p = inv_tr_b * world_v_a

            loc, norm, _, _ = bvh_b.find_nearest(local_p)
            if loc is None or norm is None or norm.length == 0.0:
                continue

            # Convert nearest surface point and normal back to world space
            world_loc = tr_b * loc
            world_norm = (tr_b * (loc + norm)) - world_loc
            world_norm.normalize()

            # Calculate signed distance (+ represents outside, - represents inside)
            diff = world_v_a - world_loc
            dist = diff.length
            sign = 1.0 if world_norm.dot(diff) > 0.0 else -1.0
            signed_dist = dist * sign

            # Penalty computation according to current collision mode
            if self.mode == self.Mode.OUTSIDE:
                if signed_dist < self.margin:
                    violation = self.margin - signed_dist
                    s.aim(self.priority * (violation ** 2))

            elif self.mode == self.Mode.INSIDE:
                if signed_dist > -self.margin:
                    violation = signed_dist + self.margin
                    s.aim(self.priority * (violation ** 2))

            elif self.mode == self.Mode.INTERSECT:
                if signed_dist > -self.margin:
                    violation = signed_dist + self.margin
                    s.aim(self.priority * (violation ** 2))


class CurveAtRule(Rule):
    """Pins objects onto analytical parametric paths at parameter t."""
    def __init__(self, curve: CurveLike):
        super().__init__()
        self.curve = extract_curve(curve)

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        for e in elements:
            context.get_dofs(e)['x'].enabled = False
            context.get_dofs(e)['y'].enabled = False
            context.get_dofs(e)['z'].enabled = False
            context.get_dofs(e)['t'] = Dof(0.5, min=0.0, max=1.0, inited=True)

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        for e in elements:
            if 't' in context.get_dofs(e) and context.get_dofs(e)['t'].enabled:
                t_val = context.get_dofs(e)['t'].value
                eval_pos = self.curve.at(t_val)
                context.set_global_transform(e, x=eval_pos.x, y=eval_pos.y, z=eval_pos.z)

class LookAtRule(Rule):
    """Look-at constraint. Disables DOFs and directly sets rotation values."""
    def __init__(self, target: PosNodeLike, soft: bool = False):
        super().__init__()
        self.target = target
        self.soft = soft

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        is_soft = context.force_soft_rules or self.soft
        for e in elements:
            context.get_dofs(e)['rx'].enabled = is_soft
            context.get_dofs(e)['ry'].enabled = is_soft
            context.get_dofs(e)['rz'].enabled = is_soft

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        is_soft = context.force_soft_rules or self.soft
        for e in elements:
            e_tr = context.get_global_transform(e)
            t_pos = context.resolve_position(self.target, e)
            forward_vec = (t_pos - e_tr.position).normalized()
            world_up = Vector((0.0, 1.0, 0.0))
            if abs(forward_vec.dot(world_up)) > 0.999:
                world_up = Vector((0.0, 0.0, 1.0))
            
            up_vec = forward_vec.cross(world_up).normalized()
            left_vec = up_vec.cross(forward_vec).normalized()
            mat = Matrix((
                (forward_vec.x, left_vec.x, up_vec.x),
                (forward_vec.y, left_vec.y, up_vec.y),
                (forward_vec.z, left_vec.z, up_vec.z)
            ))
            
            euler_angles = mat.to_euler()
            target_rx = math.degrees(euler_angles.x)
            target_ry = math.degrees(euler_angles.y)
            target_rz = math.degrees(euler_angles.z)
            
            if is_soft:
                s.aim_equal(e_tr.rotation_loc, Rot(X=target_rx, Y=target_ry, Z=target_rz), k=self.priority)
            else:
                context.set_global_transform(e, rx=target_rx, ry=target_ry, rz=target_rz)


class SizeRule(Rule):
    """Controls element size."""
    
    def __init__(
        self,
        x: Optional[ScalarLike] = None,
        y: Optional[ScalarLike] = None,
        z: Optional[ScalarLike] = None,
        soft: bool = False,
        local: bool = True,
        init_only: bool = False
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.soft: bool = soft
        self.local: bool = local
        self.init_only: bool = init_only

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        if self.init_only:
            self._eval(context, elements)
            return
        
        is_soft = context.force_soft_rules or self.soft
        for e in elements:
            if self.x is not None:
                context.get_dofs(e)['sx'].enabled = is_soft
            if self.y is not None:
                context.get_dofs(e)['sy'].enabled = is_soft
            if self.z is not None:
                context.get_dofs(e)['sz'].enabled = is_soft

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        if self.init_only:
            return
        self._eval(context, elements, s)

    def _eval(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Optional[Solver.Session] = None) -> None:
        is_soft = context.force_soft_rules or self.soft
        for e in elements:
            transform_updates = {}

            e_scale = context.get_global_transform(e).scale
            e_orig_size = context.get_part_data(e).part.orig_size
            e_size = e_scale * e_orig_size
            for i, t_size_scalar in enumerate([self.x, self.y, self.z]):
                if t_size_scalar is None:
                    continue
                t_size_scalar = context.resolve_scalar(t_size_scalar, e)
                assert t_size_scalar > 0.0, 'Negative size is not allowed'

                if is_soft:
                    assert s, 'soft mode requires a solver'
                    s.aim_equal(abs(e_size[i]), t_size_scalar, k=self.priority)
                else:
                    transform_updates[['sx', 'sy', 'sz'][i]] = t_size_scalar / e_orig_size[i] * (1.0 if e_size[i] > 0.0 else -1.0)

            if transform_updates:
                if self.local:
                    context.set_local_transform(e, **transform_updates)
                else:
                    context.set_global_transform(e, **transform_updates)


class GravityRule(Rule):
    """Solver gravity rule pulling elements closer or pushing them away from a spatial target point."""
    def __init__(self, target: PosNodeLike, pull: bool = True):
        super().__init__()
        self.target = target
        self.pull = pull

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        for e in elements:
            context.get_dofs(e)['x'].enabled = True
            context.get_dofs(e)['y'].enabled = True
            context.get_dofs(e)['z'].enabled = True

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        for e in elements:
            e_pos = context.get_global_transform(e).position
            t_pos = context.resolve_position(self.target, e)

            dist_sq = (e_pos - t_pos).length_squared
            
            if self.pull:
                s.aim(self.priority * dist_sq)
            else:
                s.aim(self.priority / (dist_sq + 1e-5))


class StackRule(Rule):
    """Distributes consecutive objects linearly along directional structural layout paths."""
    def __init__(self, direction: VectorLike, gap: Union[float, Tuple[float, float], Callable[[int], float]] = 1.0):
        super().__init__(scope=Rule.Scope.SELF)
        self.direction: Vector = extract_vector(direction).normalized()
        self.gap: Union[float, Tuple[float, float], Callable[[int], float]] = gap

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        for e in elements:
            # Enable translational DOFs for the group itself so it can be positioned by other rules/solver
            context.get_dofs(e)['x'].enabled = True
            context.get_dofs(e)['y'].enabled = True
            context.get_dofs(e)['z'].enabled = True

            # Disable ALL DOFs for all elements inside this group to let the rule fully control them
            for phys_elem in e.all_children:
                for dof in context.get_dofs(phys_elem).values():
                    dof.enabled = False

            # If the gap is specified as a range interval, register a new parameter (DOF) on the group
            if isinstance(self.gap, tuple):
                g_min, g_max = self.gap
                context.get_dofs(e)['gap'] = Dof((g_min + g_max) / 2.0, min=g_min, max=g_max, inited=True)

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        # Outer loop: iterate over the groups (elements)
        for e in elements:
            current_offset = 0.0
            
            # Inner loop: stack the physical elements inside the local space of the current group
            for idx, phys_elem in enumerate(e._get_phys_elements_deep()):
                # Set local transform of the relative to its parent group
                offset = self.direction * current_offset
                context.set_local_transform(phys_elem, x=offset.x, y=offset.y, z=offset.z)
                
                # Resolve the gap step based on its configuration
                if isinstance(self.gap, tuple):
                    step_gap = context.get_dofs(e)['gap'].value if 'gap' in context.get_dofs(e) else (self.gap[0] + self.gap[1]) / 2.0
                elif callable(self.gap):
                    step_gap = self.gap(idx)
                else:
                    step_gap = self.gap
                    
                current_offset += float(step_gap)


class DofRule(Rule):
    """Configures parameter optimization state flags across execution scopes."""
    def __init__(
        self,
        x: Optional[bool] = None,
        y: Optional[bool] = None,
        z: Optional[bool] = None,
        rx: Optional[bool] = None,
        ry: Optional[bool] = None,
        rz: Optional[bool] = None,
        sx: Optional[bool] = None,
        sy: Optional[bool] = None,
        sz: Optional[bool] = None,
        pos: Optional[bool] = None,
        rot: Optional[bool] = None,
        scale: Optional[bool] = None,
    ):
        super().__init__(init_priority=1000000)
        self.dof_settings: Dict[str, bool] = {
            k: v for k, v in {
                'x': x, 'y': y, 'z': z,
                'rx': rx, 'ry': ry, 'rz': rz,
                'sx': sx, 'sy': sy, 'sz': sz,
                'pos': pos, 'rot': rot, 'scale': scale
            }.items() if v is not None
        }

    def initialize(self, context: RuntimeContext, elements: List['RuleBasedLayout']) -> None:
        for e in elements:
            for k, enabled in self.dof_settings.items():
                if k in ['pos', 'rot', 'scale']:
                    prefix = {'pos': '', 'rot': 'r', 'scale': 's'}[k]
                    for axis in ['x', 'y', 'z']:
                        context.get_dofs(e)[prefix + axis].enabled = enabled
                elif k in context.get_dofs(e):
                    context.get_dofs(e)[k].enabled = enabled

    def evaluate(self, context: RuntimeContext, elements: List['RuleBasedLayout'], s: Solver.Session) -> None:
        pass


class RuleBasedLayout:
    """Manages hierarchical structural scenes, compiling evaluation pipelines."""

    Rule = Rule
    CollisionRule = CollisionRule
    RuntimeContext = RuntimeContext
    PartData = PartData
    PartNodeLike = PartNodeLike
    PosNodeLike = PosNodeLike

    TAG_OBJECT = "rl:object"
    TAG_GROUP = "rl:group"
    TAG_TAGGED = "rl:tagged"

    class Query(ABC):
        @abstractmethod
        def execute(self) -> List['RuleBasedLayout']:
            raise NotImplementedError
        
        def to_group(self):
            return RuleBasedLayout.group(self)

    class QueryFilterChain(Query):
        Callback = Callable[['RuleBasedLayout'], bool]

        @dataclass
        class Context:
            seen: set['RuleBasedLayout.QueryFilterChain'] = field(default_factory=set)

        _ctx: ContextVar[Optional[Context]] = ContextVar(
            "_context",
            default=None,
        )

        def __init__(
            self,
            filter_chain: list[Callback] = [],
            root: Optional['RuleBasedLayout'] = None
        ):
            self.filter_chain = filter_chain
            self.root = root

        @override
        def execute(self) -> List['RuleBasedLayout']:
            with self._context() as ctx:
                if self in ctx.seen:
                    return []
                ctx.seen.add(self)
                result = (self.root or RuntimeContext.get_current().root).all_children
                for query in self.filter_chain:
                    result = [obj for obj in result if query(obj)]
                return result
        
        def filter(self, callback: Callback) -> 'RuleBasedLayout.QueryFilterChain':
            return RuleBasedLayout.QueryFilterChain(filter_chain=self.filter_chain + [callback], root=self.root)

        def tagged(self, *tags: str) -> 'RuleBasedLayout.QueryFilterChain':
            return self.filter(self.query_by_tag(*tags, invert=False))
        
        def untagged(self, *tags: str) -> 'RuleBasedLayout.QueryFilterChain':
            return self.filter(self.query_by_tag(*tags, invert=True))
        
        @staticmethod
        def query_by_tag(*tags: str, invert: bool) -> Callback:
            return lambda obj: match_tags(obj.tags, tags, invert)
        
        @classmethod
        @contextmanager
        def _context(cls):
            current_ctx = cls._ctx.get()
            if current_ctx is not None:
                yield current_ctx
                return
            ctx = cls.Context()
            token = cls._ctx.set(ctx)
            try:
                yield ctx
            finally:
                cls._ctx.reset(token)

    def __init__(
        self,
        children: List[PartNodeLike] = [],
        rules: List[Rule] = [],
        part: Optional[PartLike] = None,
        id: Optional[str] = None,
        tag: str | Iterable[str] = []
    ):
        self._children = [
            child if isinstance(child, (RuleBasedLayout, RuleBasedLayout.Query)) else RuleBasedLayout.object(child)
            for child in children
        ]
        self.rules = rules
        self.part = part
        self.id = id or str(uuid.uuid4())
        self.tags = tag_to_list(tag)

    @property
    def children(self) -> list['RuleBasedLayout']:
        return self._get_children()

    @property
    def all_children(self) -> Iterator['RuleBasedLayout']:
        return self._get_all_children()
    
    def _get_children(
        self,
        ignore_queries = False,
        seen: Optional[set['RuleBasedLayout']] = None,
    ) -> list['RuleBasedLayout']:
        if seen is None:
            seen = set()
        result: List['RuleBasedLayout'] = []
        for child in self._children:
            if isinstance(child, RuleBasedLayout):
                if child in seen:
                    continue
                result.append(child)
                seen.add(child)
            elif not ignore_queries:
                for obj in child.execute():
                    if obj in seen:
                        continue
                    result.append(obj)
                    seen.add(obj)
        return result

    def _get_all_children(
        self,
        seen: Optional[set['RuleBasedLayout']] = None,
    ) -> Iterator['RuleBasedLayout']:
        for child in self._get_children(seen=seen):
            yield child
            yield from child._get_all_children(seen)

    @property
    def all_elements(self) -> list['RuleBasedLayout']:
        return [self] + list(self.all_children)

    @property
    def is_physical(self) -> bool:
        return self.part is not None

    def clone(self) -> Self:
        return RuleBasedLayout(self._children, self.rules, self.part, self.id, self.tags)

    def __or__(self, other: Union[Rule, Iterable[Rule], PartNodeLike, None]) -> 'RuleBasedLayout':
        if other is None:
            return self
        if isinstance(other, Rule):
            return self | [other]
        if isinstance(other, Iterable):
            other_rules = list(other)
            assert all(isinstance(r, Rule) for r in other_rules), 'All rules must be of type Rule.'
            clone = self.clone()
            clone.rules = list(dict.fromkeys(self.rules + other_rules))
            return clone
        return self.group(self, other)

    @staticmethod
    def object(part: PartLike, tag: str | Iterable[str] = []) -> 'RuleBasedLayout':
        """Wraps a part into a single group."""
        return RuleBasedLayout(part=part, tag=tag_to_list(tag) + [RuleBasedLayout.TAG_OBJECT])

    @staticmethod
    def group(*args: PartNodeLike, part: Optional[PartLike] = None, tag: str | Iterable[str] = []) -> 'RuleBasedLayout':
        """Wraps multiple layout elements into a single group."""
        return RuleBasedLayout(children=list(_flatten_items(args)), part=part, tag=tag_to_list(tag) + [RuleBasedLayout.TAG_GROUP])
    
    @staticmethod
    def _from(target: PartNodeLike) -> 'RuleBasedLayout':
        if isinstance(target, RuleBasedLayout):
            return target
        if isinstance(target, RuleBasedLayout.Query):
            return target.to_group()
        return RuleBasedLayout.object(target)
    
    @staticmethod
    def current_rule():
        """Retrieves the current rule being executed."""
        ctx = RuntimeContext.get_current()
        if ctx._current_rule is None:
            raise RuntimeError("No current rule")
        return ctx._current_rule
    
    @staticmethod
    def current_rule_element():
        """Retrieves the current rule element where the current rule is defined."""
        ctx = RuntimeContext.get_current()
        if ctx._current_rule_element is None:
            raise RuntimeError("No current rule element")
        return ctx._current_rule_element

    @staticmethod
    def current_target_element():
        """Retrieves the current layout element being affected by the current rule."""
        ctx = RuntimeContext.get_current()
        if ctx._current_target_element is None:
            raise RuntimeError("No current target element")
        return ctx._current_target_element

    @staticmethod
    def get_transform(elem: 'RuleBasedLayout', local: bool = False):
        """Retrieves the current transform of a layout element."""
        ctx = RuntimeContext.get_current()
        return ctx.get_local_transform(elem) if local else ctx.get_global_transform(elem)

    @staticmethod
    def get_position(elem: 'RuleBasedLayout'):
        """Retrieves the current global position of a layout element."""
        ctx = RuntimeContext.get_current()
        return ctx.resolve_node_position(elem)
    
    @staticmethod
    def get_size(elem: 'RuleBasedLayout', local: bool = False):
        """Retrieves the current size of a layout element."""
        ctx = RuntimeContext.get_current()
        return ctx.get_part_data(elem).part.orig_size * RuleBasedLayout.get_transform(elem, local).scale
    
    @staticmethod
    def get_physical_elements(elem: 'RuleBasedLayout') -> List['RuleBasedLayout']:
        """Retrieves the current global position of a layout element."""
        return [elem] if elem.is_physical else elem._get_phys_elements_deep()

    @staticmethod
    def transform(
        x: Optional[Union[float, Callable[[], float]]] = None,
        y: Optional[Union[float, Callable[[], float]]] = None,
        z: Optional[Union[float, Callable[[], float]]] = None,
        rx: Optional[Union[float, Callable[[], float]]] = None,
        ry: Optional[Union[float, Callable[[], float]]] = None,
        rz: Optional[Union[float, Callable[[], float]]] = None,
        sx: Optional[Union[float, Callable[[], float]]] = None,
        sy: Optional[Union[float, Callable[[], float]]] = None,
        sz: Optional[Union[float, Callable[[], float]]] = None,
        local: bool = True,
        init: bool = False
    ) -> Rule:
        """Applies a transform to a layout element."""
        return TransformRule(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, sx=sx, sy=sy, sz=sz, local=local, init_only=init)

    @staticmethod
    def _collide(
        mode: CollisionRule.Mode,
        target: Optional[PartNodeLike] = None,
        target_shell_selector: Optional[str] = None,
        margin: float = 0.0,
        full_check: bool = False,
        shell_override: Optional[PartNodeLike] = None
    ) -> Rule:
        """Constrains elements to intersect target/each other."""
        return CollisionRule(
            target=target,
            target_shell_selector=target_shell_selector,
            mode=mode,
            margin=margin,
            full_check=full_check,
            shell_override=shell_override
        )

    @staticmethod
    def outside(
        target: Optional[PartNodeLike] = None,
        target_shell_selector: Optional[str] = None,
        margin: float = 0.0,
        full_check: bool = False,
        shell_override: Optional[PartNodeLike] = None
    ) -> Rule:
        """Constrains elements to stay outside target/each other without intersecting."""
        return RuleBasedLayout._collide(
            mode=CollisionRule.Mode.OUTSIDE,
            target=target,
            target_shell_selector=target_shell_selector,
            margin=margin,
            full_check=full_check,
            shell_override=shell_override
        )

    @staticmethod
    def inside(
        target: Optional[PartNodeLike] = None,
        target_shell_selector: Optional[str] = None,
        margin: float = 0.0,
        full_check: bool = False,
        shell_override: Optional[PartNodeLike] = None
    ) -> Rule:
        """Constrains elements to stay inside target boundary without poking out."""
        return RuleBasedLayout._collide(
            mode=CollisionRule.Mode.INSIDE,
            target=target or RuleBasedLayout.query_current_rule_element(),
            target_shell_selector=target_shell_selector,
            margin=margin,
            full_check=full_check,
            shell_override=shell_override
        )
    
    @staticmethod
    def intersect(
        target: Optional[PartNodeLike] = None,
        target_shell_selector: Optional[str] = None,
        margin: float = 0.0,
        full_check: bool = False,
        shell_override: Optional[PartNodeLike] = None
    ) -> Rule:
        """Constrains elements to intersect target/each other."""
        return RuleBasedLayout._collide(
            mode=CollisionRule.Mode.INTERSECT,
            target=target,
            target_shell_selector=target_shell_selector,
            margin=margin,
            full_check=full_check,
            shell_override=shell_override
        )

    @staticmethod
    def at_curve(curve: CurveLike) -> Rule: 
        """Pins elements exactly onto the surface/perimeter of the curve using parametric t."""
        return CurveAtRule(curve)

    @staticmethod
    def stack(direction: VectorLike, gap: Union[float, Tuple[float, float], Callable[[int], float]] = 1.0) -> Rule:
        """Distributes consecutive objects linearly along directional structural layout paths."""
        return StackRule(direction, gap)

    @staticmethod
    def look_at(target: PosNodeLike, soft: bool = False) -> Rule:
        """Creates a look-at constraint, choosing between soft (solver aims) and hard (direct) methods."""
        return LookAtRule(target, soft)

    @staticmethod
    def look_along(direction: VectorLike, soft: bool = False) -> Rule:
        """Rotates the element to look along a constant direction vector by reusing look_at."""
        return RuleBasedLayout.look_at(
            lambda: RuleBasedLayout.get_transform(RuleBasedLayout.current_target_element()).position + extract_vector(direction),
            soft=soft
        )
    
    @staticmethod
    def size(
        x: Optional[ScalarLike] = None,
        y: Optional[ScalarLike] = None,
        z: Optional[ScalarLike] = None,
        soft: bool = False,
        local: bool = True,
        init: bool = False
    ) -> Rule:
        """Applies a size constraint to a layout element."""
        return SizeRule(
            x=x,
            y=y,
            z=z,
            soft=soft,
            local=local,
            init_only=init
        )
    
    @staticmethod
    def grow(size = 10e3) -> Rule:
        """Applies a grow soft constraint to a layout element."""
        return RuleBasedLayout.size(x=size, y=size, z=size, soft=True).with_priority(100.0 / (size ** 2))
    
    @staticmethod
    def shrink() -> Rule:
        """Applies a shrink soft constraint to a layout element."""
        return RuleBasedLayout.size(x=0, y=0, z=0, soft=True)
    
    @staticmethod
    def gravity(
        target: PosNodeLike,
        pull: Optional[bool] = None,
        push: Optional[bool] = None
    ) -> Rule:
        """Creates a gravity constraint, pulling or pushing elements towards a spatial target point."""
        return GravityRule(target, pull=pull if pull is not None else not push)

    @staticmethod
    def configure_dofs(
        x: Optional[bool] = None,
        y: Optional[bool] = None,
        z: Optional[bool] = None,
        rx: Optional[bool] = None,
        ry: Optional[bool] = None,
        rz: Optional[bool] = None,
        sx: Optional[bool] = None,
        sy: Optional[bool] = None,
        sz: Optional[bool] = None,
        pos: Optional[bool] = None,
        rot: Optional[bool] = None,
        scale: Optional[bool] = None,
    ) -> Rule: 
        return DofRule(
            x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, sx=sx, sy=sy, sz=sz, 
            pos=pos, rot=rot, scale=scale
        )
    
    @staticmethod
    def point():
        from .primitives import Point
        return Point(mode=Mode.PRIVATE).create_part()
    
    @staticmethod
    def root(node: Optional[Union['RuleBasedLayout', Query]] = None) -> 'RuleBasedLayout.QueryFilterChain':
        return RuleBasedLayout.QueryFilterChain(root=node and RuleBasedLayout._from(node))
    
    @staticmethod
    def filter(
        callback: QueryFilterChain.Callback,
        root: Optional[Union['RuleBasedLayout', Query]] = None
    ) -> 'RuleBasedLayout.QueryFilterChain':
        return RuleBasedLayout.root(root).filter(callback)
    
    @staticmethod
    def query_current_rule_element() -> 'RuleBasedLayout.Query':
        class Query(RuleBasedLayout.Query):
            def execute(self) -> List['RuleBasedLayout']:
                return [RuleBasedLayout.current_rule_element()]
        return Query()
    
    @staticmethod
    def tagged(*tags: str, root: Optional[Union['RuleBasedLayout', Query]] = None) -> 'RuleBasedLayout.QueryFilterChain':
        return RuleBasedLayout.root(root).tagged(*tags)
    
    @staticmethod
    def untagged(*tags: str, root: Optional[Union['RuleBasedLayout', Query]] = None) -> 'RuleBasedLayout.QueryFilterChain':
        return RuleBasedLayout.root(root).untagged(*tags)

    def _get_phys_elements_deep(self, seen: Optional[set['RuleBasedLayout']] = None) -> List['RuleBasedLayout']:
        """Returns only physical nodes recursively."""
        result: List['RuleBasedLayout'] = []
        for obj in self._get_children(seen=seen):
            if obj.is_physical:
                result.append(obj)
            else:
                result.extend(obj._get_phys_elements_deep(seen))
        return result

    def _resolve_rule_targets(self, rule: Rule) -> List['RuleBasedLayout']:
        if rule.scope == Rule.Scope.SELF:
            return [self]
        elif rule.scope == Rule.Scope.EACH_CHILD:
            return self.children
        elif rule.scope == Rule.Scope.EACH_CHILD_WITH_SELF:
                return self.children + [self]
        elif rule.scope == Rule.Scope.DEEP_PHYSICAL:
            return self._get_phys_elements_deep()
        elif rule.scope == Rule.Scope.DEEP_PHYSICAL_WITH_SELF:
            return self._get_phys_elements_deep() + ([self] if self.is_physical else [])
        elif rule.scope == Rule.Scope.DEEP_ALL:
            return list(self.all_children)
        elif rule.scope == Rule.Scope.DEEP_ALL_WITH_SELF:
            return list(self.all_children) + [self]
        return [self]
    
    def _compile_rules(self) -> List[RuleBinding]:
        """Compiles all rules into a list of active bindings."""
        active_bindings: List[RuleBinding] = []
        for child in self.all_elements:
            for rule in child.rules:
                active_bindings.append(RuleBinding(rule, child, list(dict.fromkeys(child._resolve_rule_targets(rule)))))
        active_bindings.sort(key=lambda x: x.rule.init_priority)
        return active_bindings

    def resolve(self, solver: SolverLike = Solver(), mode=Mode.ADD, instantiation_delay_sec=0.1):
        """Executes the mathematical optimizer loop and generates evaluated scene placements maps."""
        context = DefaultRuntimeContext(self)
        with context:
            active_bindings = self._compile_rules()
        
            # 1. Pipeline initialization loop
            context.initialize_bindings(active_bindings)
            context.init_dofs_from_local_transforms()

            # 2. Solver optimization loop (Solver parameters act as SOT)
            for s in solver:
                for target, dof_map in context._dofs.items():
                    for name, dof in dof_map.items():
                        if dof.enabled:
                            # Feed back mutations downstream from optimizer iterations
                            dof.value = s.param(init_value=dof.value, min=dof.min, max=dof.max)

                # Recompute global spaces based on current parameter states downstream
                context.sync_dofs_to_local_transforms()
                context.evaluate_bindings(active_bindings, s)

            # 3. Instantiate and add evaluated parts directly to the scene
            time.sleep(instantiation_delay_sec) # hack: to avoid crash
            built_id: set[str] = set()
            for child in self.all_elements:
                if child.is_physical and child.id not in built_id:
                    global_transform = context.get_global_transform(child)
                    # Add the underlying geometry part to the scene with its calculated world transform
                    part_data = context.get_part_data(child)
                    add(part_data.part, transform=global_transform, mode=mode)
                    built_id.add(child.id)
    

rl = RuleBasedLayout
