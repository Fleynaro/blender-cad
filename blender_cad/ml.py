import inspect
import math
import re
import time
import types
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    TypeAlias,
    TypedDict,
    Union,
)

from typing_extensions import override

from .build_part import BuildPart, Mode, add_tags, faces, wires
from .chain import _chain_joint_axis_name
from .common import (
    AbstractCurve,
    Axis,
    CurveLike,
    DualMethod,
    PartLike,
    VectorLike,
    _flatten_items,
    extract_curve,
    extract_part,
    tag_to_list,
)
from .curve import BaseCurve, BuildCurve, Curve, FillMode, Polyline, Spline
from .joint import Joint
from .location import (
    FlipX,
    FlipY,
    Location,
    Locations,
    Origin,
    Pos,
    Rot,
    Scale,
    SizeAlongAxis,
    Transform,
    TransformExpr,
)
from .material import mat
from .modifiers import (
    ProportionalEdit,
    add,
    bend,
    delete,
    dissolve,
    make_box_sides_edit,
    subdivide,
    transform,
)
from .modifiers import bevel as bevel_modifier
from .modifiers import extrude as extrude_modifier
from .object import Object
from .part import BoxSetPart, Part
from .primitives import Box
from .rbl import rl
from .shape_list import ShapeList
from .solver import Solver, SolverLike, sm, solver
from .text import Text, t

StyleFn: TypeAlias = Callable[[], "Length"] | Callable[["ml"], "Length"]
Length: TypeAlias = int | float | str | StyleFn
Children: TypeAlias = Union["ml", str, PartLike]
MlItem: TypeAlias = Union[Children, "MLStyle", "rl.Rule", "rl"]


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _percent_to_m(value: str) -> float:
    if not value.endswith("%"):
        raise ValueError(f"value {value!r} is not a percentage")
    return float(value[:-1]) / 100


def _len_to_m(
    value: Optional[Length], ref: float | None, unit_scale: float
) -> float | None:
    """Convert a local unit or percentage value to meters."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value) * unit_scale

    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("%"):
            if ref is None:
                raise ValueError(f"Percentage value {value!r} needs a reference size")
            return _percent_to_m(raw) * ref
        return float(raw) * unit_scale

    raise TypeError(f"Unsupported length value: {type(value)!r}")


def _get_z_epsilon(style: "MLStyle") -> float:
    return max(style.unit_scale * 0.001, 1e-5)


def _fix_subtract_part_size(part: Part, style: "MLStyle"):
    epsilon = _get_z_epsilon(style)
    part.transform *= (
        SizeAlongAxis(Axis.X, part.size.x + epsilon)
        * SizeAlongAxis(-Axis.X, part.size.x + epsilon * 2)
        * SizeAlongAxis(Axis.Y, part.size.y + epsilon)
        * SizeAlongAxis(-Axis.Y, part.size.y + epsilon * 2)
    )


def _box_to_4(
    value: Any, ref: float | None, unit_scale: float
) -> tuple[float, float, float, float]:
    """
    CSS-like shorthand:
    - 1 value  -> all sides
    - 2 values -> top/bottom, left/right
    - 4 values -> top/right/bottom/left
    """
    if value is None:
        return 0.0, 0.0, 0.0, 0.0

    if isinstance(value, (int, float, str)):
        side = _len_to_m(value, ref, unit_scale) or 0.0
        return side, side, side, side

    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            tb = _len_to_m(value[0], ref, unit_scale) or 0.0
            lr = _len_to_m(value[1], ref, unit_scale) or 0.0
            return tb, lr, tb, lr

        if len(value) == 4:
            top = _len_to_m(value[0], ref, unit_scale) or 0.0
            right = _len_to_m(value[1], ref, unit_scale) or 0.0
            bottom = _len_to_m(value[2], ref, unit_scale) or 0.0
            left = _len_to_m(value[3], ref, unit_scale) or 0.0
            return top, right, bottom, left

    raise ValueError(f"Invalid box shorthand: {value!r}")


def _combine_mat(parent_mat: mat.Layer, child_mat: mat.Layer) -> mat.Layer:
    """Combine two materials using the wrapper's additive composition."""
    if parent_mat is None:
        return child_mat
    if child_mat is None:
        return parent_mat
    return parent_mat + child_mat


def _with_alpha(layer: mat.Layer, alpha: float | None) -> mat.Layer:
    """Attach alpha to a material layer if possible."""
    if layer is None or alpha is None:
        return layer
    return layer + mat.PBR(alpha=alpha)


def _bbox_wh(obj: Object) -> tuple[float, float]:
    bb = obj.bbox
    return float(bb.max.x - bb.min.x), float(bb.max.y - bb.min.y)


def _points_bbox(points: Iterable[Any], unit_scale: float = 1.0) -> tuple[float, float]:
    """Compute a simple 2D bounding box from a point list."""
    pts = list(points)
    if not pts:
        return 0.0, 0.0

    xs: list[float] = []
    ys: list[float] = []

    for point in pts:
        if len(point) < 2:
            continue
        xs.append(float(point[0]) * unit_scale)
        ys.append(float(point[1]) * unit_scale)

    if not xs or not ys:
        return 0.0, 0.0

    return max(xs) - min(xs), max(ys) - min(ys)


CornerScalar = float
CornerPair = tuple[float, float]


def _len_to_m_signed(
    value: Optional[Length], ref: float | None, unit_scale: float
) -> float | None:
    """Convert a local unit or percentage value to meters, preserving sign."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value) * unit_scale

    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("%"):
            if ref is None:
                raise ValueError(f"Percentage value {value!r} needs a reference size")
            return float(raw[:-1]) / 100.0 * ref
        return float(raw) * unit_scale

    raise TypeError(f"Unsupported length value: {type(value)!r}")


def _resolve_corner_radii(
    style: "MLStyle", w: float, h: float
) -> tuple[Any, Any, Any, Any]:
    """
    Resolve corner radii with CSS-like priority.

    Returns:
        - circle mode: (tl, tr, br, bl) as signed scalars
        - ellipse mode: (tl, tr, br, bl) as signed (rx, ry)
    """
    unit_scale = style.unit_scale

    def radius(value: Any) -> tuple[float, float]:
        r = _len_to_m_signed(value, min(w, h), unit_scale) or 0.0
        return (r, r)

    tl = tr = br = bl = radius(style.border_radius)

    if style.border_radius_left is not None:
        tl = bl = radius(style.border_radius_left)
    if style.border_radius_right is not None:
        tr = br = radius(style.border_radius_right)
    if style.border_radius_top is not None:
        tl = tr = radius(style.border_radius_top)
    if style.border_radius_bottom is not None:
        bl = br = radius(style.border_radius_bottom)

    if style.border_radius_tl is not None:
        tl = radius(style.border_radius_tl)
    if style.border_radius_tr is not None:
        tr = radius(style.border_radius_tr)
    if style.border_radius_bl is not None:
        bl = radius(style.border_radius_bl)
    if style.border_radius_br is not None:
        br = radius(style.border_radius_br)

    # CSS-like normalization, but preserving signs.
    def _scale_pair(
        a: tuple[float, float], b: tuple[float, float], limit: float
    ) -> float:
        s = abs(a[0]) + abs(b[0])
        if s > limit and s > 1e-9:
            return limit / s
        return 1.0

    sx_top = _scale_pair(tl, tr, w)
    sx_bottom = _scale_pair(bl, br, w)
    sy_left = _scale_pair(tl, bl, h)
    sy_right = _scale_pair(tr, br, h)

    sx = min(sx_top, sx_bottom)
    sy = min(sy_left, sy_right)

    return (
        (tl[0] * sx, tl[1] * sy),
        (tr[0] * sx, tr[1] * sy),
        (br[0] * sx, br[1] * sy),
        (bl[0] * sx, bl[1] * sy),
    )


def _rounded_rect_points_4(
    w: float,
    h: float,
    radii: tuple[Any, Any, Any, Any],
    segments: int = 8,
) -> list[tuple[float, float]]:
    """
    Rounded rect polygon.

    shape="circle"  -> old scalar-radius behavior, supports negative (concave) radii
    shape="ellipse" -> per-axis radii (rx, ry), supports negative (concave) radii too
    """
    pts: list[tuple[float, float]] = []

    def add_point(px: float, py: float) -> None:
        pt = (px, py)
        if not pts or abs(pts[-1][0] - pt[0]) > 1e-9 or abs(pts[-1][1] - pt[1]) > 1e-9:
            pts.append(pt)

    def add_arc(
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        a0: float,
        a1: float,
        inward: bool = False,
    ) -> None:
        rx = abs(rx)
        ry = abs(ry)
        if rx <= 1e-9 or ry <= 1e-9:
            add_point(cx, cy)
            return

        for i in range(1, segments + 1):
            t = i / segments
            a = a0 + (a1 - a0) * t

            x = math.cos(a) * rx
            y = math.sin(a) * ry

            if inward:
                x = -x
                y = -y

            add_point(cx + x, cy + y)

    def as_pair(v: Any) -> tuple[float, float]:
        if isinstance(v, (tuple, list)):
            rx, ry = v
            return float(rx), float(ry)
        r = float(v)
        return r, r

    tl, tr, br, bl = map(as_pair, radii)

    def emit_corner(
        r: tuple[float, float],
        edge_a: tuple[float, float],
        edge_b: tuple[float, float],
        convex_center: tuple[float, float],
        concave_center: tuple[float, float],
        a0: float,
        a1: float,
    ) -> None:
        rx, ry = r
        add_point(*edge_a)

        inward = (rx < 0.0) or (ry < 0.0)
        if inward:
            add_arc(
                concave_center[0],
                concave_center[1],
                rx,
                ry,
                a1,
                a0,
                inward=True,
            )
        else:
            add_arc(
                convex_center[0],
                convex_center[1],
                rx,
                ry,
                a0,
                a1,
                inward=False,
            )

        add_point(*edge_b)

    add_point(abs(tl[0]), 0.0)

    emit_corner(
        tr,
        (w - abs(tr[0]), 0.0),
        (w, abs(tr[1])),
        (w - abs(tr[0]), abs(tr[1])),
        (w, 0.0),
        -math.pi / 2,
        0.0,
    )
    emit_corner(
        br,
        (w, h - abs(br[1])),
        (w - abs(br[0]), h),
        (w - abs(br[0]), h - abs(br[1])),
        (w, h),
        0.0,
        math.pi / 2,
    )
    emit_corner(
        bl,
        (abs(bl[0]), h),
        (0.0, h - abs(bl[1])),
        (abs(bl[0]), h - abs(bl[1])),
        (0.0, h),
        math.pi / 2,
        math.pi,
    )
    emit_corner(
        tl,
        (0.0, abs(tl[1])),
        (abs(tl[0]), 0.0),
        (abs(tl[0]), abs(tl[1])),
        (0.0, 0.0),
        math.pi,
        3 * math.pi / 2,
    )

    return pts


def _warp_side_scales_points(
    points: Iterable[tuple[float, float]],
    w: float,
    h: float,
    top_scale: float = 1.0,
    right_scale: float = 1.0,
    bottom_scale: float = 1.0,
    left_scale: float = 1.0,
) -> list[tuple[float, float]]:
    """Render-only warp for a rectangle-like outline.

    Important:
    - This does NOT affect layout/measure.
    - It is applied only to the visual contour.
    - Works fine for background, border, rounded corners, etc.
    - Strictly fits the resulting shape inside [0, w] x [0, h] without overflow or gaps.
    """
    points_list = list(points)
    if not points_list:
        return []

    if (
        abs(top_scale - 1.0) < 1e-9
        and abs(right_scale - 1.0) < 1e-9
        and abs(bottom_scale - 1.0) < 1e-9
        and abs(left_scale - 1.0) < 1e-9
    ):
        return points_list

    w = max(w, 1e-9)
    h = max(h, 1e-9)
    cx = w * 0.5
    cy = h * 0.5

    # Step 1: Apply initial side-scale warp and compute actual bounding box
    warped: list[tuple[float, float]] = []
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for x, y in points_list:
        u = y / h
        v = x / w

        sx = top_scale + (bottom_scale - top_scale) * u
        sy = left_scale + (right_scale - left_scale) * v

        wx = cx + (x - cx) * sx
        wy = cy + (y - cy) * sy

        warped.append((wx, wy))

        min_x = min(min_x, wx)
        max_x = max(max_x, wx)
        min_y = min(min_y, wy)
        max_y = max(max_y, wy)

    # Step 2: Linearly scale points back to touch [0, w] x [0, h] boundaries exactly
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Prevent division by zero if all points form a flat line
    if range_x < 1e-9 or range_y < 1e-9:
        return warped

    final_points: list[tuple[float, float]] = []
    for wx, wy in warped:
        final_x = (wx - min_x) / range_x * w
        final_y = (wy - min_y) / range_y * h
        final_points.append((final_x, final_y))

    return final_points


def _curve_to_outer_points(
    curve: "Curve", width: Optional[float] = None, height: Optional[float] = None
):
    pts = curve.points
    assert len(pts) > 0
    outer_pts = [(px, py) for px, py, *_rest in pts[0]]

    # normalize to [0, inner_w] x [0, inner_h]
    min_x = min(x for x, _ in outer_pts)
    max_x = max(x for x, _ in outer_pts)
    min_y = min(y for _, y in outer_pts)
    max_y = max(y for _, y in outer_pts)

    src_w = max(max_x - min_x, 1e-9)
    src_h = max(max_y - min_y, 1e-9)

    outer_pts = [
        (
            (x - min_x) / src_w * (width if width is not None else src_w),
            (y - min_y) / src_h * (height if height is not None else src_h),
        )
        for x, y in outer_pts
    ]
    return outer_pts, src_w, src_h


def _pairwise_closed(
    points: list[tuple[float, float]],
) -> Iterator[tuple[tuple[float, float], tuple[float, float]]]:
    """Iterate over consecutive pairs in a closed polyline."""
    if len(points) < 2:
        return
    for i in range(len(points)):
        yield points[i], points[(i + 1) % len(points)]


def _clean_closed_loop(
    points: list[tuple[float, float]], eps: float = 1e-9
) -> list[tuple[float, float]]:
    """Remove duplicates and collapse obvious degenerate points."""
    if len(points) < 2:
        return points[:]

    out: list[tuple[float, float]] = []
    for p in points:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
            out.append(p)

    if (
        len(out) >= 2
        and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= eps
    ):
        out.pop()

    return out


def _simplify_collinear_closed(
    points: list[tuple[float, float]], eps: float = 1e-9
) -> list[tuple[float, float]]:
    """
    Remove vertices that lie on a straight continuation.
    This is what saves polygons on long straight borders.
    """
    pts = _clean_closed_loop(points, eps)
    if len(pts) < 3:
        return pts

    changed = True
    while changed and len(pts) >= 3:
        changed = False
        res: list[tuple[float, float]] = []
        n = len(pts)

        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]

            abx = b[0] - a[0]
            aby = b[1] - a[1]
            bcx = c[0] - b[0]
            bcy = c[1] - b[1]

            cross = abx * bcy - aby * bcx
            dot = abx * bcx + aby * bcy

            # Same direction and on one line -> drop b
            if abs(cross) <= eps and dot >= 0.0:
                changed = True
                continue

            res.append(b)

        pts = res

    return pts


def _closed_loop_segments(
    points: list[tuple[float, float]],
) -> tuple[list[tuple[tuple[float, float], tuple[float, float], float]], float]:
    segs = []
    total = 0.0
    for p0, p1 in _pairwise_closed(points):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        L = math.hypot(dx, dy)
        if L <= 1e-9:
            continue
        segs.append((p0, p1, L))
        total += L
    return segs, total


def _sample_closed_path(
    segs: list[tuple[tuple[float, float], tuple[float, float], float]],
    total: float,
    dist: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return (point, tangent_unit) at distance dist along closed polyline.
    """
    if not segs or total <= 1e-9:
        return (0.0, 0.0), (1.0, 0.0)

    d = dist % total
    acc = 0.0

    for p0, p1, L in segs:
        if acc + L >= d:
            t = (d - acc) / L
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            tx = (p1[0] - p0[0]) / L
            ty = (p1[1] - p0[1]) / L
            return (x, y), (tx, ty)
        acc += L

    p0, p1, L = segs[-1]
    return p1, ((p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L)


def _normalize2(x: float, y: float) -> tuple[float, float]:
    l = math.hypot(x, y)
    if l <= 1e-9:
        return 0.0, 0.0
    return x / l, y / l


def _line_normal(
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> tuple[float, float]:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return _normalize2(-dy, dx)


def _offset_closed_polyline(
    pts: list[tuple[float, float]],
    offset: float,
    miter_limit: float = 4.0,
) -> list[tuple[float, float]]:
    """
    Stable closed offset with SEAM ALIGNMENT FIX.
    """
    if len(pts) < 3:
        return pts

    n = len(pts)
    out: list[tuple[float, float]] = []

    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]

        # edges
        ax, ay = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
        bx, by = p_next[0] - p_curr[0], p_next[1] - p_curr[1]

        ax, ay = _normalize2(ax, ay)
        bx, by = _normalize2(bx, by)

        # normals
        n0x, n0y = -ay, ax
        n1x, n1y = -by, bx

        # bisector
        mx, my = n0x + n1x, n0y + n1y
        ml = math.hypot(mx, my)

        if ml <= 1e-9:
            out.append(
                (
                    p_curr[0] + n0x * offset,
                    p_curr[1] + n0y * offset,
                )
            )
            continue

        mx /= ml
        my /= ml

        denom = mx * n1x + my * n1y
        scale = 1.0 / denom if abs(denom) > 1e-9 else 1.0

        scale = min(scale, miter_limit)

        out.append(
            (
                p_curr[0] + mx * offset * scale,
                p_curr[1] + my * offset * scale,
            )
        )

    # FORCE SEAM CONTINUITY
    # align last point EXACTLY to first
    dx = out[0][0] - out[-1][0]
    dy = out[0][1] - out[-1][1]

    if dx * dx + dy * dy < 1e-6:
        out[-1] = out[0]
    else:
        # if mismatch too big → force closure
        out.append(out[0])

    return out


def _rotate_loop_to_canonical_start(
    pts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Rotate cyclic polygon so it always starts
    from the same deterministic vertex.

    Strategy:
    - smallest Y
    - then smallest X
    """

    if not pts:
        return pts

    start_i = min(
        range(len(pts)),
        key=lambda i: (
            round(pts[i][1], 9),
            round(pts[i][0], 9),
        ),
    )

    return pts[start_i:] + pts[:start_i]


def _part_outline_loops_xy(part: Part) -> list[list[tuple[float, float]]]:
    """
    Return XY outline loops from a flat/mostly-flat part.
    We take the top-most face(s) and extract their wires.
    """
    loops: list[list[tuple[float, float]]] = []
    for face in part.project_2d(remove_doubles=False).faces():
        pts = [(v.co.x, v.co.y) for v in face.bm_verts()]
        if len(pts) < 3:
            continue
        pts = _rotate_loop_to_canonical_start(pts)
        if abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
            pts = pts[:-1]

        if len(pts) >= 3:
            loops.append(pts)
    return loops


def _set_box_model(
    parent_box: Optional["MLBox"],
    style: "MLStyle",
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:

    ref_w = parent_box.w if parent_box else None
    ref_h = parent_box.h if parent_box else None
    unit_scale = style.unit_scale

    padding_top, padding_right, padding_bottom, padding_left = _box_to_4(
        style.padding,
        min(ref_w or 0.0, ref_h or 0.0) or None,
        unit_scale,
    )
    margin_top, margin_right, margin_bottom, margin_left = _box_to_4(
        style.margin,
        min(ref_w or 0.0, ref_h or 0.0) or None,
        unit_scale,
    )

    if style.padding_tb is not None:
        resolved = _len_to_m(style.padding_tb, ref_h, unit_scale)
        if resolved is not None:
            padding_top = padding_bottom = resolved

    if style.padding_lr is not None:
        resolved = _len_to_m(style.padding_lr, ref_w, unit_scale)
        if resolved is not None:
            padding_left = padding_right = resolved

    if style.margin_tb is not None:
        resolved = _len_to_m(style.margin_tb, ref_h, unit_scale)
        if resolved is not None:
            margin_top = margin_bottom = resolved

    if style.margin_lr is not None:
        resolved = _len_to_m(style.margin_lr, ref_w, unit_scale)
        if resolved is not None:
            margin_left = margin_right = resolved

    if style.padding_top is not None:
        resolved = _len_to_m(style.padding_top, ref_h, unit_scale)
        if resolved is not None:
            padding_top = resolved

    if style.padding_right is not None:
        resolved = _len_to_m(style.padding_right, ref_w, unit_scale)
        if resolved is not None:
            padding_right = resolved

    if style.padding_bottom is not None:
        resolved = _len_to_m(style.padding_bottom, ref_h, unit_scale)
        if resolved is not None:
            padding_bottom = resolved

    if style.padding_left is not None:
        resolved = _len_to_m(style.padding_left, ref_w, unit_scale)
        if resolved is not None:
            padding_left = resolved

    if style.margin_top is not None:
        resolved = _len_to_m(style.margin_top, ref_h, unit_scale)
        if resolved is not None:
            margin_top = resolved

    if style.margin_right is not None:
        resolved = _len_to_m(style.margin_right, ref_w, unit_scale)
        if resolved is not None:
            margin_right = resolved

    if style.margin_bottom is not None:
        resolved = _len_to_m(style.margin_bottom, ref_h, unit_scale)
        if resolved is not None:
            margin_bottom = resolved

    if style.margin_left is not None:
        resolved = _len_to_m(style.margin_left, ref_w, unit_scale)
        if resolved is not None:
            margin_left = resolved

    return (padding_top, padding_right, padding_bottom, padding_left), (
        margin_top,
        margin_right,
        margin_bottom,
        margin_left,
    )


def _resolve_flow_alignment_offset_y(
    style: "MLStyle",
    box_h: float,
    content_h: float,
) -> tuple[float, float]:
    """
    Return y offset for standard-flow alignment inside the content box.
    """
    if style.align_y == "center":
        return (box_h - content_h) / 2.0
    elif style.align_y == "end":
        return box_h - content_h
    return 0.0


def _resolve_relative_offset(
    style: "MLStyle", parent_box: Optional["MLBox"]
) -> tuple[float, float]:
    """
    CSS-like relative offset:
    left  -> +X
    right -> -X
    top   -> +Y
    bottom-> -Y
    """
    if style.position != "relative":
        return 0.0, 0.0

    ref_w = parent_box.w if parent_box else None
    ref_h = parent_box.h if parent_box else None
    unit_scale = style.unit_scale

    dx = (_len_to_m(style.left, ref_w, unit_scale) or 0.0) - (
        _len_to_m(style.right, ref_w, unit_scale) or 0.0
    )
    dy = (_len_to_m(style.top, ref_h, unit_scale) or 0.0) - (
        _len_to_m(style.bottom, ref_h, unit_scale) or 0.0
    )
    return dx, dy


def _has_inline_box_style(style: "MLStyle") -> bool:
    """
    Returns True if the node must be treated as an atomic inline box
    instead of a transparent text wrapper.
    """
    default = MLStyle()

    for field in fields(MLStyle):
        name = field.name

        if name in _TEXT_ONLY_FIELDS:
            continue

        if getattr(style, name) != getattr(default, name):
            return True

    return False


def _is_transparent_inline_wrapper(node: "ml", style: "MLStyle") -> bool:
    """
    A transparent wrapper does not create its own visible/boxy inline object.
    Its children are merged directly into the parent flow.
    """
    if node.kind not in {"block", "inline"}:
        return False
    return not _has_inline_box_style(style)


def _effective_wrap_mode(style: "MLStyle") -> str:
    """Resolve the effective wrapping mode."""
    if style.white_space in {"nowrap"}:
        return "none"

    if style.white_space == "pre":
        # Preserve explicit line breaks, but do not wrap by width.
        return "none"

    if style.wrap_mode is not None:
        return style.wrap_mode

    return "word"


def _normalize_text_source(text: str, white_space: str) -> str:
    """Apply CSS-like whitespace normalization."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if white_space in {"normal", "nowrap"}:
        # Collapse spaces/tabs, but keep explicit line breaks.
        return re.sub(r"[^\S\n]+", " ", text)

    if white_space in {"pre", "pre-wrap"}:
        # Preserve line breaks and spaces.
        return text

    return text


def _iter_text_tokens(text: str, wrap_mode: str) -> Iterator[str]:
    """
    Tokenize text for wrapping.
    - word: words + spaces + newlines
    - character/anywhere: character-level tokenization, newlines preserved
    """
    if wrap_mode in {"character", "anywhere"}:
        yield from text
        return

    # word mode
    yield from re.findall(r"\n|[^\S\n]+|[^\s]+", text)


def _text_stroke_offsets(
    width_m: float,
    samples: int,
) -> list[tuple[float, float]]:
    """
    Offsets around the glyph used to simulate an outline.
    8 directions is a decent cheap default.
    """
    if width_m <= 1e-9:
        return []

    radius = width_m
    return [
        (
            math.cos(i / samples * math.tau) * radius,
            math.sin(i / samples * math.tau) * radius,
        )
        for i in range(samples)
    ]


def _make_text_object(
    text: str, style: "MLStyle", layer: mat.Layer | None, extrude_amount: float
) -> Text:
    """Build a configured Text object."""
    unit_scale = _as_float(style.unit_scale, _as_float(root_style.unit_scale, 1.0))
    font_size = _as_float(style.font_size, _as_float(root_style.font_size, 12.0))

    obj = Text(
        text=t(
            text=text,
            mat=layer,
            bold=style.font_weight == "bold",
            italic=style.font_style == "italic",
        ),
        size=font_size * unit_scale,
        loc=Pos(Z=-extrude_amount),
    )
    obj.spacing_character = 1.0 + _as_float(style.letter_spacing, 0.0)
    obj.spacing_word = 1.0 + _as_float(style.word_spacing, 0.0)
    obj.align("LEFT", "TOP")
    obj.extrude(abs(extrude_amount))
    obj.fill_mode = FillMode.FRONT if extrude_amount >= 0.0 else FillMode.BACK
    return obj


def _distribute_free_space(free: float, count: int, mode: str) -> tuple[float, float]:
    """Return the start offset and gap for a distribution mode."""
    if count <= 0:
        return 0.0, 0.0

    match mode:
        case "flex-start":
            return 0.0, 0.0
        case "flex-end":
            return free, 0.0
        case "center":
            return free / 2.0, 0.0
        case "space-between":
            return (0.0, 0.0) if count == 1 else (0.0, free / (count - 1))
        case "space-around":
            gap = free / count
            return gap / 2.0, gap
        case "space-evenly":
            gap = free / (count + 1)
            return gap, gap
        case _:
            return 0.0, 0.0


def _resolve_boolean_mode(style: Optional["MLStyle"]) -> Mode:
    if style is not None and style.mode == "add":
        return Mode.ADD
    return Mode.JOIN


def _is_extrude_mode(style: Optional["MLStyle"]) -> bool:
    return style is not None and style.mode == "extrude"


def is_percent(v):
    return isinstance(v, str) and v.strip().endswith("%")


def _freeze_for_hash(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Locations):
        return _freeze_for_hash(value.local_locations)
    if isinstance(value, TransformExpr):
        return hash(value)
    if isinstance(value, Part):
        return value.hash(use_materials=True)
    if isinstance(value, mat.Layer):
        return value.hash
    if isinstance(value, BuildCallback):
        return value.hash
    if callable(value):
        return _hash_callable(value)
    if isinstance(value, Enum):
        return (value.__class__.__qualname__, value.value)
    if is_dataclass(value) and not isinstance(value, ml):
        return (
            value.__class__.__qualname__,
            tuple(
                (f.name, _freeze_for_hash(getattr(value, f.name)))
                for f in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        return (
            "dict",
            tuple(
                sorted(
                    (_freeze_for_hash(k), _freeze_for_hash(v)) for k, v in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            value.__class__.__qualname__,
            tuple(_freeze_for_hash(v) for v in value),
        )
    if isinstance(value, set):
        return ("set", tuple(sorted(_freeze_for_hash(v) for v in value)))
    return (value.__class__.__qualname__, repr(value))


def _style_hash(style: "MLStyle") -> tuple[Any, ...]:
    return tuple(
        (f.name, _freeze_for_hash(getattr(style, f.name)))
        for f in fields(MLStyle)
        if f.name in _RENDER_STYLE_FIELDS
    )


def _attrs_hash(attrs: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(sorted((k, _freeze_for_hash(v)) for k, v in attrs.items()))


def _hash_callable(fn: Callable[..., Any]) -> tuple[Any, ...]:
    bound_self = None
    if inspect.ismethod(fn):
        bound_self = fn.__self__
        fn = fn.__func__

    code: Optional[types.CodeType] = getattr(fn, "__code__", None)
    closure: tuple[types.CellType, ...] = getattr(fn, "__closure__", None) or ()

    return (
        "callable",
        id(bound_self) if bound_self is not None else None,
        getattr(fn, "__module__", None),
        getattr(fn, "__qualname__", getattr(fn, "__name__", None)),
        code.co_code if code else None,
        code.co_consts if code else None,
        code.co_names if code else None,
        getattr(fn, "__defaults__", None),
        getattr(fn, "__kwdefaults__", None),
        tuple(_freeze_for_hash(cell.cell_contents) for cell in closure),
    )


@dataclass
class StyleResolveContext:
    node: "ml"
    field: str
    index: int = 0


_style_resolve_ctx: ContextVar[Optional[StyleResolveContext]] = ContextVar(
    "_style_resolve_ctx", default=None
)


@contextmanager
def style_resolve_context(node: "ml", field_name: str):
    """Context manager setting active node and style field during property evaluation."""
    new_context = StyleResolveContext(node=node, field=field_name)
    token = _style_resolve_ctx.set(new_context)
    try:
        yield
    finally:
        _style_resolve_ctx.reset(token)


_dof_ctx: ContextVar[Optional[dict[tuple[int, str, int], float]]] = ContextVar(
    "_dof_ctx", default=None
)


@contextmanager
def dof_context():
    """Provides a fresh scope for DOF parameter caching."""
    token_cache = _dof_ctx.set({})
    try:
        yield
    finally:
        _dof_ctx.reset(token_cache)


BuildCallbackType: TypeAlias = Callable[[BuildPart], Any] | Callable[[], Any]


def points_to_curve(points: list[tuple[float, float]]) -> Curve:
    with BuildCurve() as bc:
        Polyline(*[(p[0], p[1], 0.0) for p in points])
        return bc.curve


class RlRuntimeContext(rl.RuntimeContext):
    def __init__(self):
        super().__init__(None)
        self.force_soft_rules = True

    @override
    def get_global_transform(self, target: rl) -> Transform:
        if isinstance(target.part, ml) and target.part.is_building:
            return target.part.eval_box.transform
        return super().get_global_transform(target)

    @override
    def get_part_data(
        self, target: rl, target_shell_selector: Optional[str] = None
    ) -> rl.PartData:
        if isinstance(target.part, ml) and target.part.is_building:
            eval_box = target.part.eval_box
            shell_type = target_shell_selector or self._get_default_shell_type(target)
            key = (
                target.part,
                f"{shell_type}_{self._calc_eval_box_hash(eval_box, shell_type)}",
            )

            if key not in self.part_data:
                if shell_type == "box":
                    part_like = eval_box.part
                elif shell_type == "curve":
                    part_like = points_to_curve(eval_box.custom_data.curve_points)
                else:
                    raise RuntimeError(f"Unknown shell type: {shell_type}")
                self.part_data[key] = rl.PartData(part_like=part_like)
            return self.part_data[key]
        return super().get_part_data(target)

    def _get_default_shell_type(self, target: rl) -> str:
        rule = self._current_rule
        if (
            isinstance(rule, rl.CollisionRule)
            and rule.mode == rl.CollisionRule.Mode.INSIDE
            and rule.target is target
        ):
            return "curve"
        return "box"

    def _calc_eval_box_hash(
        self, box: BoxSetPart.Box["EvaluationNodeData"], shell_type: str
    ) -> int:
        if shell_type == "box":
            return hash((box.size.x, box.size.y, box.size.z))
        if shell_type == "curve":
            return hash(tuple(tuple(p) for p in box.custom_data.curve_points))
        raise RuntimeError(f"Unknown shell type: {shell_type}")


@dataclass
class MLBox:
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


@dataclass
class NodeGeneratorState:
    phase: Literal["grow", "bisect"] = "grow"
    lo: int = 0
    hi: int = 4
    n: int = 4
    done: bool = False
    initialized: bool = False
    cache: dict[int, Union["ml", list["ml"], None]] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildCallback:
    fn: BuildCallbackType
    pure: bool = False
    name: Optional[str] = None

    @property
    def hash(self) -> int:
        return hash(_hash_callable(self.fn), self.name)


@dataclass(slots=True)
class MLCacheEntry:
    part: Part
    subtract_parts: list[Part] = field(default_factory=list)


@dataclass(slots=True)
class MLBuildInfo:
    box: "MLBox" = field(default_factory=MLBox)
    eval_box: Optional[BoxSetPart.Box["EvaluationNodeData"]] = None
    text_measure_cache: dict[tuple[Any, ...], tuple[float, float]] = field(
        default_factory=dict
    )
    flow_lines: list["_FlowLine"] = field(default_factory=list)
    abs_children: list["ml"] = field(default_factory=list)
    cache_hash: int | None = None
    resolved_style: Optional["MLStyle"] = None
    layout_overflow: bool = False
    generated_children: list["ml"] = field(default_factory=list)
    generator_state: Optional[NodeGeneratorState] = None


@dataclass(slots=True)
class MLBuildContext:
    part_cache: dict[int, MLCacheEntry] = field(default_factory=dict)
    node_data: dict["ml", MLBuildInfo] = field(default_factory=dict)
    evaluate: bool = False
    eval_transform: Optional[Transform] = None
    rl_nodes: list["rl"] = field(default_factory=list)
    root_bp: Optional[BuildPart] = None


_ml_build_ctx: ContextVar[Optional[MLBuildContext]] = ContextVar(
    "_ml_build_ctx",
    default=None,
)


@contextmanager
def ml_build_context(ctx: MLBuildContext):
    """Set the active ML build context for the current build scope."""
    token = _ml_build_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ml_build_ctx.reset(token)


@dataclass(slots=True)
class EvaluationNodeData:
    node: "ml"
    curve_points: list[tuple[float, float]] = field(default_factory=list)
    offset: Location = field(default_factory=Location)


MaterialLayer = mat.Layer


@dataclass
class BorderNode:
    node: "ml"
    side: Optional[Literal["left", "right", "top", "bottom"]] = None
    selector: Optional[Callable[[], AbstractCurve]] = None
    subtract_parts_passthrough: bool = True
    layout_solver: SolverLike = field(default_factory=Solver())
    layout_objective: Callable = field(default_factory=lambda: None)
    evaluate: bool = False


@dataclass
class MLStyle:
    # Name
    name: Optional[str] = None
    tag: Optional[str | Iterable[str]] = None  # inherited by children
    root_tag: Optional[str | Iterable[str]] = None  # NOT inherited by children

    # Size and units.
    unit_scale: Optional[float] = None
    width: Optional[Length] = None
    height: Optional[Length] = None
    min_width: Optional[Length] = None
    min_height: Optional[Length] = None
    max_width: Optional[Length] = None
    max_height: Optional[Length] = None
    aspect_ratio: float | None = None
    top_scale: float = 1.0
    right_scale: float = 1.0
    bottom_scale: float = 1.0
    left_scale: float = 1.0

    # Positioning.
    position: Literal["relative", "absolute"] = "relative"
    top: Optional[Length] = None
    right: Optional[Length] = None
    bottom: Optional[Length] = None
    left: Optional[Length] = None
    x_offset: Optional[Length] = None
    y_offset: Optional[Length] = None
    z_offset: Optional[Length] = None
    pivot_x: Optional[Length] = None
    pivot_y: Optional[Length] = None
    pivot_z: Optional[Length] = None
    z_index: float = 0.0
    anchor_x: float = 0.5
    anchor_y: float = 0.5
    align: Optional[Literal["left", "center", "right", "justify"]] = None
    align_y: Literal["start", "center", "end"] = "start"
    locations: Optional[Locations] = None
    loc_ctx_passthrough: bool = False

    # 3d operations.
    extrude: float = 0.0
    extrude_transform: Optional[TransformExpr] = None
    extrude_delete_source_faces: bool = True
    extrude_prop_edit: Optional[ProportionalEdit] = None
    extrude_subtract_part_height_k: float = 2.0
    text_extrude: Optional[float] = None
    text_extrude_delete_source_faces: bool = True
    transform: Optional[TransformExpr] = None
    adapt_transform: Optional[bool] = False
    apply_transform_for_subtract_parts: bool = True
    mode: Optional[Literal["add", "join", "extrude"]] = None
    dissolve: Optional[float] = None
    bevel: list[tuple[Callable[[bool], ShapeList], float, int]] = field(
        default_factory=list
    )
    bend_angle: float = 0.0
    bend_direction: Literal["horizontal", "vertical"] = "vertical"
    bend_segments: int = 0
    subtract: bool = False
    background_cuts: Optional[int] = None
    background_on_build: list[BuildCallback] = field(default_factory=list)
    background_from_curve: Optional[CurveLike] = None

    # Display / flex.
    display: Literal["block", "flex", "none"] = "block"
    overflow: Literal["visible", "hidden", "hidden-border"] = "visible"
    flex_direction: Literal["row", "column"] = "row"
    flex_wrap: Literal["nowrap", "wrap"] = "nowrap"
    flex_grow: float = 0.0
    flex_shrink: float = 1.0
    justify_content: Literal[
        "flex-start",
        "flex-end",
        "center",
        "space-between",
        "space-around",
        "space-evenly",
    ] = "flex-start"
    align_items: Literal["flex-start", "flex-end", "center", "baseline", "stretch"] = (
        "stretch"
    )
    align_content: Literal[
        "flex-start", "flex-end", "center", "space-between", "space-around", "stretch"
    ] = "stretch"
    gap: Optional[Length] = 0

    # Box model.
    padding: Optional[Length] = 0
    padding_tb: Optional[Length] = None
    padding_lr: Optional[Length] = None
    padding_top: Optional[Length] = None
    padding_right: Optional[Length] = None
    padding_bottom: Optional[Length] = None
    padding_left: Optional[Length] = None

    margin: Optional[Length] = 0
    margin_tb: Optional[Length] = None
    margin_lr: Optional[Length] = None
    margin_top: Optional[Length] = None
    margin_right: Optional[Length] = None
    margin_bottom: Optional[Length] = None
    margin_left: Optional[Length] = None

    border_in_measure: bool = True
    border_width: Optional[Length] = 0
    border_offset: Optional[Length] = None
    border_z_index: Optional[float] = None
    border_style: Literal["solid", "dashed", "dotted", "double", "none"] = "solid"
    border_step_scale: float = 1.0
    border_dash_length: Optional[Length] = None
    border_extrude: Optional[float] = None
    border_extrude_transform: Optional[TransformExpr] = None
    border_extrude_delete_source_faces: bool = True
    border_extrude_prop_edit: Optional[ProportionalEdit] = None
    border_radius: Optional[Length] = 0
    border_radius_left: Optional[Length] = None
    border_radius_right: Optional[Length] = None
    border_radius_top: Optional[Length] = None
    border_radius_bottom: Optional[Length] = None
    border_around_background: bool = False
    border_nodes: list[BorderNode] = field(default_factory=list)

    border_radius_tl: Optional[Length] = None
    border_radius_tr: Optional[Length] = None
    border_radius_bl: Optional[Length] = None
    border_radius_br: Optional[Length] = None
    border_radius_segments: int = 16
    border_mat: Optional[MaterialLayer] = None

    # Paint.
    mat: Optional[MaterialLayer] = None
    background_mat: Optional[MaterialLayer] = None
    background_opacity: Optional[float] = None
    opacity: Optional[float] = None

    # Text.
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    font_weight: Optional[Literal["bold"]] = None
    font_style: Optional[Literal["italic", "normal"]] = None
    text_align: Optional[Literal["left", "center", "right", "justify"]] = None
    line_height: Optional[float] = None
    letter_spacing: Optional[float] = None
    white_space: Literal["normal", "pre", "pre-wrap", "nowrap"] = "normal"
    wrap_mode: Optional[Literal["word", "character", "anywhere", "none"]] = None
    word_spacing: Optional[float] = 0.0

    # Text stroke / outline.
    text_stroke_width: Optional[Length] = None
    text_stroke_mat: Optional[MaterialLayer] = None
    text_stroke_opacity: Optional[float] = None
    text_stroke_extrude: Optional[float] = None
    text_stroke_samples: Optional[int] = None

    # Other
    evaluate: Optional[bool] = None
    show_eval_box: Optional[bool | MaterialLayer] = None

    def __post_init__(self):
        self._attached = False
        if "ml" in globals() and ml._ctx_stack:
            ml._ctx_stack[-1]._pending_styles.append(self)

    def __add__(self, other: "MLStyle") -> "MLStyle":
        self._attached = True
        other._attached = True
        changes = {}
        for f in fields(MLStyle):
            v1, v2 = getattr(self, f.name), getattr(other, f.name)
            if v2 != f.default:
                if f.name in ("tag", "root_tag"):
                    v1 = tag_to_list(v1)
                    v2 = tag_to_list(v2)
                changes[f.name] = (
                    v1 + v2 if isinstance(v1, list) and isinstance(v2, list) else v2
                )
        combined = replace(self, **changes)
        return combined

    @staticmethod
    def abs_size(
        width: Optional[float] = None, height: Optional[float] = None
    ) -> "MLStyle":
        """Sets width and height in absolute units."""
        return MLStyle(
            width=(lambda n: width / n.unit_scale) if width is not None else None,
            height=(lambda n: height / n.unit_scale) if height is not None else None,
        )

    @staticmethod
    def circle(radius: Length, mat: Optional[MaterialLayer] = None) -> "MLStyle":
        """
        Creates a circular shape.
        Sets width and height to 2x radius and applies full border radius.
        """

        def diameter(node: "ml"):
            r = node.resolve_value(radius, "circle_radius")
            return r * 2 if isinstance(r, (int, float)) else r

        return MLStyle(
            width=diameter, height=diameter, border_radius="50%", background_mat=mat
        )

    @staticmethod
    def square(size: Length, mat: Optional[MaterialLayer] = None) -> "MLStyle":
        """Sets equal width and height."""
        return MLStyle(width=size, height=size, background_mat=mat)

    @staticmethod
    def absolute_center(z_index: float = 0.0) -> "MLStyle":
        """
        Centers an element perfectly using absolute positioning.
        """
        return MLStyle(
            position="absolute",
            top="50%",
            left="50%",
            anchor_x=0.5,
            anchor_y=0.5,
            z_index=z_index,
        )

    @staticmethod
    def flex_center(
        direction: Literal["row", "column"] = "row", gap: Length = 0
    ) -> "MLStyle":
        """
        Quick setup for a centered flex container.
        Aligns children to the center on both axes.
        """
        return MLStyle(
            display="flex",
            flex_direction=direction,
            justify_content="center",
            align_items="center",
            gap=gap,
        )

    @staticmethod
    def align_center() -> "MLStyle":
        """
        Aligns children to the center on both axes.
        """
        return MLStyle(align="center", align_y="center")

    @staticmethod
    def full_size() -> "MLStyle":
        """Stretches the element to 100% of its parent's dimensions."""
        return MLStyle(width="100%", height="100%")

    @staticmethod
    def ghost() -> "MLStyle":
        """Makes the element invisible but keeps it in the layout."""
        return MLStyle(opacity=0.0)

    @staticmethod
    def overlay() -> "MLStyle":
        """Typical absolute overlay covering the entire parent area."""
        return MLStyle(position="absolute", top=0, left=0, width=100, height=100)

    @staticmethod
    def column(gap: Length = 0) -> "MLStyle":
        """Shortcut for a vertical flex layout."""
        return MLStyle(display="flex", flex_direction="column", gap=gap)

    @staticmethod
    def row(gap: Length = 0) -> "MLStyle":
        """Shortcut for a horizontal flex layout."""
        return MLStyle(display="flex", flex_direction="row", gap=gap)

    @staticmethod
    def prop_box_extrude(
        left=1.0, right=1.0, top=1.0, bottom=1.0, multiply: bool = False
    ) -> "MLStyle":
        return MLStyle(
            extrude_prop_edit=make_box_sides_edit(left, right, top, bottom, multiply)
        )

    @staticmethod
    def prop_box_border_extrude(
        left=1.0, right=1.0, top=1.0, bottom=1.0, multiply: bool = False
    ) -> "MLStyle":
        return MLStyle(
            border_extrude_prop_edit=make_box_sides_edit(
                left, right, top, bottom, multiply
            )
        )

    @staticmethod
    def extrude_delete_face(
        top=False,
        bottom=False,
        side_left=False,
        side_right=False,
        side_top=False,
        side_bottom=False,
    ) -> "MLStyle":
        sides: list[tuple[bool, Callable[[ShapeList], ShapeList]]] = [
            (top, lambda s: s.bottom()),
            (bottom, lambda s: s.top()),
            (side_left, lambda s: s.min_x()),
            (side_right, lambda s: s.max_x()),
            (side_top, lambda s: s.min_y()),
            (side_bottom, lambda s: s.max_y()),
        ]
        return MLStyle(
            background_on_build=[
                BuildCallback(lambda bp, s=side: delete(s(bp.faces())), pure=True)
                for val, side in sides
                if val
            ]
        )

    @staticmethod
    def bevel_box_top(
        all=0.0, left=0.0, right=0.0, top=0.0, bottom=0.0, lr=0.0, tb=0.0, segments=10
    ) -> "MLStyle":
        sides: list[tuple[float, Callable[[ShapeList], ShapeList]]] = [
            (all, lambda s: s),
            (left, lambda s: s.min_x()),
            (right, lambda s: s.max_x()),
            (top, lambda s: s.min_y()),
            (bottom, lambda s: s.max_y()),
            (lr, lambda s: s.min_x() + s.max_x()),
            (tb, lambda s: s.min_y() + s.max_y()),
        ]
        return MLStyle(
            bevel=[
                (
                    lambda neg, s=side: s(
                        faces().top().edges() if neg else faces().bottom().edges()
                    ),
                    val,
                    segments,
                )
                for val, side in sides
                if val != 0
            ]
        )

    @staticmethod
    def border_ml(
        *nodes: MlItem,
        side: Optional[Literal["left", "right", "top", "bottom"]] = None,
        selector: Optional[Callable[[], AbstractCurve]] = None,
        subtract_parts_passthrough: bool = True,
        layout_solver: SolverLike = Solver(),
        layout_objective: Callable = lambda: 0,
        evaluate: bool = False,
    ) -> "MLStyle":
        return MLStyle(
            border_nodes=[
                BorderNode(
                    ml(*nodes),
                    side,
                    selector,
                    subtract_parts_passthrough,
                    layout_solver=layout_solver,
                    layout_objective=layout_objective,
                    evaluate=evaluate,
                )
            ]
        )

    @staticmethod
    def border_extrude_loc(
        X: float = 0.0,
        Y: float = 0.0,
        Z: float = 0.0,
        transform: TransformExpr = Transform(),
    ) -> "MLStyle":
        return MLStyle(
            x_offset="-50%",
            y_offset="-50%",
            left=X,
            transform=Rot(X=90) * Pos(Y=Y, Z=-Z) * transform,
        )

    @classmethod
    def dof_abs_pos(
        cls,
        min_x: float = 0.0,
        max_x: float = 1.0,
        min_y: float = 0.0,
        max_y: float = 1.0,
        steps_x: int = 10,
        steps_y: int = 10,
        step_x: Optional[float] = None,
        step_y: Optional[float] = None,
    ):
        return MLStyle(
            position="absolute",
            left=ml.dof(min=min_x, max=max_x, steps=steps_x, step=step_x),
            top=ml.dof(min=min_y, max=max_y, steps=steps_y, step=step_y),
        )

    @classmethod
    def dof_abs_pos_p(
        cls,
        min_x: str = "0%",
        max_x: str = "100%",
        min_y: str = "0%",
        max_y: str = "100%",
        steps_x: int = 10,
        steps_y: int = 10,
        step_x: Optional[float] = None,
        step_y: Optional[float] = None,
    ):
        return MLStyle(
            position="absolute",
            left=ml.dof_p(min=min_x, max=max_x, steps=steps_x, step=step_x),
            top=ml.dof_p(min=min_y, max=max_y, steps=steps_y, step=step_y),
        )

    @classmethod
    def dof_size(
        cls,
        min_w: float = 0.0,
        max_w: float = 1.0,
        min_h: float = 0.0,
        max_h: float = 1.0,
        steps_w: int = 10,
        steps_h: int = 10,
        step_w: Optional[float] = None,
        step_h: Optional[float] = None,
    ):
        return MLStyle(
            width=ml.dof(min=min_w, max=max_w, steps=steps_w, step=step_w),
            height=ml.dof(min=min_h, max=max_h, steps=steps_h, step=step_h),
        )

    @classmethod
    def dof_size_p(
        cls,
        min_w: str = "0%",
        max_w: str = "100%",
        min_h: str = "0%",
        max_h: str = "100%",
        steps_w: int = 10,
        steps_h: int = 10,
        step_w: Optional[float] = None,
        step_h: Optional[float] = None,
    ):
        return MLStyle(
            width=ml.dof_p(min=min_w, max=max_w, steps=steps_w, step=step_w),
            height=ml.dof_p(min=min_h, max=max_h, steps=steps_h, step=step_h),
        )


STYLE_FIELD_NAMES = frozenset(field.name for field in fields(MLStyle))

_TEXT_ONLY_FIELDS = {
    "font_size",
    "font_family",
    "font_weight",
    "font_style",
    "text_align",
    "line_height",
    "letter_spacing",
    "word_spacing",
    "white_space",
    "wrap_mode",
    "text_extrude",
    "opacity",
    "mat",
    "text_stroke_width",
    "text_stroke_mat",
    "text_stroke_opacity",
    "text_stroke_extrude",
    "text_stroke_samples",
}

_INHERITED_FIELDS = {
    "font_size",
    "font_family",
    "font_weight",
    "font_style",
    "text_align",
    "line_height",
    "letter_spacing",
    "text_extrude",
    "opacity",
    "mat",
    "unit_scale",
    "text_stroke_width",
    "text_stroke_mat",
    "text_stroke_opacity",
    "text_stroke_extrude",
    "text_stroke_samples",
    "mode",
    "dissolve",
    "tag",
}

_RENDER_STYLE_FIELDS = {
    # Geometry
    "width",
    "height",
    "min_width",
    "min_height",
    "max_width",
    "max_height",
    "aspect_ratio",
    # Shape
    "padding",
    "padding_tb",
    "padding_lr",
    "padding_top",
    "padding_right",
    "padding_bottom",
    "padding_left",
    "border_width",
    "border_offset",
    "border_style",
    "border_step_scale",
    "border_dash_length",
    "border_radius",
    "border_radius_left",
    "border_radius_right",
    "border_radius_top",
    "border_radius_bottom",
    "border_radius_tl",
    "border_radius_tr",
    "border_radius_bl",
    "border_radius_br",
    "border_radius_segments",
    "border_around_background",
    # Materials
    "mat",
    "background_mat",
    "background_opacity",
    "opacity",
    "border_mat",
    # Text
    "font_size",
    "font_family",
    "font_weight",
    "font_style",
    "text_align",
    "line_height",
    "letter_spacing",
    "white_space",
    "wrap_mode",
    "word_spacing",
    "text_stroke_width",
    "text_stroke_mat",
    "text_stroke_opacity",
    "text_stroke_extrude",
    "text_stroke_samples",
    # 3D
    "extrude",
    "extrude_transform",
    "extrude_delete_source_faces",
    "extrude_prop_edit",
    "text_extrude",
    "text_extrude_delete_source_faces",
    "border_extrude",
    "border_extrude_transform",
    "border_extrude_delete_source_faces",
    "border_extrude_prop_edit",
    "border_objects",
    "mode",
    "dissolve",
    "bevel",
    "bend_angle",
    "bend_direction",
    "bend_segments",
    "background_cuts",
    # Rendering behaviour
    "overflow",
    "display",
    # Warp
    "top_scale",
    "right_scale",
    "bottom_scale",
    "left_scale",
}

root_style = MLStyle(
    unit_scale=1.0,
    mode="join",
    background_opacity=1.0,
    opacity=1.0,
    font_size=12.0,
    font_family="Arial",
    text_align="left",
    line_height=1.2,
    letter_spacing=0.0,
)


@dataclass
class _FlowFragment:
    """
    One inline-flow fragment.

    kind == "text"  -> text token that may wrap
    kind == "atom"  -> atomic inline box (image, span with background/padding,
                       custom component, etc.)
    """

    kind: Literal["text", "atom", "break"]

    text: str = ""
    style: "MLStyle | None" = None
    node: "ml | None" = None

    # Content size.
    w: float = 0.0
    h: float = 0.0
    line_h: float = 0.0

    y_offset: float = 0.0

    # Margins in inline flow.
    margin_top: float = 0.0
    margin_right: float = 0.0
    margin_bottom: float = 0.0
    margin_left: float = 0.0

    relative_x: float = 0.0
    relative_y: float = 0.0

    @property
    def outer_w(self) -> float:
        return self.w + self.margin_left + self.margin_right

    @property
    def outer_h(self) -> float:
        return self.line_h + self.margin_top + self.margin_bottom


@dataclass
class _FlowLine:
    items: list[_FlowFragment]
    width: float = 0.0
    height: float = 0.0


@dataclass
class _FlexItem:
    node: "ml"

    # Content size, without margins.
    main_size: float = 0.0
    cross_size: float = 0.0

    # Margins.
    margin_top: float = 0.0
    margin_right: float = 0.0
    margin_bottom: float = 0.0
    margin_left: float = 0.0

    rel_main: float = 0.0
    rel_cross: float = 0.0

    def outer_main(self, is_row: bool) -> float:
        if is_row:
            return self.main_size + self.margin_left + self.margin_right
        return self.main_size + self.margin_top + self.margin_bottom

    def outer_cross(self, is_row: bool) -> float:
        if is_row:
            return self.cross_size + self.margin_top + self.margin_bottom
        return self.cross_size + self.margin_left + self.margin_right


@dataclass
class _FlexLine:
    items: list[_FlexItem]
    main_size: float = 0.0
    cross_size: float = 0.0


class MLExtraParams(TypedDict, total=False):
    kind: str
    src: str
    points: Any
    radius: float
    segments: int
    part: Any


class ml:
    """Generic node for the mini HTML-like layout engine."""

    BuildContext = MLBuildContext

    _ctx_stack: ClassVar[list["ml"]] = []

    def __enter__(self):
        ml._ctx_stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        # styles
        for style in self._pending_styles:
            if not style._attached:
                self.style += style
                self.style._attached = True

        self._pending_styles.clear()

        # children
        for child in self._pending_children:
            if not child.parent:
                self.children.append(child)
                child.parent = self

        self._pending_children.clear()

        ml._ctx_stack.pop()

    def __init__(
        self,
        *args: MlItem,
        on_build: Optional[BuildCallback | BuildCallbackType] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an ml node.

        Positional arguments can be a mix of MLStyle objects and children.
        MLStyle objects are merged from left to right.
        """
        ml_items = list(_flatten_items(args))

        attrs = kwargs
        self.kind: Literal[
            "block",
            "inline",
            "generator",
            "new_line",
            "text",
            "img",
            "curve",
            "line",
            "circle",
            "part",
            "joint",
        ] = attrs.pop("kind", "block")
        self.style = sum(
            [item for item in ml_items if isinstance(item, MLStyle)], MLStyle()
        )
        self.style._attached = True
        self.attrs: dict[str, Any] = attrs

        self.parent: Optional[ml] = None
        self.children: list[ml] = [
            self._normalize_child(item)
            for item in ml_items
            if not isinstance(item, (MLStyle, rl.Rule, rl))
        ]
        for child in self.children:
            if child.parent:
                raise ValueError("Child already has a parent")
            child.parent = self
        self.rl_rules = [item for item in ml_items if isinstance(item, rl.Rule)]
        self.rl_elements = [item for item in ml_items if isinstance(item, rl)]

        self._on_build_callbacks: list[BuildCallback] = []
        self._pending_styles: list[MLStyle] = []
        self._pending_children: list[ml] = []

        if on_build is not None:
            if isinstance(on_build, BuildCallback):
                self._on_build_instance(on_build.fn, pure=on_build.pure)
            else:
                self._on_build_instance(on_build)

        if ml._ctx_stack:
            ml._ctx_stack[-1]._pending_children.append(self)

    @property
    def _build_ctx(self) -> MLBuildContext:
        ctx = _ml_build_ctx.get()
        if ctx is None:
            raise RuntimeError("ML build context is not active")
        return ctx

    @property
    def _build_info(self) -> MLBuildInfo:
        """Return build-time state for this node from the active context."""
        return self._build_ctx.node_data.setdefault(self, MLBuildInfo())

    @property
    def is_building(self) -> bool:
        return self in self._build_ctx.node_data

    @property
    def box(self) -> MLBox:
        return self._build_info.box

    @box.setter
    def box(self, value: MLBox) -> None:
        self._build_info.box = value

    @property
    def eval_box(self) -> BoxSetPart.Box["EvaluationNodeData"]:
        info = self._build_info
        assert info.eval_box is not None, "Eval box is not set"
        return info.eval_box

    @eval_box.setter
    def eval_box(self, value: BoxSetPart.Box["EvaluationNodeData"]) -> None:
        self._build_info.eval_box = value

    @property
    def _resolved_style(self) -> MLStyle:
        style = self._build_info.resolved_style
        assert style is not None, "Resolved style is not set"
        return style

    @_resolved_style.setter
    def _resolved_style(self, value: Optional[MLStyle]) -> None:
        self._build_info.resolved_style = value

    @property
    def _text_measure_cache(
        self,
    ) -> dict[tuple[Any, ...], tuple[float, float, float, float]]:
        return self._build_info.text_measure_cache

    @property
    def _flow_lines(self) -> list[_FlowLine]:
        return self._build_info.flow_lines

    @_flow_lines.setter
    def _flow_lines(self, value: list[_FlowLine]) -> None:
        self._build_info.flow_lines = value

    @property
    def _abs_children(self) -> list["ml"]:
        return self._build_info.abs_children

    @_abs_children.setter
    def _abs_children(self, value: list["ml"]) -> None:
        self._build_info.abs_children = value

    @property
    def _cache_hash(self) -> int | None:
        return self._build_info.cache_hash

    @_cache_hash.setter
    def _cache_hash(self, value: int | None) -> None:
        self._build_info.cache_hash = value

    @property
    def _layout_overflow(self) -> bool:
        return self._build_info.layout_overflow

    @_layout_overflow.setter
    def _layout_overflow(self, value: bool) -> None:
        self._build_info.layout_overflow = value

    @property
    def _generated_children(self) -> list["ml"]:
        return self._build_info.generated_children

    @_generated_children.setter
    def _generated_children(self, value: list["ml"]) -> None:
        self._build_info.generated_children = value

    @property
    def unit_scale(self) -> float:
        return (
            (
                self._build_info.resolved_style
                and self._build_info.resolved_style.unit_scale
            )
            or (self.parent and self.parent.unit_scale)
            or 1
        )

    @property
    def width(self) -> float:
        return self.box.w / self.unit_scale

    @property
    def height(self) -> float:
        return self.box.h / self.unit_scale

    @property
    def x(self) -> float:
        return self.box.x / self.unit_scale

    @property
    def y(self) -> float:
        return self.box.y / self.unit_scale

    @property
    def part(self):
        return self.build(mode=Mode.PRIVATE)

    def to_part(self, width: Optional[float] = None, height: Optional[float] = None):
        return self.build(mode=Mode.PRIVATE, width=width, height=height)

    def to_rl_node(self):
        """Builds a RuleBasedLayout from this ml node."""
        global_rl: list[rl] = []

        def walk(node: ml) -> rl:
            group = rl.group(
                [walk(child) for child in node._iter_children_expanded()],
                part=node,
                tag=tag_to_list(node._resolved_style.tag)
                + tag_to_list(node._resolved_style.root_tag),
            )
            if node.rl_rules:
                group = group | node.rl_rules
            global_rl.extend(node.rl_elements)
            return group

        return rl.group(walk(self), global_rl)

    @staticmethod
    def _normalize_child(child: Children):
        if isinstance(child, ml):
            return child
        if isinstance(child, str):
            return ml(kind="text", text=child)
        if isinstance(child, (Part, BuildPart, BaseCurve, BuildCurve, Joint)):
            return ml.from_part(child)
        raise ValueError(f"Unsupported child type: {type(child)}")

    @classmethod
    def b(cls, *children: MlItem, **kwargs: Any) -> "ml":
        return cls(MLStyle(font_weight="bold"), *children, kind="inline", **kwargs)

    @classmethod
    def i(cls, *children: MlItem, **kwargs: Any) -> "ml":
        return cls(MLStyle(font_style="italic"), *children, kind="inline", **kwargs)

    @classmethod
    def stack(cls, *children: MlItem, **kwargs: Any) -> "ml":
        return cls(
            MLStyle(display="flex", flex_direction="column"),
            *children,
            kind="block",
            **kwargs,
        )

    @classmethod
    def img(cls, src: str, **kwargs: Any) -> "ml":
        return cls(kind="img", src=src, **kwargs)

    @classmethod
    def line(cls, points: Any, **kwargs: Any) -> "ml":
        return cls(kind="line", points=points, **kwargs)

    @classmethod
    def curve(cls, points: Any, **kwargs: Any) -> "ml":
        return cls(kind="curve", points=points, **kwargs)

    @classmethod
    def circle(cls, radius: float = 1.0, segments: int = 24, **kwargs: Any) -> "ml":
        return cls(kind="circle", radius=radius, segments=segments, **kwargs)

    @classmethod
    def from_part(cls, part: PartLike, style=MLStyle(), **kwargs: Any) -> "ml":
        return cls(
            style,
            kind="part",
            part=extract_part(part, to_loc=Rot(X=180), ensure_copy=True),
            **kwargs,
        )

    @classmethod
    def joint(
        cls,
        name: str,
        X: Optional[Length] = None,
        Y: Optional[Length] = None,
        Z: Optional[float] = None,
        flip: bool = False,
        style=MLStyle(),
        **kwargs: Any,
    ) -> "ml":
        return cls(
            MLStyle(
                position="absolute",
                width=0,
                height=0,
                left=X,
                top=Y,
                z_offset=Z,
                transform=Rot(X=180) if flip else None,
            )
            + style,
            kind="joint",
            name=name,
            **kwargs,
        )

    @classmethod
    def new_line(cls, **kwargs: Any) -> "ml":
        return cls(kind="new_line", **kwargs)

    GenerateResult: TypeAlias = Union["ml", list["ml"], None]

    @classmethod
    def generate(
        cls,
        factory: Optional[Callable[["ml"], GenerateResult]],
        **kwargs: Any,
    ) -> "ml":
        return cls(kind="generator", generator=factory, **kwargs)

    @classmethod
    def generate_array(
        cls,
        factory: Optional[
            Callable[[int, "ml"], GenerateResult]
            | Callable[[int], GenerateResult]
            | Callable[[], GenerateResult]
        ],
        fill_mode: Literal["box", "line"] = "box",
        start: int = 4,
        **kwargs: Any,
    ) -> "ml":
        # Layout passes probe the count first by growth and then by bisection.
        # Retaining factory results by index makes that convergence deterministic
        # and avoids rebuilding the same component during every probe.
        def wrapped(node: "ml"):
            if not node._build_info.generator_state:
                node._build_info.generator_state = NodeGeneratorState(hi=start, n=start)
            state = node._build_info.generator_state

            parent = node.parent
            overflow = False
            if parent:
                if fill_mode == "box":
                    overflow = parent._layout_overflow
                elif fill_mode == "line":
                    overflow = len(parent._flow_lines) > 1

            # First pass: measure exactly `start` items.
            if state.initialized and not state.done:
                if state.phase == "grow":
                    if overflow:
                        state.hi = state.n
                        state.phase = "bisect"
                        state.n = max(1, (state.lo + state.hi) // 2)
                    else:
                        state.lo = state.n
                        state.n *= 2
                else:
                    if overflow:
                        state.hi = state.n
                    else:
                        state.lo = state.n

                    if state.lo + 1 >= state.hi:
                        state.done = True
                        state.n = state.lo
                        state.cache = {
                            k: v for k, v in state.cache.items() if k < state.n
                        }
                    else:
                        state.n = max(1, (state.lo + state.hi) // 2)
            state.initialized = True

            out: list[ml] = []
            for i in range(state.n):
                child = state.cache.get(i)
                if child is None and i not in state.cache:
                    factory_sig = inspect.signature(factory)
                    if len(factory_sig.parameters) == 0:
                        child = factory()
                    elif len(factory_sig.parameters) == 1:
                        child = factory(i)
                    else:
                        child = factory(i, node)
                    state.cache[i] = child
                out.extend(_flatten_items([child]))

            return out

        return cls.generate(wrapped, **kwargs)

    @classmethod
    def array(
        cls,
        node: Callable[[int], "ml"],
        side: Literal["left", "right", "top", "bottom"] = "right",
        count: int = 1,
        style=MLStyle(),
        **kwargs: Any,
    ) -> "ml":
        """
        Generates a sequence of nodes by calling the lambda with an index.
        """
        children = []
        if side in ("left", "top"):
            indices = list(range(count - 1, -1, -1))
        else:
            indices = list(range(count))
        for idx, i in enumerate(indices):
            children.append(node(i))
            if side in ("top", "bottom") and idx < count - 1:
                children.append(cls.new_line())
        return ml(*children, style, **kwargs)

    @classmethod
    def mirror(
        cls,
        node: Callable[[], "ml"],
        side: Literal["left", "right", "top", "bottom"] = "right",
        flip: bool = True,
        offset=0.0,
        style=MLStyle(),
        **kwargs: Any,
    ) -> "ml":
        """Creates a mirror."""

        def mirror_factory(i: int) -> "ml":
            instance = node()
            if i == 0 and offset:
                setattr(
                    instance.style,
                    f"margin_{side}",
                    (getattr(instance.style, f"margin_{side}") or 0.0) + offset,
                )
            if i == 1 and flip:
                instance.style.transform = (instance.style.transform or Transform()) * (
                    FlipX if side in ("left", "right") else FlipY
                )
            return instance

        return cls.array(node=mirror_factory, side=side, count=2, style=style, **kwargs)

    @classmethod
    def hole(
        cls,
        *children: MlItem,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        mat: Optional[mat.Layer] = None,
        cuts: int = 0,
        **kwargs: Any,
    ) -> "ml":
        return cls(
            MLStyle(
                width=width,
                height=height,
                extrude=-depth,
                background_mat=mat,
                background_cuts=cuts,
            ),
            MLStyle.extrude_delete_face(bottom=True),
            *children,
            kind="block",
            **kwargs,
        )

    @classmethod
    def dof_get(
        cls,
        min: float = 0.0,
        max: float = 1.0,
        steps: int = 10,
        step: Optional[float] = None,
    ) -> float:
        """
        Retrieves a Solver parameter using positional order within field resolution.
        """
        init_value = (min + max) / 2
        cache = _dof_ctx.get()
        if cache is None:
            return init_value

        ctx = _style_resolve_ctx.get()
        assert ctx is not None, "dof called outside of style resolve context"

        cache_key = (id(ctx.node), ctx.field, ctx.index)
        ctx.index += 1
        if cache_key not in cache:
            cache[cache_key] = solver().param(
                init_value, min=min, max=max, steps=steps, step=step
            )
        return cache[cache_key]

    @classmethod
    def dof(
        cls,
        min: float = 0.0,
        max: float = 1.0,
        steps: int = 10,
        step: Optional[float] = None,
    ):
        """Registers a Solver parameter."""
        return lambda: cls.dof_get(min=min, max=max, steps=steps, step=step)

    @classmethod
    def dof_p(
        cls,
        min: str = "0%",
        max: str = "100%",
        steps: int = 10,
        step: Optional[float] = None,
    ):
        """Registers a Solver percentage parameter."""
        return lambda: (
            f"{cls.dof_get(min=_percent_to_m(min), max=_percent_to_m(max), steps=steps, step=step) * 100}%"
        )

    def _on_build_instance(self, *args, pure: bool = False):
        if args and callable(args[0]):
            fn = args[0]
            self._on_build_callbacks.append(BuildCallback(fn=fn, pure=False))
            return fn

        def decorator(fn: BuildCallbackType):
            self._on_build_callbacks.append(BuildCallback(fn=fn, pure=pure))
            return fn

        return decorator

    @classmethod
    def _on_build_class(cls, *args, pure: bool = False, name: Optional[str] = None):
        if not cls._ctx_stack:
            raise RuntimeError("ml.on_build used outside of with ml()")
        if args and callable(args[0]):
            fn = args[0]
            cls._ctx_stack[-1]._on_build_callbacks.append(
                BuildCallback(fn=fn, pure=False)
            )
            return fn

        def decorator(fn: BuildCallbackType):
            cls._ctx_stack[-1]._on_build_callbacks.append(
                BuildCallback(fn=fn, pure=pure, name=name)
            )
            return fn

        return decorator

    on_build = DualMethod(
        _on_build_instance,
        _on_build_class,
    )

    def _apply_transform_to_part(self, part: Part, style: MLStyle):
        old_transform = part.transform
        part.transform = (style.transform or Transform()) * part.transform
        if style.adapt_transform:
            part.transform *= Rot(X=90) * Scale(style.unit_scale)

        def restore():
            part.transform = old_transform

        return restore

    def _refresh_generator_children(self) -> None:
        """Rebuild generator children in the active build context."""
        if self.kind != "generator":
            return

        factory: Callable[["ml"], ml.GenerateResult] = self.attrs.get("generator")
        produced = [factory(self)]

        for ch in self._generated_children:
            if ch.parent is self.parent:
                ch.parent = None

        new_children: list[ml] = []

        for ch in _flatten_items(produced):
            ch = self._normalize_child(ch)
            if ch.parent is not None and ch.parent != self.parent:
                raise ValueError("Child already has a parent")
            ch.parent = self.parent
            new_children.append(ch)

        self._generated_children = new_children

    def _iter_children_expanded(self) -> Iterable["ml"]:
        """
        Expand generator nodes inline.
        Generator children inherit styles from the generator node.
        """
        for child in self.children:
            if child.kind == "generator":
                yield from child._generated_children
            else:
                yield child

    def _compute_cache_hash(self) -> int | None:
        style = self._resolved_style
        for cb in self._on_build_callbacks + (
            style.background_on_build if style is not None else []
        ):
            if not cb.pure:
                return None

        child_hashes = []
        for child in self._iter_children_expanded():
            child_hash = child._compute_cache_hash()
            if child_hash is None:
                return None
            child_hashes.append(child_hash)

        self._cache_hash = hash(
            (
                self.kind,
                _style_hash(style) if style is not None else None,
                _attrs_hash(self.attrs),
                (
                    round(self.box.w, 6),
                    round(self.box.h, 6),
                ),
                tuple(cb.hash for cb in self._on_build_callbacks),
                tuple(child_hashes),
            )
        )
        return self._cache_hash

    def resolve_value(self, value: Any, field_name: str = "") -> Any:
        """Resolve callable style values within context."""
        for _ in range(16):
            if not callable(value) or isinstance(value, mat.Layer):
                return value
            sig = inspect.signature(value)
            with style_resolve_context(self, field_name):
                value = value(self) if len(sig.parameters) == 1 else value()
        raise RuntimeError("Too many nested callable style values")

    def _resolve_style(
        self,
        parent_style: MLStyle | None,
    ) -> MLStyle:
        style = copy(self.style)
        self._resolved_style = style

        # 1) First resolve callable values.
        for field in fields(MLStyle):
            name = field.name
            raw = getattr(style, name)
            raw = self.resolve_value(raw, name)
            setattr(style, name, raw)

        # 2) Then apply inheritance.
        if parent_style is not None:
            for name in _INHERITED_FIELDS:
                if name in ["opacity", "background_opacity", "mat", "tag"]:
                    continue
                if getattr(style, name) is None:
                    setattr(style, name, getattr(parent_style, name))

            style.opacity = _as_float(parent_style.opacity, 1.0) * _as_float(
                style.opacity, 1.0
            )
            style.background_opacity = (
                _as_float(style.background_opacity, 1.0) * style.opacity
            )
            style.mat = _combine_mat(parent_style.mat, style.mat)
            style.tag = tag_to_list(parent_style.tag) + tag_to_list(style.tag)

        # 3) Resolve "background from curve".
        if style.background_from_curve is not None:
            if (
                self._resolved_style
                and getattr(self._resolved_style, "_background_from_curve_orig", None)
                == style.background_from_curve
            ):
                curve = self._resolved_style.background_from_curve
            else:
                curve = extract_curve(style.background_from_curve)
            setattr(style, "_background_from_curve_orig", style.background_from_curve)
            style.background_from_curve = curve
            _, width, height = _curve_to_outer_points(curve)
            if style.width is None:
                style.width = width
            if style.height is None:
                style.height = height

        return style

    def _collect_flow_fragments(
        self,
    ) -> list[_FlowFragment]:
        """
        Collect inline-flow fragments from node.children.

        Transparent wrappers are flattened.
        Boxy wrappers become atomic inline items.
        """
        style = self._resolved_style
        parent_box = self.parent and self.parent.box
        if style.display == "none" or self.kind == "generator":
            return []

        rel_dx, rel_dy = _resolve_relative_offset(style, parent_box)

        if self.kind == "new_line":
            return [_FlowFragment(kind="break")]

        if self.kind == "text":
            text = str(self.attrs.get("text", ""))
            if not text:
                return []
            return [_FlowFragment(kind="text", text=text, style=style)]

        if self.kind in {"img", "curve", "line", "circle", "part"}:
            self._measure_node()

            _, margin = _set_box_model(parent_box, style)

            return [
                _FlowFragment(
                    kind="atom",
                    node=self,
                    style=style,
                    w=self.box.w,
                    line_h=self.box.h,
                    margin_top=margin[0],
                    margin_right=margin[1],
                    margin_bottom=margin[2],
                    margin_left=margin[3],
                    relative_x=rel_dx,
                    relative_y=rel_dy,
                )
            ]

        if _is_transparent_inline_wrapper(self, style):
            out: list[_FlowFragment] = []
            for child in self._iter_children_expanded():
                out.extend(child._collect_flow_fragments())
            return out

        # Atomic inline container: measure it as its own box.
        self._measure_node()

        _, margin = _set_box_model(parent_box, style)

        return [
            _FlowFragment(
                kind="atom",
                node=self,
                style=style,
                w=self.box.w,
                line_h=self.box.h,
                margin_top=margin[0],
                margin_right=margin[1],
                margin_bottom=margin[2],
                margin_left=margin[3],
                relative_x=rel_dx,
                relative_y=rel_dy,
            )
        ]

    def _layout_flow_fragments(
        self,
        fragments: list[_FlowFragment],
        style: MLStyle,
        max_w: float | None,
    ) -> list[_FlowLine]:
        """
        Lay out mixed text + atomic inline fragments into lines.
        """
        wrap_mode = _effective_wrap_mode(style)
        width_limited = max_w is not None and max_w > 0.0

        lines: list[_FlowLine] = []
        current = _FlowLine(items=[], width=0.0, height=0.0)

        def flush() -> None:
            nonlocal current
            lines.append(current)
            current = _FlowLine(items=[], width=0.0, height=0.0)

        def push_item(item: _FlowFragment) -> None:
            current.items.append(item)
            current.width += max(0.0, item.outer_w + item.relative_x)
            current.height = max(current.height, item.outer_h + item.relative_y)

        for frag in fragments:
            if frag.kind == "break":
                flush()
                continue

            if frag.kind == "atom":
                if (
                    width_limited
                    and current.items
                    and current.width + max(0.0, frag.outer_w + frag.relative_x)
                    > max_w + 1e-9
                ):
                    flush()
                push_item(frag)
                continue

            text = _normalize_text_source(
                frag.text, frag.style.white_space if frag.style else style.white_space
            )
            seg_style = frag.style or style

            for token in _iter_text_tokens(text, wrap_mode):
                if token == "\n":
                    flush()
                    continue

                token_w, token_h, token_line_h, token_y_offset = (
                    self._measure_text_plain(token, seg_style)
                )

                # Skip leading spaces in normal word-flow.
                if (
                    not current.items
                    and token.isspace()
                    and wrap_mode == "word"
                    and seg_style.white_space in {"normal", "nowrap"}
                ):
                    continue

                if width_limited and wrap_mode != "none":
                    if current.items and current.width + token_w > max_w:
                        # If it is just a wrapping space, drop it.
                        if (
                            token.isspace()
                            and wrap_mode == "word"
                            and seg_style.white_space in {"normal", "nowrap"}
                        ):
                            flush()
                            continue
                        flush()

                    # Break too-long token by characters.
                    if (
                        token_w > max_w
                        and not token.isspace()
                        and wrap_mode in {"word", "character", "anywhere"}
                    ):
                        for ch in token:
                            ch_w, ch_h, ch_line_h, ch_y_offset = (
                                self._measure_text_plain(ch, seg_style)
                            )
                            if (
                                width_limited
                                and current.items
                                and current.width + ch_w > max_w
                            ):
                                flush()
                            push_item(
                                _FlowFragment(
                                    kind="text",
                                    text=ch,
                                    style=seg_style,
                                    w=ch_w,
                                    h=ch_h,
                                    line_h=ch_line_h,
                                    y_offset=ch_y_offset,
                                )
                            )
                        continue

                push_item(
                    _FlowFragment(
                        kind="text",
                        text=token,
                        style=seg_style,
                        w=token_w,
                        h=token_h,
                        line_h=token_line_h,
                        y_offset=token_y_offset,
                    )
                )

        if current.items or not lines:
            lines.append(current)

        return lines

    def _measure_flow_lines(
        self, lines: list[_FlowLine], fallback_style: MLStyle
    ) -> tuple[float, float]:
        """Measure the final content box of the flow layout."""
        if not lines:
            unit_scale = _as_float(
                fallback_style.unit_scale, _as_float(root_style.unit_scale, 1.0)
            )
            font_size = _as_float(
                fallback_style.font_size, _as_float(root_style.font_size, 12.0)
            )
            line_height = _as_float(
                fallback_style.line_height, _as_float(root_style.line_height, 1.2)
            )
            return 0.0, font_size * unit_scale * line_height

        max_line_w = 0.0
        total_h = 0.0

        for line in lines:
            max_line_w = max(max_line_w, line.width)
            if line.height > 0.0:
                total_h += line.height
            else:
                unit_scale = _as_float(
                    fallback_style.unit_scale, _as_float(root_style.unit_scale, 1.0)
                )
                font_size = _as_float(
                    fallback_style.font_size, _as_float(root_style.font_size, 12.0)
                )
                line_height = _as_float(
                    fallback_style.line_height, _as_float(root_style.line_height, 1.2)
                )
                total_h += font_size * unit_scale * line_height

        return max_line_w, total_h

    def _measure_text_plain(
        self, text: str, style: MLStyle
    ) -> tuple[float, float, float, float]:
        """
        Calculates the advance width and line height of a text string within the Blender
        environment using a differential measurement technique to account for
        font side bearings.
        """
        size = _as_float(style.font_size, _as_float(root_style.font_size, 12.0))
        unit_scale = _as_float(style.unit_scale, _as_float(root_style.unit_scale, 1.0))
        line_h = (
            size
            * unit_scale
            * _as_float(style.line_height, _as_float(root_style.line_height, 1.2))
        )
        stroke_w = _len_to_m(style.text_stroke_width, None, unit_scale) or 0.0

        key = (
            text,
            size,
            unit_scale,
            style.font_weight,
            style.font_style,
            _as_float(style.letter_spacing, 0.0),
            _as_float(style.word_spacing, 0.0),
            stroke_w,
        )

        cached = self._text_measure_cache.get(key)
        if cached is not None:
            return cached

        marker = "."

        def create_probe(content: str):
            p = Text(
                text=t(
                    text=content,
                    bold=style.font_weight == "bold",
                    italic=style.font_style == "italic",
                ),
                size=size * unit_scale,
            )
            p.spacing_character = 1.0 + _as_float(style.letter_spacing, 0.0)
            p.spacing_word = 1.0 + _as_float(style.word_spacing, 0.0)
            p.align("LEFT", "TOP")
            return p

        probe_full = create_probe(text + marker)
        probe_marker = create_probe(marker)

        w_full, h = _bbox_wh(probe_full)
        w_marker, _ = _bbox_wh(probe_marker)

        w = max(0.0, w_full - w_marker)
        y_offset = -probe_full.bbox.max.y

        if not text or (text.isspace() and w <= 0):
            em = size * unit_scale
            space_count = sum(4 if ch == "\t" else 1 for ch in text)
            w = space_count * em * 0.33

        if stroke_w > 0.0:
            # Stroke expands on BOTH sides.
            w += stroke_w * 2.0
            h += stroke_w * 2.0
            line_h += stroke_w * 2.0

        self._text_measure_cache[key] = (w, h, line_h, y_offset)
        return w, h, line_h, y_offset

    def _measure_leaf(self) -> tuple[float, float]:
        style = self._resolved_style
        unit_scale = style.unit_scale

        if self.kind == "img":
            placeholder = style.font_size * unit_scale * 4
            return placeholder, placeholder

        if self.kind == "circle":
            radius = _len_to_m(self.attrs.get("radius", 1.0), None, unit_scale) or 0.0
            return radius * 2.0, radius * 2.0

        if self.kind in {"line", "curve"}:
            points = self.attrs.get("points", [])
            return _points_bbox(points, unit_scale)

        if self.kind == "part":
            part_obj: Part = self.attrs.get("part")
            if part_obj is not None:
                restore = self._apply_transform_to_part(part_obj, style)
                size = _bbox_wh(part_obj)
                restore()
                return size

        return 0.0, 0.0

    def _measure_node(self) -> MLBox:
        style = self._resolved_style
        parent_box = self.parent and self.parent.box
        parent_style = self.parent and self.parent._resolved_style

        if style.display == "none" or self.kind == "generator":
            self.box = MLBox()
            return self.box

        unit_scale = style.unit_scale
        padding, _ = _set_box_model(parent_box, style)
        padding_top, padding_right, padding_bottom, padding_left = padding

        parent_w = parent_box.w if parent_box else None
        parent_h = parent_box.h if parent_box else None

        border_w = _len_to_m(style.border_width, parent_w, unit_scale) or 0.0
        border_offset = (
            _len_to_m(
                style.border_offset,
                parent_box.w if parent_box else None,
                style.unit_scale,
            )
            or 0.0
        )
        measure_border = (
            max(border_w + border_offset, 0.0) if style.border_in_measure else 0.0
        )
        gap = _len_to_m(style.gap, parent_w, unit_scale) or 0.0

        w = _len_to_m(style.width, parent_w, unit_scale)
        h = _len_to_m(style.height, parent_h, unit_scale)

        # border eats space ONLY for percent-based sizing
        if style.border_in_measure:
            if style.width is not None and is_percent(style.width):
                w = max(0.0, w - 2.0 * measure_border)
            if style.height is not None and is_percent(style.height):
                h = max(0.0, h - 2.0 * measure_border)

        natural_w, natural_h = self._measure_leaf()

        is_flex_item = parent_style is not None and parent_style.display == "flex"

        available_w = (
            w
            if w is not None
            else (natural_w if is_flex_item or parent_w is None else parent_w)
        )

        available_h = (
            h
            if h is not None
            else (natural_h if is_flex_item or parent_h is None else parent_h)
        )

        inner_w = max(0.0, available_w - padding_left - padding_right)

        inner_h = max(0.0, available_h - padding_top - padding_bottom)

        self.box = MLBox(0.0, 0.0, inner_w, inner_h)

        # Generic inline flow: text + atomic inline items in one normal flow.
        if (
            self.kind not in {"text", "img", "curve", "line", "circle", "part"}
            and style.display != "none"
        ):
            if style.display == "flex":
                is_row = style.flex_direction == "row"
                main_limit = inner_w if is_row else inner_h
                cross_limit = inner_h if is_row else inner_w

                lines: list[_FlexLine] = []
                current = _FlexLine(items=[])

                def _min_main_for(c_style: MLStyle) -> float:
                    ref = parent_w if is_row else parent_h
                    min_len = c_style.min_width if is_row else c_style.min_height
                    return max(0.0, _len_to_m(min_len, ref, unit_scale) or 0.0)

                def _main_advance(item: _FlexItem) -> float:
                    return max(0.0, item.outer_main(is_row) + item.rel_main)

                def _recalc_line_main_size(line: _FlexLine) -> float:
                    return sum(_main_advance(item) for item in line.items) + gap * max(
                        0, len(line.items) - 1
                    )

                # 1) Collect items into lines.
                for child in self._iter_children_expanded():
                    c_style = child._resolved_style
                    if c_style.display == "none":
                        continue

                    child_box = child._measure_node()

                    if c_style.position == "absolute":
                        continue

                    cm = _box_to_4(c_style.margin, inner_w, unit_scale)

                    rel_dx, rel_dy = _resolve_relative_offset(
                        c_style,
                        MLBox(0.0, 0.0, inner_w, inner_h),
                    )

                    item = _FlexItem(
                        node=child,
                        main_size=child_box.w if is_row else child_box.h,
                        cross_size=child_box.h if is_row else child_box.w,
                        margin_top=cm[0],
                        margin_right=cm[1],
                        margin_bottom=cm[2],
                        margin_left=cm[3],
                        rel_main=rel_dx if is_row else rel_dy,
                        rel_cross=rel_dy if is_row else rel_dx,
                    )

                    advance = _main_advance(item)
                    projected_main = current.main_size
                    if current.items:
                        projected_main += gap
                    projected_main += advance

                    if (
                        style.flex_wrap == "wrap"
                        and current.items
                        and projected_main > main_limit
                    ):
                        lines.append(current)
                        current = _FlexLine(items=[])

                    if current.items:
                        current.main_size += gap

                    current.items.append(item)
                    current.main_size += advance
                    current.cross_size = max(
                        current.cross_size,
                        item.outer_cross(is_row) + max(item.rel_cross, 0),
                    )

                if current.items:
                    lines.append(current)

                # 2) Shrink lines that do not fit.
                for line in lines:
                    overflow = line.main_size - main_limit
                    if overflow <= 1e-9:
                        continue

                    shrinkables: list[dict[str, float | _FlexItem]] = []
                    for item in line.items:
                        c_style = item.node._resolved_style
                        shrink = max(0.0, _as_float(c_style.flex_shrink, 1.0))
                        if shrink <= 0.0:
                            continue

                        min_main = _min_main_for(c_style)
                        min_main = min(min_main, item.main_size)
                        free_shrink = max(0.0, item.main_size - min_main)
                        if free_shrink <= 1e-9:
                            continue

                        shrinkables.append(
                            {
                                "item": item,
                                "shrink": shrink,
                                "basis": max(item.main_size, 1e-9),
                                "min_main": min_main,
                                "free_shrink": free_shrink,
                            }
                        )

                    remaining = overflow
                    while remaining > 1e-9 and shrinkables:
                        total_weight = sum(
                            float(entry["shrink"]) * float(entry["basis"])
                            for entry in shrinkables
                        )
                        if total_weight <= 1e-9:
                            break

                        consumed = 0.0
                        next_round: list[dict[str, float | _FlexItem]] = []

                        for entry in shrinkables:
                            item = entry["item"]  # type: ignore[assignment]
                            shrink = float(entry["shrink"])
                            basis = float(entry["basis"])
                            min_main = float(entry["min_main"])

                            share = remaining * (shrink * basis) / total_weight
                            new_main = item.main_size - share

                            if new_main <= min_main + 1e-9:
                                consumed += item.main_size - min_main
                                item.main_size = min_main
                            else:
                                consumed += share
                                item.main_size = new_main
                                next_round.append(
                                    {
                                        "item": item,
                                        "shrink": shrink,
                                        "basis": max(item.main_size, 1e-9),
                                        "min_main": min_main,
                                        "free_shrink": max(
                                            0.0, item.main_size - min_main
                                        ),
                                    }
                                )

                        if consumed <= 1e-9:
                            break

                        remaining -= consumed
                        shrinkables = next_round

                    # Recompute line size after shrink.
                    line.main_size = _recalc_line_main_size(line)

                # Grow items if there is free space.
                free_space = main_limit - line.main_size

                if free_space > 1e-9:
                    growables: list[tuple[_FlexItem, float]] = []

                    total_grow = 0.0

                    for item in line.items:
                        c_style = item.node._resolved_style
                        grow = max(0.0, _as_float(c_style.flex_grow, 0.0))

                        if grow <= 0.0:
                            continue

                        growables.append((item, grow))
                        total_grow += grow

                    if total_grow > 1e-9:
                        for item, grow in growables:
                            delta = free_space * (grow / total_grow)
                            item.main_size += delta

                        line.main_size = _recalc_line_main_size(line)

                total_cross = sum(line.cross_size for line in lines) + gap * max(
                    0, len(lines) - 1
                )

                if w is None:
                    w = (
                        (inner_w + padding_left + padding_right)
                        if is_row
                        else (
                            max((line.main_size for line in lines), default=0.0)
                            + padding_left
                            + padding_right
                        )
                    )
                if h is None:
                    h = (
                        (total_cross + padding_top + padding_bottom)
                        if is_row
                        else (inner_h + padding_top + padding_bottom)
                    )

                # 3) Layout pass.
                use_align_content = style.flex_wrap == "wrap" and len(lines) > 1

                cross_cursor = 0.0
                free_cross = cross_limit - total_cross

                self._layout_overflow = free_cross < -1e-9

                if use_align_content:
                    start_cross, cross_gap = _distribute_free_space(
                        free_cross,
                        len(lines),
                        style.align_content,
                    )
                else:
                    start_cross, cross_gap = 0.0, 0.0

                cross_cursor += start_cross
                line_gap = gap if len(lines) > 1 else 0.0

                for line in lines:
                    line_cross_extent = inner_h if is_row else inner_w

                    free_main = main_limit - line.main_size
                    start_main, main_gap = _distribute_free_space(
                        free_main,
                        len(line.items),
                        style.justify_content,
                    )
                    main_cursor = start_main

                    for item in line.items:
                        child = item.node
                        c_style = child._resolved_style

                        cm = (
                            item.margin_top,
                            item.margin_right,
                            item.margin_bottom,
                            item.margin_left,
                        )

                        child_w = item.main_size if is_row else item.cross_size
                        child_h = item.cross_size if is_row else item.main_size

                        # Stretch on cross-axis.
                        if style.align_items == "stretch":
                            if is_row:
                                if c_style.height is None:
                                    child_h = max(
                                        0.0, line_cross_extent - cm[0] - cm[2]
                                    )
                            else:
                                if c_style.width is None:
                                    child_w = max(
                                        0.0, line_cross_extent - cm[3] - cm[1]
                                    )

                        child.box.w = child_w
                        child.box.h = child_h

                        if is_row:
                            if style.align_items == "center":
                                child_y = (
                                    cross_cursor + (line_cross_extent - child_h) / 2
                                )
                            elif style.align_items == "flex-end":
                                child_y = cross_cursor + (line_cross_extent - child_h)
                            else:
                                child_y = cross_cursor

                            child.box.x = main_cursor + cm[3]
                            child.box.y = child_y + cm[0]

                            advance = max(0.0, child_w + cm[1] + cm[3] + item.rel_main)
                            main_cursor += advance + gap + main_gap
                        else:
                            if style.align_items == "center":
                                child_x = (
                                    cross_cursor + (line_cross_extent - child_w) / 2
                                )
                            elif style.align_items == "flex-end":
                                child_x = cross_cursor + (line_cross_extent - child_w)
                            else:
                                child_x = cross_cursor

                            child.box.x = child_x + cm[3]
                            child.box.y = main_cursor + cm[0]

                            advance = max(0.0, child_h + cm[0] + cm[2] + item.rel_main)
                            main_cursor += advance + gap + main_gap

                        child.box.x += item.rel_main
                        child.box.y += item.rel_cross

                    cross_cursor += line.cross_size + line_gap + cross_gap

            else:
                # Mixed inline flow: text + arbitrary components.
                fragments: list[_FlowFragment] = []
                abs_children: list[ml] = []
                for child in self._iter_children_expanded():
                    c_style = child._resolved_style
                    if c_style.position == "absolute":
                        child._measure_node()
                        abs_children.append(child)
                        continue
                    fragments.extend(child._collect_flow_fragments())

                lines = self._layout_flow_fragments(fragments, style, inner_w)
                content_w, content_h = self._measure_flow_lines(lines, style)

                self._layout_overflow = content_h > inner_h + 1e-9

                # Inline boxes must have stable line-box height.
                if content_h <= 0:
                    content_h = (
                        _as_float(
                            style.font_size, _as_float(root_style.font_size, 12.0)
                        )
                        * style.unit_scale
                        * _as_float(
                            style.line_height, _as_float(root_style.line_height, 1.2)
                        )
                    )

                if w is None:
                    w = content_w + padding_left + padding_right

                if h is None:
                    h = content_h + padding_top + padding_bottom

                self._flow_lines = lines
                self._abs_children = abs_children

        if w is None and h is not None and style.aspect_ratio:
            w = h * style.aspect_ratio
        if h is None and w is not None and style.aspect_ratio:
            h = w / style.aspect_ratio

        is_text_like = self.kind in {"text", "inline", "block"}

        min_w = style.font_size * unit_scale if is_text_like else 0.0
        min_h = style.font_size * unit_scale if is_text_like else 0.0

        if w is None:
            w = max(
                natural_w + padding_left + padding_right,
                min_w,
            )

        if h is None:
            h = max(
                natural_h + padding_top + padding_bottom,
                min_h,
            )

        if style.border_in_measure:
            w += 2.0 * measure_border
            h += 2.0 * measure_border

        if style.min_width is not None:
            w = max(w, _len_to_m(style.min_width, parent_w, unit_scale) or 0.0)
        if style.max_width is not None:
            w = min(w, _len_to_m(style.max_width, parent_w, unit_scale) or w)

        if style.min_height is not None:
            h = max(h, _len_to_m(style.min_height, parent_h, unit_scale) or 0.0)
        if style.max_height is not None:
            h = min(h, _len_to_m(style.max_height, parent_h, unit_scale) or h)

        self.box = MLBox(0.0, 0.0, w, h)
        return self.box

    def _snapshot_layout(self) -> tuple:
        out: list[tuple[int, float, float, float, float]] = []

        def walk(node: "ml") -> None:
            out.append(
                hash(
                    (
                        _style_hash(node._resolved_style)
                        if node._resolved_style is not None
                        else None,
                        _attrs_hash(self.attrs),
                        round(node.box.x, 6),
                        round(node.box.y, 6),
                        round(node.box.w, 6),
                        round(node.box.h, 6),
                    )
                )
            )
            for child in node._iter_children_expanded():
                walk(child)

        walk(self)
        return tuple(out)

    def _prepare_layout(self, parent_style: MLStyle | None = None) -> None:
        """Prepare generator children and resolve all styles for this layout pass."""
        style = self._resolve_style(parent_style)

        for child in self.children:
            if child.kind == "generator":
                child._refresh_generator_children()
                child._resolve_style(style)

                for generated_child in child._generated_children:
                    generated_child._prepare_layout(style)
            else:
                child._prepare_layout(style)

    def _layout_until_stable(self, max_passes: int = 1024):
        """
        Re-run measurement passes until dynamic styles stop changing.

        Style callbacks may read boxes computed by the prior pass. Stabilizing
        the complete tree snapshot makes those dependencies settle before RBL
        consumes evaluation boxes; a cycle fails explicitly at the pass limit.
        """
        prev = None

        for i in range(max_passes):
            self._prepare_layout()
            self._measure_node()
            cur = self._snapshot_layout()

            if cur == prev:
                return

            prev = cur

        raise RuntimeError(
            "Dynamic layout did not stabilize. "
            "Most likely you created a cyclic dependency between sizes."
        )

    def _layout(
        self,
        solver: SolverLike,
        objective: Callable,
    ):
        """
        Calculates tree layout, running solver optimization loop.
        Does not emit final 3D geometry.

        Evaluation uses BoxSetPart boxes instead of final meshes so candidate
        solver passes can cheaply evaluate RBL rules and collision boundaries.
        Final mesh emission begins only after the solver has selected a layout.
        """
        context = RlRuntimeContext()
        root_rl = None
        with context:
            for s in solver:
                with dof_context():
                    self._layout_until_stable()
                self._evaluate()

                root_rl = self.to_rl_node()
                context.root = rl.group(root_rl, self._build_ctx.rl_nodes)
                active_bindings = root_rl._compile_rules()
                if s.is_init:
                    context.initialize_bindings(active_bindings)
                else:
                    context.evaluate_bindings(active_bindings, s)

                objective()
        self._build_ctx.rl_nodes.append(root_rl)

    def _emit_flow(
        self,
        z_offset: float,
        content_w: float,
        lines: list[_FlowLine],
        subtract_parts: list[Part],
    ) -> None:
        """
        Emit mixed text + inline atomic fragments.
        """
        style = self._resolved_style
        pen_y = 0.0
        is_last = len(lines) - 1
        align = style.align or style.text_align or "left"

        for line_index, line in enumerate(lines):
            line_w = line.width
            line_h = line.height

            if align == "center":
                pen_x = (content_w - line_w) / 2.0
                justify_extra = 0.0
            elif align == "right":
                pen_x = content_w - line_w
                justify_extra = 0.0
            elif align == "justify" and line_index != is_last:
                space_count = sum(
                    1
                    for item in line.items
                    if item.kind == "text" and item.text.isspace()
                )
                if space_count > 0:
                    pen_x = 0.0
                    justify_extra = max(0.0, content_w - line_w) / space_count
                else:
                    pen_x = 0.0
                    justify_extra = 0.0
            else:
                pen_x = 0.0
                justify_extra = 0.0

            for item in line.items:
                pen_x += item.relative_x
                if item.kind == "text":
                    with Locations(Pos(X=pen_x, Y=pen_y + item.relative_y)):
                        self._emit_text_run(item.text, item, z_offset, subtract_parts)

                    if justify_extra and item.text.isspace():
                        pen_x += justify_extra
                else:
                    child = item.node
                    if child is None:
                        continue

                    child_x = pen_x + item.margin_left
                    child_y = pen_y + item.margin_top + item.relative_y

                    with Locations(Pos(X=child_x, Y=child_y)):
                        child_subtract_parts: list[Part] = []
                        child._emit_node(
                            MLBox(0.0, 0.0, content_w, line_h),
                            z_offset,
                            child_subtract_parts,
                        )
                        subtract_parts.extend(child_subtract_parts)
                pen_x += item.outer_w

            pen_y += line_h

    def _emit_text_run(
        self,
        text: str,
        frag: _FlowFragment,
        z_offset: float,
        subtract_parts: list[Part],
    ) -> None:
        """
        Emit one text run with optional stroke and fill.
        Stroke is drawn first, fill is drawn on top.
        """
        parent_style = self._resolved_style
        style = frag.style or parent_style
        mode = _resolve_boolean_mode(style)
        z_epsilon = _get_z_epsilon(style)
        z_step = 0.0 if mode == Mode.ADD else z_epsilon

        fill_extrude = _as_float(style.text_extrude, 0.0) * 0.5 * style.unit_scale

        stroke_extrude = min(
            fill_extrude,
            (
                _as_float(
                    style.text_stroke_extrude,
                    _as_float(style.text_extrude, 0.0),
                )
                * 0.5
                * style.unit_scale
            ),
        )

        if fill_extrude < 0.0:
            z_offset = 0.0

        z = z_offset

        if _is_extrude_mode(style):
            fill_extrude += z / 2
            stroke_extrude += z / 2
            z = 0.0

        # Fill material.
        fill_layer = _combine_mat(parent_style.mat, style.mat)
        if fill_layer is not None:
            fill_layer = _with_alpha(
                fill_layer,
                style.opacity if style.opacity is not None else parent_style.opacity,
            )

        # Stroke material.
        stroke_width_m = (
            _len_to_m(style.text_stroke_width, None, style.unit_scale) or 0.0
        )
        stroke_layer = style.text_stroke_mat or style.mat or parent_style.mat
        has_stroke = stroke_width_m > 1e-9 and stroke_layer is not None

        fill_z = z + z_step if has_stroke else z
        stroke_z = z

        if self._is_eval:
            if text.isspace() or not text:
                return
            box_w = max(frag.w, 1e-6)
            box_h = max(frag.h, 1e-6)
            z_min = min(0.0, fill_extrude, stroke_extrude) * 2
            z_max = max(0.0, fill_extrude, stroke_extrude) * 2
            box_d = max(z_max - z_min, 1e-6)

            center_offset = Pos(
                X=box_w / 2.0, Y=box_h / 2.0 + frag.y_offset, Z=-(z_min + z_max) / 2.0
            )

            with Locations(center_offset):
                Box(
                    box_w,
                    box_h,
                    box_d,
                    custom_data=EvaluationNodeData(node=self, offset=center_offset),
                )
            return

        if has_stroke:
            stroke_layer = _with_alpha(
                stroke_layer,
                style.text_stroke_opacity
                if style.text_stroke_opacity is not None
                else (
                    style.opacity if style.opacity is not None else parent_style.opacity
                ),
            )

            stroke_obj = _make_text_object(text, style, stroke_layer, stroke_extrude)
            stroke_obj_src_transform = stroke_obj.transform
            samples = style.text_stroke_samples or 16
            for dx, dy in _text_stroke_offsets(stroke_width_m, samples=samples):
                stroke_obj.transform = stroke_obj_src_transform
                stroke_obj.loc *= Pos(
                    X=stroke_width_m + dx - stroke_obj.bbox.min.x,
                    Y=stroke_width_m + dy,
                    Z=-stroke_z,
                ) * Rot(X=180)
                if not stroke_obj.empty:
                    add(stroke_obj, mode=mode)

        fill_obj = _make_text_object(text, style, fill_layer, fill_extrude)
        fill_obj.loc *= Pos(
            X=stroke_width_m - fill_obj.bbox.min.x,
            Y=stroke_width_m,
            Z=-fill_z,
        ) * Rot(X=180)
        if not fill_obj.empty:
            fill_obj_part = fill_obj.part
            if fill_extrude < 0.0:
                cutout_obj = fill_obj
                cutout_obj.fill_mode = FillMode.BOTH
                cutout_obj.extrude(abs(fill_extrude) * 10)
                cutout_obj_part = cutout_obj.part
                cutout_obj_part.loc = (
                    BuildPart._get_context()._location_context[0] * cutout_obj_part.loc
                )
                _fix_subtract_part_size(cutout_obj_part, style)
                subtract_parts.append(cutout_obj_part)
            add(fill_obj_part, mode=mode)

    def _emit_polyline_face(
        self,
        pts: Iterable[tuple[float, float]],
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        extrude: float = 0.0,
        dissolve_angle: float = 2.0,
        mat: Optional[mat.Layer] = None,
        mode: Mode = Mode.PRIVATE,
        close: bool = True,
        pts2: Iterable[tuple[float, float]] = [],
        extrude_transform: Optional[TransformExpr] = None,
        extrude_delete_source: bool = True,
        extrude_prop_edit: Optional[ProportionalEdit] = None,
        bevel: list[tuple[Callable[[bool], ShapeList], float, int]] = [],
        cuts: int = 0,
    ):
        """Build a 2D face using Polyline only."""
        if len(pts) < 3:
            return None
        pts = [(x + px, y + py, 0) for px, py in pts]
        with BuildCurve() as bc:
            Polyline(*pts, close=close)
            if pts2:
                Polyline(*pts2, close=close)
        bc.fill()
        part = bc.part
        with BuildPart(part, mode=Mode.PRIVATE):
            if dissolve_angle != 0.0:
                dissolve(angle_limit=dissolve_angle)
            if cuts != 0:
                subdivide(cuts=cuts)
            if extrude != 0.0:
                extrude_modifier(
                    op=Pos(Z=-extrude) * (extrude_transform or Transform()),
                    prop_edit=extrude_prop_edit,
                    delete_source=extrude_delete_source,
                )
            for entities, radius, segments in bevel:
                bevel_modifier(entities(extrude < 0.0), radius, segments)
        if mat is not None:
            part.mat = mat
        part.loc = Pos(Z=-z)
        add(part, mode=mode)
        return part

    def _emit_stroke_band(
        self,
        outline: list[tuple[float, float]],
        x: float,
        y: float,
        z: float,
        extrude: float,
        dissolve: float,
        layer: Optional[mat.Layer],
        thickness: float,
        offset: float,
        mode: Mode = Mode.PRIVATE,
        extrude_transform: Optional[TransformExpr] = None,
        extrude_delete_source=True,
        extrude_prop_edit: Optional[ProportionalEdit] = None,
    ) -> None:
        outline = _simplify_collinear_closed(outline)
        if len(outline) < 3 or thickness <= 1e-9:
            return
        inner = _offset_closed_polyline(outline, -offset)
        outer = _offset_closed_polyline(outline, -offset - thickness)
        if len(outer) < 3 or len(inner) < 3:
            return
        return self._emit_polyline_face(
            inner,
            x,
            y,
            z,
            extrude,
            dissolve,
            layer,
            mode,
            close=True,
            pts2=outer,
            extrude_transform=extrude_transform,
            extrude_delete_source=extrude_delete_source,
            extrude_prop_edit=extrude_prop_edit,
        )

    def _emit_clip_part(
        self,
        outline: list[tuple[float, float]],
        offset: float,
        mode: Mode = Mode.PRIVATE,
        depth=100.0,
    ) -> None:
        outer = _offset_closed_polyline(outline, -offset)
        return self._emit_polyline_face(
            outer,
            0.0,
            0.0,
            z=-depth / 2,
            extrude=depth,
            dissolve_angle=0.0,
            mat=None,
            mode=mode,
            close=True,
            extrude_delete_source=False,
        )

    def _emit_open_stroke_band(
        self,
        points: list[tuple[float, float]],
        x: float,
        y: float,
        z: float,
        extrude: float,
        dissolve,
        layer: mat.Layer,
        thickness: float,
        offset: float,
        mode: Mode,
        extrude_transform: Optional[TransformExpr] = None,
        extrude_delete_source=True,
        extrude_prop_edit: Optional[ProportionalEdit] = None,
    ) -> None:
        """
        Stroke open polyline as a clean ribbon.
        """
        if len(points) < 2:
            return

        left: list[tuple[float, float]] = []
        right: list[tuple[float, float]] = []

        for i in range(len(points)):
            if i == 0:
                p0 = points[i]
                p1 = points[i + 1]
            elif i == len(points) - 1:
                p0 = points[i - 1]
                p1 = points[i]
            else:
                p0 = points[i - 1]
                p1 = points[i + 1]

            nx, ny = _line_normal(p0, p1)

            px, py = points[i]

            left.append((px - nx * offset, py - ny * offset))
            right.append(
                (px - nx * (thickness + offset), py - ny * (thickness + offset))
            )

        poly = left + list(reversed(right))

        self._emit_polyline_face(
            poly,
            x,
            y,
            z,
            extrude,
            dissolve,
            layer,
            mode,
            close=True,
            extrude_transform=extrude_transform,
            extrude_delete_source=extrude_delete_source,
            extrude_prop_edit=extrude_prop_edit,
        )

    def _emit_circle_piece(
        self,
        cx: float,
        cy: float,
        radius: float,
        x: float,
        y: float,
        z: float,
        extrude: float,
        dissolve: float,
        layer: mat.Layer,
        mode: Mode,
        segments: int = 12,
    ) -> None:
        """Emit a circular dot for dotted borders."""
        if radius <= 1e-9:
            return

        pts = [
            (
                cx + math.cos(i / segments * math.tau) * radius,
                cy + math.sin(i / segments * math.tau) * radius,
            )
            for i in range(segments)
        ]
        self._emit_polyline_face(
            pts, x, y, z, extrude, dissolve, layer, mode, close=True
        )

    def _emit_pattern_border_on_path(
        self,
        outline: list[tuple[float, float]],
        x: float,
        y: float,
        z: float,
        extrude: float,
        dissolve: float,
        layer: mat.Layer,
        border_style: str,
        border_w: float,
        offset: float,
        step_scale: float,
        dash_length: Optional[float],
        mode: Mode,
        extrude_transform: Optional[TransformExpr] = None,
        extrude_delete_source=True,
        extrude_prop_edit: Optional[ProportionalEdit] = None,
    ):
        outline = _simplify_collinear_closed(outline)

        if border_w <= 1e-9 or len(outline) < 2:
            return

        segs, total = _closed_loop_segments(outline)
        if total <= 1e-9:
            return

        if border_style == "dashed":
            dash_len = dash_length if dash_length is not None else (border_w * 3.0)
            gap_len = dash_len * step_scale
            cycle = dash_len + gap_len

            # Make an integer number of cycles around the loop.
            cycles = max(1, round(total / cycle))
            actual_cycle = total / cycles
            actual_dash = actual_cycle * (dash_len / cycle)
            actual_gap = actual_cycle - actual_dash

            # Center pattern so the seam lands in a gap, not in the middle of a dash.
            pos = actual_gap * 0.5
            for _ in range(cycles):
                a = pos
                b = pos + actual_dash
                p0, _ = _sample_closed_path(segs, total, a)
                p1, _ = _sample_closed_path(segs, total, b)
                self._emit_open_stroke_band(
                    [p0, p1],
                    x,
                    y,
                    z,
                    extrude,
                    dissolve,
                    layer,
                    border_w,
                    offset,
                    mode,
                    extrude_transform=extrude_transform,
                    extrude_delete_source=extrude_delete_source,
                    extrude_prop_edit=extrude_prop_edit,
                )
                pos += actual_cycle

        elif border_style == "dotted":
            radius = border_w * 0.5
            step = 2 * border_w * step_scale
            count = max(1, round(total / step))
            actual_step = total / count

            # Place dots at the centers of equal intervals.
            for i in range(count):
                dist = (i + 0.5) * actual_step
                (cx, cy), (tx, ty) = _sample_closed_path(segs, total, dist)
                nx, ny = -ty, tx
                cx_offset = cx - nx * (offset + radius)
                cy_offset = cy - ny * (offset + radius)
                self._emit_circle_piece(
                    cx_offset,
                    cy_offset,
                    radius,
                    x,
                    y,
                    z,
                    extrude,
                    dissolve,
                    layer,
                    mode,
                )

    def _emit_node(
        self, parent_box: MLBox | None, z_offset: float, subtract_parts: list[Part]
    ) -> None:
        if self.kind == "new_line":
            return

        is_root = parent_box is None
        style = self._resolved_style
        parent_style = self.parent and self.parent._resolved_style
        parent_mode = _resolve_boolean_mode(parent_style)
        mode = _resolve_boolean_mode(style)

        if style.display == "none" or self.kind == "generator":
            return

        w = self.box.w
        h = self.box.h
        padding, _ = _set_box_model(parent_box, style)
        padding_top, padding_right, padding_bottom, padding_left = padding
        extrude = style.extrude * style.unit_scale

        z_epsilon = _get_z_epsilon(style)
        z_step = 0.0 if mode == Mode.ADD else (z_epsilon if not is_root else 0.0)
        has_bg = style.background_mat is not None
        has_border = bool(
            style.border_width
            and style.border_width > 0
            and style.border_style != "none"
        )

        bg_z = 0.0
        flow_z = 0.0
        current_z = 0.0
        if has_border:
            current_z += z_step
        if has_bg:
            bg_z = current_z
            current_z += z_step
        flow_z = current_z
        abs_z = flow_z
        if self._flow_lines or self.children:
            abs_z = flow_z + z_step

        border_w = (
            _len_to_m(
                style.border_width,
                parent_box.w if parent_box else None,
                style.unit_scale,
            )
            or 0.0
        )
        border_offset = (
            _len_to_m(
                style.border_offset,
                parent_box.w if parent_box else None,
                style.unit_scale,
            )
            or 0.0
        )
        border_z_offset = _as_float(style.border_z_index) * z_epsilon
        border_extrude = (
            (style.border_extrude * style.unit_scale)
            if style.border_extrude is not None
            else extrude
        )
        measure_border = (
            max(border_w + border_offset, 0.0) if style.border_in_measure else 0.0
        )
        inner_w = max(0.0, w - 2.0 * measure_border)
        inner_h = max(0.0, h - 2.0 * measure_border)
        dissolve = _as_float(style.dissolve, 2.0)

        if extrude < 0:
            z_offset = 0.0

        x = 0.0
        y = 0.0
        z = z_offset + style.z_index * z_epsilon
        if style.position == "absolute" and parent_box is not None:
            if style.left is not None:
                x = _len_to_m(style.left, parent_box.w, style.unit_scale) or 0.0
            elif style.right is not None:
                x = (
                    parent_box.w
                    - w
                    - (_len_to_m(style.right, parent_box.w, style.unit_scale) or 0.0)
                )

            if style.top is not None:
                y = _len_to_m(style.top, parent_box.h, style.unit_scale) or 0.0
            elif style.bottom is not None:
                y = (
                    parent_box.h
                    - h
                    - (_len_to_m(style.bottom, parent_box.h, style.unit_scale) or 0.0)
                )

            x -= w * style.anchor_x
            y -= h * style.anchor_y

        if _is_extrude_mode(style):
            extrude += z
            border_extrude += z + border_z_offset
            z = 0.0
            border_z_offset = 0.0

        def build_children():
            if self.kind in {"block", "inline"}:
                if self._abs_children:
                    abs_parent_box = MLBox(
                        padding_left + measure_border,
                        padding_top + measure_border,
                        max(
                            0.0, w - padding_left - padding_right - 2.0 * measure_border
                        ),
                        max(
                            0.0, h - padding_top - padding_bottom - 2.0 * measure_border
                        ),
                    )

                    for index, child in enumerate(self._abs_children):
                        with Locations(Pos(X=abs_parent_box.x, Y=abs_parent_box.y)):
                            child_subtract_parts: list[Part] = []
                            child._emit_node(
                                abs_parent_box,
                                z_offset=abs_z,
                                subtract_parts=child_subtract_parts,
                            )
                            subtract_parts.extend(child_subtract_parts)

                if self._flow_lines:
                    flow_box_w = max(0.0, inner_w - padding_left - padding_right)
                    flow_box_h = max(0.0, inner_h - padding_top - padding_bottom)

                    _, content_h = self._measure_flow_lines(self._flow_lines, style)

                    offset_y = _resolve_flow_alignment_offset_y(
                        style,
                        flow_box_h,
                        content_h,
                    )

                    with Locations(
                        Pos(
                            X=padding_left + measure_border,
                            Y=padding_top + measure_border + offset_y,
                        )
                    ):
                        self._emit_flow(
                            flow_z, flow_box_w, self._flow_lines, subtract_parts
                        )
                    return

            child_parent_box = MLBox(
                measure_border,
                measure_border,
                inner_w,
                inner_h,
            )
            if self.children:
                content_x = padding_left + measure_border
                content_y = padding_top + measure_border

                for index, child in enumerate(self._iter_children_expanded()):
                    with Locations(
                        Pos(X=content_x + child.box.x, Y=content_y + child.box.y)
                    ):
                        child_subtract_parts: list[Part] = []
                        child._emit_node(
                            child_parent_box,
                            z_offset=flow_z,
                            subtract_parts=child_subtract_parts,
                        )
                        subtract_parts.extend(child_subtract_parts)

        def build_part(part_obj: Part):
            restore = self._apply_transform_to_part(part_obj, style)
            bb = part_obj.bbox
            part_obj.loc = (
                Pos(
                    X=-bb.min.x,
                    Y=-bb.min.y,
                    Z=0,
                )
                * part_obj.loc
            )
            add(part_obj, mode=mode)
            restore()

        def build_other_kinds():
            if self.kind == "img":
                src = self.attrs.get("src")
                img_layer = style.mat
                if img_layer is None and src is not None:
                    img_layer = mat.PBR(base_color=mat.Tex(path=src))

                if img_layer is not None:
                    img_layer = _with_alpha(img_layer, style.opacity)

                pts = _rounded_rect_points_4(w, h, (0.0, 0.0, 0.0, 0.0), 4)
                self._emit_polyline_face(
                    pts,
                    0,
                    0,
                    0,
                    extrude,
                    dissolve,
                    img_layer,
                    mode,
                    close=True,
                    extrude_transform=style.extrude_transform,
                    extrude_prop_edit=style.extrude_prop_edit,
                    extrude_delete_source=style.extrude_delete_source_faces,
                    bevel=style.bevel,
                )

            elif self.kind == "circle":
                radius = self.attrs.get("radius", 1.0)
                segments = int(self.attrs.get("segments", 24))
                r = _len_to_m(radius, None, style.unit_scale) or 0.0
                pts = [
                    (
                        math.cos(i / segments * math.tau) * r + r,
                        math.sin(i / segments * math.tau) * r + r,
                    )
                    for i in range(segments)
                ]
                layer = _with_alpha(
                    _combine_mat(style.mat, style.background_mat), style.opacity
                )
                self._emit_polyline_face(
                    pts,
                    0,
                    0,
                    0,
                    extrude,
                    dissolve,
                    layer,
                    mode,
                    close=True,
                    extrude_transform=style.extrude_transform,
                    extrude_prop_edit=style.extrude_prop_edit,
                    extrude_delete_source=style.extrude_delete_source_faces,
                    bevel=style.bevel,
                )

            elif self.kind == "line":
                points = self.attrs.get("points", [])
                if points:
                    pts2d = [(float(px), float(py)) for px, py, *_rest in points]
                    with BuildCurve() as bc:
                        Polyline(*[(px, py, 0) for px, py in pts2d], close=False)
                    part = bc.part
                    part.mat = _with_alpha(
                        _combine_mat(style.mat, style.background_mat), style.opacity
                    )
                    add(part, mode=mode)

            elif self.kind == "curve":
                points = self.attrs.get("points", [])
                if points:
                    pts3d = [(float(px), float(py), 0) for px, py, *_rest in points]
                    with BuildCurve() as bc:
                        Spline(*pts3d)
                    part = bc.part
                    part.mat = _with_alpha(
                        _combine_mat(style.mat, style.background_mat), style.opacity
                    )
                    add(part, mode=mode)

            elif self.kind == "part":
                part_obj: Optional[Part] = self.attrs.get("part")
                if part_obj is not None:
                    old_mat = part_obj.mat
                    if style.mat is not None:
                        part_obj.mat = _with_alpha(
                            _combine_mat(style.mat, style.background_mat), style.opacity
                        )
                    build_part(part_obj)
                    part_obj.mat = old_mat

        def calculate_outer_pts(offset_x=0.0, offset_y=0.0):
            curve = None
            if style.background_from_curve:
                curve = extract_curve(style.background_from_curve)
                outer_pts, _, _ = _curve_to_outer_points(curve, inner_w, inner_h)
            else:
                outer_radii = _resolve_corner_radii(style, inner_w, inner_h)
                outer_pts = _rounded_rect_points_4(
                    inner_w,
                    inner_h,
                    outer_radii,
                    style.border_radius_segments,
                )
            outer_pts = _warp_side_scales_points(
                outer_pts,
                inner_w,
                inner_h,
                top_scale=_as_float(style.top_scale, 1.0),
                right_scale=_as_float(style.right_scale, 1.0),
                bottom_scale=_as_float(style.bottom_scale, 1.0),
                left_scale=_as_float(style.left_scale, 1.0),
            )
            outer_pts = [
                (px + measure_border + offset_x, py + measure_border + offset_y)
                for px, py in outer_pts
            ]
            return outer_pts, curve

        def build_background(outer_pts: list[tuple[float, float]], bp: BuildPart):
            if not has_bg:
                return
            self._emit_polyline_face(
                outer_pts,
                0,
                0,
                bg_z,
                extrude,
                dissolve,
                _with_alpha(style.background_mat, style.background_opacity),
                mode,
                close=True,
                extrude_transform=style.extrude_transform,
                extrude_prop_edit=style.extrude_prop_edit,
                extrude_delete_source=style.extrude_delete_source_faces,
                bevel=style.bevel,
                cuts=style.background_cuts or 0,
            )
            add_tags([Object.TAG_ML_BACKGROUND])
            for cb in style.background_on_build:
                sig = inspect.signature(cb.fn)
                if len(sig.parameters) == 0:
                    cb.fn()
                else:
                    cb.fn(bp)

        def build_extrude_subtract_parts(outer_pts: list[tuple[float, float]]):
            # Negative extrusion is emitted as a deferred, slightly oversized
            # cutter. It must survive child emission so the parent can subtract
            # it after all geometry sharing this layout space exists.
            if extrude < 0.0:
                part = self._emit_clip_part(
                    outer_pts,
                    border_w + border_offset,
                    depth=2 * abs(extrude) * style.extrude_subtract_part_height_k,
                )
                _fix_subtract_part_size(part, style)
                subtract_parts.append(part)
            elif border_extrude < 0.0:
                depth = abs(border_extrude) * style.extrude_subtract_part_height_k
                part = self._emit_stroke_band(
                    outer_pts,
                    0,
                    0,
                    -depth,
                    depth * 2,
                    0.0,
                    None,
                    border_w,
                    border_offset,
                    extrude_delete_source=False,
                )
                _fix_subtract_part_size(part, style)
                subtract_parts.append(part)

        def apply_clip(outer_pts: list[tuple[float, float]]):
            if "hidden" in style.overflow:
                self._emit_clip_part(
                    outer_pts,
                    border_w + border_offset
                    if style.overflow == "hidden-border"
                    else 0.0,
                    Mode.INTERSECT_FAST,
                )

        def build_border(
            outer_pts: list[tuple[float, float]],
            source_curve: Optional[Curve],
            bp: BuildPart,
            bp_child: BuildPart,
        ):
            if not has_border and not style.border_nodes:
                return
            loops = [outer_pts]
            if style.background_mat is None and not style.border_around_background:
                with BuildPart(mode=Mode.PRIVATE) as bp_combined:
                    add(bp, mode=Mode.JOIN)
                    add(bp_child, mode=Mode.JOIN)
                    loops = _part_outline_loops_xy(bp_combined.part)
            if not loops:
                return

            if style.border_nodes:
                curves: list[AbstractCurve] = []
                with BuildPart(mode=Mode.PRIVATE):
                    for loop in loops:
                        loop = _simplify_collinear_closed(loop)
                        self._emit_polyline_face(
                            loop, z=border_z_offset + extrude, mode=Mode.JOIN
                        )
                    for border_node in style.border_nodes:
                        if border_node.selector:
                            curve = border_node.selector()
                            if isinstance(curve, Curve):
                                if (
                                    source_curve is None
                                    or curve.source_curve != source_curve.source_curve
                                ):
                                    raise ValueError(
                                        "Border node selector must return the same curve as the curve used to create the background (background_from_curve)"
                                    )
                                with BuildCurve() as bc:
                                    pts = [
                                        (outer_pts[i][0], outer_pts[i][1], 0.0)
                                        for i in curve.source_point_indices
                                    ]
                                    Polyline(*pts)
                                    curve = bc.curve
                        else:
                            axis = {
                                "left": -Axis.X,
                                "right": Axis.X,
                                "top": -Axis.Y,
                                "bottom": Axis.Y,
                            }[border_node.side or "left"]
                            curve = wires().group_by(axis)[-1][-1]
                        curves.append(curve)

                for curve, border_node in zip(curves, style.border_nodes):
                    node = border_node.node
                    orig_style = node.style
                    node.style += MLStyle(
                        width=curve.length() / style.unit_scale,
                        height=0,
                        locations=Locations(curve.location()),
                    )
                    border_subtract_parts: list[Part] = []
                    eval_box = self.eval_box
                    part = node.build(
                        mode=Mode.PRIVATE,
                        build_ctx=MLBuildContext(
                            node_data=self._build_ctx.node_data,
                            root_bp=self._build_ctx.root_bp,
                            eval_transform=eval_box.transform
                            * eval_box.custom_data.offset.inverse,
                            rl_nodes=self._build_ctx.rl_nodes,
                        ),
                        subtract_parts=border_subtract_parts,
                        layout_solver=border_node.layout_solver,
                        layout_objective=border_node.layout_objective,
                        evaluate=border_node.evaluate,
                    )
                    if border_node.subtract_parts_passthrough:
                        subtract_parts.extend(border_subtract_parts)
                    node.style = orig_style
                    add(
                        part,
                        mode=mode,
                        tag=tag_to_list(style.root_tag) + [Object.TAG_ML_BORDER],
                    )

            if not has_border:
                return
            border_mat = style.border_mat or style.mat
            if border_mat is not None:
                border_mat = _with_alpha(border_mat, style.opacity)

            with BuildPart(mode=mode):
                for loop in loops:
                    loop = _simplify_collinear_closed(loop)
                    if not loop:
                        continue
                    if style.border_style == "solid":
                        self._emit_stroke_band(
                            loop,
                            0,
                            0,
                            border_z_offset,
                            border_extrude,
                            dissolve,
                            border_mat,
                            border_w,
                            border_offset,
                            mode,
                            extrude_transform=style.border_extrude_transform,
                            extrude_delete_source=style.border_extrude_delete_source_faces,
                            extrude_prop_edit=style.border_extrude_prop_edit,
                        )
                    elif style.border_style in {"dashed", "dotted"}:
                        border_dash_length = _len_to_m(
                            style.border_dash_length,
                            parent_box.w if parent_box else None,
                            style.unit_scale,
                        )
                        self._emit_pattern_border_on_path(
                            loop,
                            0,
                            0,
                            border_z_offset,
                            border_extrude,
                            dissolve,
                            border_mat,
                            style.border_style,
                            border_w,
                            border_offset,
                            style.border_step_scale,
                            border_dash_length,
                            mode,
                            extrude_transform=style.border_extrude_transform,
                            extrude_delete_source=style.border_extrude_delete_source_faces,
                            extrude_prop_edit=style.border_extrude_prop_edit,
                        )
                    elif style.border_style == "double":
                        stripe = max(border_w / 3.0, 1e-6)
                        self._emit_stroke_band(
                            loop,
                            0,
                            0,
                            border_z_offset,
                            border_extrude,
                            dissolve,
                            border_mat,
                            stripe,
                            border_offset,
                            mode,
                            extrude_transform=style.border_extrude_transform,
                            extrude_delete_source=style.border_extrude_delete_source_faces,
                            extrude_prop_edit=style.border_extrude_prop_edit,
                        )
                        self._emit_stroke_band(
                            loop,
                            0,
                            0,
                            border_z_offset,
                            border_extrude,
                            dissolve,
                            border_mat,
                            stripe,
                            border_offset + 2 * stripe,
                            mode,
                            extrude_transform=style.border_extrude_transform,
                            extrude_delete_source=style.border_extrude_delete_source_faces,
                            extrude_prop_edit=style.border_extrude_prop_edit,
                        )
                add_tags(tag_to_list(style.root_tag) + [Object.TAG_ML_BORDER])

        def apply_3d_ops(bp: BuildPart):
            if style.bend_angle != 0.0:
                bend(
                    angle=style.bend_angle,
                    axis=Axis.X if style.bend_direction == "horizontal" else Axis.Y,
                    segments=style.bend_segments,
                    origin=Pos(X=w * 0.5),
                )
                transform(
                    op=Pos(
                        Z=-bp.part.bbox.max.z,
                    )
                )

        cached_part: Part | None = None
        assert self._build_ctx is not None
        if self._cache_hash is not None:
            entry = self._build_ctx.part_cache.get(self._cache_hash)
            if entry is not None:
                cached_part = entry.part.copy()
                subtract_parts.extend(entry.subtract_parts)

        parent_bp = BuildPart._get_context()

        with (
            Locations(Pos(X=x, Y=y, Z=-z)),
            BuildPart(
                part=Part.box_set_empty() if self._is_eval else cached_part,
                mode=Mode.JOIN,
                loc_ctx_passthrough=style.loc_ctx_passthrough,
            ) as bp,
            style.locations or Locations(),
        ):
            if self._is_eval:
                if style.evaluate != False:
                    if self.kind == "part":
                        part_obj: Part = self.attrs.get("part")
                        box_part = part_obj.get_bbox_set_part(
                            custom_data=EvaluationNodeData(node=self)
                        )
                        build_part(box_part)
                    else:
                        box_w = max(w, 1e-6)
                        box_h = max(h, 1e-6)
                        z_min = min(0.0, extrude, border_extrude)
                        z_max = max(0.0, extrude, border_extrude)
                        box_d = max(z_max - z_min, 1e-6)

                        center_offset = Pos(
                            X=box_w / 2.0, Y=box_h / 2.0, Z=-(z_min + z_max) / 2.0
                        )

                        outer_pts, _ = calculate_outer_pts(
                            offset_x=-center_offset.x, offset_y=-center_offset.y
                        )
                        eval_node = EvaluationNodeData(
                            node=self, offset=center_offset, curve_points=outer_pts
                        )

                        with Locations(center_offset):
                            Box(box_w, box_h, box_d, custom_data=eval_node)
                with Locations(Pos(Z=-extrude)):
                    build_children()

            if not self._is_eval and cached_part is None:
                outer_pts, source_curve = calculate_outer_pts()
                build_background(outer_pts, bp)
                build_other_kinds()
                add_tags(tag_to_list(style.root_tag))
                with BuildPart(
                    mode=mode if extrude == 0.0 else Mode.JOIN, loc_ctx_passthrough=True
                ) as bp_child:
                    with Locations(Pos(Z=-extrude)):
                        build_children()
                        apply_clip(outer_pts)
                    build_border(outer_pts, source_curve, bp, bp_child)
                    with BuildPart(part=bp.part, mode=Mode.PRIVATE):
                        for part in subtract_parts:
                            add(part, mode=Mode.SUBTRACT_FAST)
                build_extrude_subtract_parts(outer_pts)
                apply_3d_ops(bp)
                if self.kind == "joint":
                    bp.part.register_joint(
                        self.attrs["name"],
                        bp.part.joint(Location(), deformable=True),
                        propagate=True,
                    )
                if self._cache_hash is not None:
                    # The cache stores render-stage geometry only. Layout has
                    # already resolved placement, so identical components can
                    # share construction while each instance gets its transform.
                    self._build_ctx.part_cache[self._cache_hash] = MLCacheEntry(
                        part=bp.part.copy(), subtract_parts=subtract_parts.copy()
                    )

            if style.pivot_x or style.pivot_y or style.pivot_z:
                pivot_transform = Pos(
                    X=_len_to_m(style.pivot_x, inner_w, style.unit_scale) or 0.0,
                    Y=_len_to_m(style.pivot_y, inner_h, style.unit_scale) or 0.0,
                    Z=-(
                        _len_to_m(
                            style.pivot_z,
                            bp.bbox.max.z - bp.bbox.min.z,
                            style.unit_scale,
                        )
                        or 0.0
                    ),
                )
                transform(op=pivot_transform)
                bp.transform = pivot_transform.inverse

            final_transform = style.transform
            if style.x_offset or style.y_offset or style.z_offset:
                final_transform = Pos(
                    X=_len_to_m(style.x_offset, inner_w, style.unit_scale) or 0.0,
                    Y=_len_to_m(style.y_offset, inner_h, style.unit_scale) or 0.0,
                    Z=-(
                        _len_to_m(
                            style.z_offset,
                            bp.bbox.max.z - bp.bbox.min.z,
                            style.unit_scale,
                        )
                        or 0.0
                    ),
                ) * (final_transform or Transform())

            if final_transform is not None and self.kind != "part":
                final_transform = Origin(XY=0.5, Z=-1) * final_transform
                final_transform = final_transform.resolve(bp.part)
                transform(op=final_transform)

            if not self._is_eval:
                for i, part in enumerate(subtract_parts):
                    clone = part.copy()
                    if style.apply_transform_for_subtract_parts and final_transform:
                        clone.transform = final_transform * clone.transform
                    clone.transform = parent_bp._location_context[0] * clone.transform
                    subtract_parts[i] = clone

            if style.subtract:
                bp._mode = Mode.SUBTRACT_FAST
            elif extrude == 0.0:
                bp._mode = parent_mode

            if not self._is_eval:
                add_tags(tag_to_list(style.tag))
                for cb in self._on_build_callbacks:
                    sig = inspect.signature(cb.fn)
                    if len(sig.parameters) == 0:
                        cb.fn()
                    else:
                        cb.fn(bp)

    @property
    def _is_eval(self):
        """Whether the current build is an evaluation."""
        return bool(self._build_ctx and self._build_ctx.evaluate)

    def _evaluate(self):
        part = Part.box_set_empty()
        ctx = MLBuildContext(
            node_data=self._build_ctx.node_data,
            root_bp=self._build_ctx.root_bp,
            evaluate=True,
        )
        with ml_build_context(ctx), BuildPart(part=part, mode=Mode.PRIVATE):
            self._emit_node(None, 0.0, [])
        if self._build_ctx.eval_transform:
            part.apply_transform(self._build_ctx.eval_transform)
        for box in part.boxes:
            box: BoxSetPart.Box[EvaluationNodeData]
            box.custom_data.node.eval_box = box

    def build(
        self,
        mode=Mode.JOIN,
        width: Optional[float] = None,
        height: Optional[float] = None,
        build_ctx: Optional[MLBuildContext] = None,
        subtract_parts: list[Part] = [],
        remove_double_verts=False,
        register_chain_joints=True,
        layout_solver: SolverLike = sm.nelder_mead(),
        layout_objective: Callable = lambda: None,
        evaluate: bool = False,
        instantiation_delay_sec=0.1,
    ):
        """Resolve the layout first, then emit the resulting final geometry."""
        ctx = build_ctx or MLBuildContext()
        ctx.evaluate = evaluate
        is_most_top_root = ctx.root_bp is None
        root = ml(root_style, MLStyle(evaluate=False), self)
        orig_style = self.style
        self.style = MLStyle.abs_size(width, height) + self.style

        with ml_build_context(ctx):
            root._layout(
                layout_solver,
                layout_objective,
            )
            root._compute_cache_hash()
            time.sleep(instantiation_delay_sec)  # hack: to avoid crash
            with BuildPart(
                part=Part.box_set_empty() if self._is_eval else None, mode=mode
            ) as bp:
                ctx.root_bp = ctx.root_bp or bp
                root._emit_node(None, 0.0, subtract_parts)
                bp.part._fix_topology(remove_double_verts=remove_double_verts)
                if register_chain_joints:

                    def reg_chain_joint(axis: VectorLike):
                        loc = bp.part.bbox_joint(axis).loc
                        bp.part.register_joint(
                            name=_chain_joint_axis_name(axis),
                            joint=bp.part.joint(
                                Pos(X=loc.x, Y=loc.y, Z=0.0), deformable=True
                            ),
                        )

                    for axis in (Axis.X, Axis.Y):
                        reg_chain_joint(axis)
                        reg_chain_joint(-axis)
                if is_most_top_root:
                    for data in ctx.node_data.values():
                        assert data.resolved_style, "Resolved style is not set"
                        show_box = data.resolved_style.show_eval_box
                        if show_box:
                            box = data.eval_box
                            assert box, "Eval box is not set"
                            with BuildPart(mode=Mode.JOIN) as eval_box_bp:
                                add(box.part)
                                add(points_to_curve(box.custom_data.curve_points))
                                eval_box_bp.transform = box.transform
                                eval_box_bp.mat = mat.PBR(alpha=0.1) + (
                                    show_box
                                    if isinstance(show_box, MaterialLayer)
                                    else mat.yellow
                                )

        self.style = orig_style
        self.parent = None
        return bp.part

    def __repr__(self) -> str:
        return f"ml(kind={self.kind!r}, box={self.box}, children={len(self.children)})"


def on_build(fn):
    if not ml._ctx_stack:
        raise RuntimeError("on_build used outside of 'with ml()' context")

    ml._ctx_stack[-1]._on_build_callbacks.append(fn)
    return fn
