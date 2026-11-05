import importlib
import math
import sys

import bpy
from mathutils import Vector as BlenderVector

if "." not in sys.path:
    sys.path.insert(0, ".")

if "blender_cad" in sys.modules:
    importlib.reload(sys.modules["blender_cad"])

from blender_cad import *

# Change these three dimensions and rerun this script to regenerate the block.
BUILDING_SIZE = (32.0, 9.0, 15.5)  # length, depth, total height in Blender units
FLOOR_HEIGHT = 2.8
ENTRANCE_MODULE = 8.0
END_WINDOW_MODULE = 3.0

BASE_HEIGHT = 0.5
ROOF_THICKNESS = 0.28
PARAPET_HEIGHT = 0.48
SITE_THICKNESS = 0.16
BALCONY_DEPTH = 1.2
BALCONY_THICKNESS = 0.18
COLLECTION = "Khrushchevka_Parametric"


def pbr(name, color, roughness=0.8, metallic=0.0):
    return mat.PBR(
        base_color=(*color, 1.0), roughness=roughness, metallic=metallic
    ) + mat.Meta(name=name)


def facade_layout(
    width,
    height,
    module_count,
    module_width,
    module_window_width,
    floor_count,
    floors=FLOOR_HEIGHT,
    window_quantity=2,
    include_balconies=True,
    include_entrances=False,
):
    window_height = min(1.45, floors * 0.58)
    door_width = min(1.2, module_width * 0.22)
    door_height = min(2.1, floors * 0.78)

    def window():
        return ml(
            style(
                width=module_window_width,
                height=window_height,
                background_mat=glass_mat,
                border_width=0.06,
                border_mat=frame_mat,
                border_extrude=0.13,
                extrude=0.12,
                tag="window",
            )
        )

    def entrance():
        return ml(
            style(
                width=door_width,
                height=door_height,
                background_mat=door_mat,
                border_width=0.06,
                border_mat=frame_mat,
                border_extrude=0.14,
                extrude=0.13,
                tag="entrance",
            ),
            ml(
                style(
                    position="absolute",
                    right="18%",
                    top="52%",
                    width=0.08,
                    height=0.13,
                    background_mat=metal_mat,
                    border_radius="30%",
                    extrude=0.16,
                )
            ),
        )

    def balcony_door():
        return ml(
            style(
                width=door_width,
                height=door_height,
                background_mat=glass_mat,
                border_width=0.06,
                border_mat=frame_mat,
                border_extrude=0.14,
                extrude=0.13,
                tag="balcony-door",
            )
        )

    def floor_row(floor_index):
        windows = [window() for _ in range(window_quantity)]
        if include_entrances and floor_index == 0:
            feature = [entrance()]
        elif include_balconies and floor_index > 0:
            feature = [balcony_door()]
        else:
            feature = []

        return ml(
            style(
                width=module_width,
                height=floors,
                display="flex",
                flex_direction="row",
                justify_content="space-around",
                align_items="center",
                border_width=0.025,
                border_mat=joint_mat,
                border_extrude=0.035,
                border_in_measure=False,
                flex_shrink=0.0,
                tag="floor-panel",
            ),
            *windows,
            *feature,
        )

    def entrance_section(section_index):
        return ml(
            style(
                width=module_width,
                height=height,
                align="center",
                align_y="center",
                tag=f"section-{section_index + 1}",
            ),
            ml.generate_array(
                lambda floor_index: floor_row(floor_index),
                fill_mode="box",
                start=floor_count,
            ),
        )

    return ml(
        style(
            width=width,
            height=height,
            background_mat=wall_mat,
            extrude=0.08,
            align="center",
            align_y="center",
            tag="facade",
        ),
        ml.generate_array(
            lambda section_index: entrance_section(section_index),
            fill_mode="line",
            start=module_count,
        ),
    )


def build_ml_facade(name, layout, transform):
    with BuildPart() as facade_part:
        layout.build()
    facade_part.part.loc = transform
    facade_part.part.show(name=name, collection_name=COLLECTION)
    return facade_part


def build_parametric_house(size):
    length, depth, total_height = size
    usable_height = total_height - BASE_HEIGHT - ROOF_THICKNESS - PARAPET_HEIGHT
    floor_count = max(1, int(usable_height // FLOOR_HEIGHT))
    body_height = floor_count * FLOOR_HEIGHT
    body_top = BASE_HEIGHT + body_height

    entrance_count = max(1, math.ceil(length / ENTRANCE_MODULE))
    section_width = length / entrance_count
    end_module_count = max(1, math.ceil(depth / END_WINDOW_MODULE))
    end_section_width = depth / end_module_count

    window_width = min(1.25, section_width * 0.18)
    end_window_width = min(1.25, end_section_width * 0.42)
    balcony_width = min(2.2, section_width * 0.72)
    balcony_center_z = (
        BASE_HEIGHT + FLOOR_HEIGHT * floor_count / 2 + BALCONY_THICKNESS / 2
    )

    # Keep the building mass as a reusable Part and let ML provide its elevations.
    with BuildPart(mat=wall_mat) as result:
        Box(length, depth, body_height)
    result.part.loc = Pos(Z=BASE_HEIGHT + body_height / 2)

    # Foundation, site slab, roof cap, parapet and roof plant boxes.
    with BuildPart(mat=concrete_mat) as envelope:
        with Locations(Pos(Z=BASE_HEIGHT / 2)):
            Box(length + 0.8, depth + 0.8, BASE_HEIGHT, mode=Mode.JOIN)
        with Locations(Pos(Z=-SITE_THICKNESS / 2)):
            Box(length + 2.0, depth + 2.0, SITE_THICKNESS, mode=Mode.JOIN)
        with Locations(Pos(Z=body_top + ROOF_THICKNESS / 2)):
            Box(length + 0.9, depth + 0.9, ROOF_THICKNESS, mode=Mode.JOIN)
        parapet_center_z = body_top + ROOF_THICKNESS + PARAPET_HEIGHT / 2
        with Locations(Pos(Y=-(depth + 0.9) / 2, Z=parapet_center_z), Pos(Y=(depth + 0.9) / 2, Z=parapet_center_z)):
            Box(length + 0.9, 0.18, PARAPET_HEIGHT, mode=Mode.JOIN)
        with Locations(Pos(X=-(length + 0.9) / 2, Z=parapet_center_z), Pos(X=(length + 0.9) / 2, Z=parapet_center_z)):
            Box(0.18, depth + 0.9, PARAPET_HEIGHT, mode=Mode.JOIN)
        with Locations(Pos(Z=body_top + ROOF_THICKNESS + 0.3)):
            with GridLocations(section_width, 1.0, entrance_count, 1):
                Box(0.55, 0.55, 0.6, mode=Mode.JOIN)
    envelope.part.show(name="Foundation_Roof_and_Site", collection_name=COLLECTION)

    # Generate each elevation from the same ML component tree; section and floor
    # generators stop at the dimensions of their parent facade.
    front_layout = facade_layout(
        length,
        body_height,
        entrance_count,
        section_width,
        window_width,
        floor_count,
        window_quantity=2,
        include_balconies=True,
        include_entrances=True,
    )
    back_layout = facade_layout(
        length,
        body_height,
        entrance_count,
        section_width,
        window_width,
        floor_count,
        window_quantity=2,
        include_balconies=True,
        include_entrances=False,
    )
    front_transform = (
        Pos(X=length / 2, Y=-depth / 2, Z=BASE_HEIGHT)
        * Rot(Z=180)
        * Rot(X=90)
    )
    back_transform = Pos(X=-length / 2, Y=depth / 2, Z=BASE_HEIGHT) * Rot(X=90)
    right_end_transform = (
        Pos(X=length / 2, Y=depth / 2, Z=BASE_HEIGHT)
        * Rot(Z=180)
        * Rot(X=90)
        * Rot(Y=90)
    )
    left_end_transform = (
        Pos(X=-length / 2, Y=-depth / 2, Z=BASE_HEIGHT)
        * Rot(X=90)
        * Rot(Y=90)
    )
    build_ml_facade("Front_ML_Facade", front_layout, front_transform)
    build_ml_facade("Rear_ML_Facade", back_layout, back_transform)
    build_ml_facade(
        "Right_End_ML_Facade",
        facade_layout(
            depth,
            body_height,
            end_module_count,
            end_section_width,
            end_window_width,
            floor_count,
            window_quantity=1,
            include_balconies=False,
            include_entrances=False,
        ),
        right_end_transform,
    )
    build_ml_facade(
        "Left_End_ML_Facade",
        facade_layout(
            depth,
            body_height,
            end_module_count,
            end_section_width,
            end_window_width,
            floor_count,
            window_quantity=1,
            include_balconies=False,
            include_entrances=False,
        ),
        left_end_transform,
    )

    # A centered GridLocations pattern places one balcony above every entrance
    # on each long elevation. Grid Y becomes world Z through the local frame.
    if floor_count > 1:
        for side, basis in (
            (-1, Rot(X=90)),
            (1, Rot(Z=180) * Rot(X=90)),
        ):
            balcony_origin = Pos(
                Y=side * (depth / 2 + BALCONY_DEPTH / 2), Z=balcony_center_z
            ) * basis
            with BuildPart(mat=concrete_mat) as balcony_slabs:
                with Locations(balcony_origin):
                    with GridLocations(
                        section_width,
                        FLOOR_HEIGHT,
                        entrance_count,
                        floor_count - 1,
                    ):
                        Box(
                            balcony_width,
                            BALCONY_THICKNESS,
                            BALCONY_DEPTH,
                            mode=Mode.JOIN,
                        )
            balcony_slabs.part.show(
                name=f"Balcony_Slabs_{'Front' if side < 0 else 'Rear'}",
                collection_name=COLLECTION,
            )

            with BuildPart(mat=rail_mat) as balcony_rails:
                with Locations(balcony_origin):
                    with GridLocations(
                        section_width,
                        FLOOR_HEIGHT,
                        entrance_count,
                        floor_count - 1,
                    ):
                        for rail_y in (0.47, 0.96):
                            with Locations(
                                Pos(Y=rail_y, Z=BALCONY_DEPTH / 2)
                            ):
                                Box(balcony_width, 0.06, 0.06, mode=Mode.JOIN)
                            with Locations(
                                Pos(X=-balcony_width / 2, Y=rail_y),
                                Pos(X=balcony_width / 2, Y=rail_y),
                            ):
                                Box(0.06, 0.06, BALCONY_DEPTH, mode=Mode.JOIN)
                        post_positions = (
                            -balcony_width / 2,
                            -balcony_width / 4,
                            0.0,
                            balcony_width / 4,
                            balcony_width / 2,
                        )
                        for x_offset in post_positions:
                            with Locations(
                                Pos(
                                    X=x_offset,
                                    Y=0.48,
                                    Z=BALCONY_DEPTH / 2,
                                )
                            ):
                                Box(0.055, 1.0, 0.055, mode=Mode.JOIN)
                        with Locations(
                            Pos(X=-balcony_width / 2, Y=0.48, Z=-BALCONY_DEPTH / 2),
                            Pos(X=balcony_width / 2, Y=0.48, Z=-BALCONY_DEPTH / 2),
                        ):
                            Box(0.055, 1.0, 0.055, mode=Mode.JOIN)
            balcony_rails.part.show(
                name=f"Balcony_Rails_{'Front' if side < 0 else 'Rear'}",
                collection_name=COLLECTION,
            )

    # Four grouped entrance canopies and tiered steps follow the same section grid.
    with BuildPart(mat=concrete_mat) as entrances:
        with Locations(Pos(Y=-depth / 2 - 0.55, Z=BASE_HEIGHT + 2.65)):
            with GridLocations(section_width, 1.0, entrance_count, 1):
                Box(min(2.5, section_width * 0.72), 1.1, 0.16, mode=Mode.JOIN)
        for y_offset, z_center in ((-0.24, 0.42), (-0.62, 0.26), (-1.0, 0.10)):
            with Locations(
                Pos(Y=-depth / 2 + y_offset, Z=z_center)
            ):
                with GridLocations(section_width, 1.0, entrance_count, 1):
                    Box(min(2.0, section_width * 0.62), 0.38, 0.16, mode=Mode.JOIN)
    entrances.part.show(name="Entrance_Canopies_and_Steps", collection_name=COLLECTION)

    return result, {
        "length": length,
        "depth": depth,
        "height": total_height,
        "floors": floor_count,
        "entrances": entrance_count,
        "windows": (
            4 * entrance_count * floor_count
            + 2 * end_module_count * floor_count
        ),
        "balconies": 2 * entrance_count * max(0, floor_count - 1),
        "sections": entrance_count,
    }


wall_mat = pbr("Warm Khrushchev Panel Concrete", (0.72, 0.68, 0.58), 0.92)
joint_mat = pbr("Panel Joints", (0.57, 0.54, 0.47), 0.9)
concrete_mat = pbr("Foundation and Balcony Concrete", (0.40, 0.40, 0.37), 0.94)
frame_mat = pbr("Ivory Window Frames", (0.83, 0.81, 0.74), 0.72)
glass_mat = pbr("Blue Grey Window Glass", (0.12, 0.23, 0.32), 0.25, 0.15)
rail_mat = pbr("Dark Balcony Steel", (0.21, 0.24, 0.24), 0.62, 0.42)
door_mat = pbr("Entrance Door Paint", (0.25, 0.20, 0.16), 0.78)
metal_mat = pbr("Entrance Hardware", (0.56, 0.53, 0.45), 0.35, 0.75)

clear_scene()
result, dimensions = build_parametric_house(BUILDING_SIZE)

result.part.show(name="TEST", collection_name=COLLECTION)
result.part.obj.name = "Building_Structure"

for key, value in dimensions.items():
    bpy.context.scene[f"khrushchevka_{key}"] = value
bpy.context.scene["khrushchevka_size"] = BUILDING_SIZE

bpy.ops.object.select_all(action="SELECT")
bpy.context.view_layer.objects.active = result.part.obj
for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        space = area.spaces.active
        space.region_3d.view_rotation = BlenderVector((0.75, -1.0, 0.62)).to_track_quat(
            "Z", "Y"
        )
        space.region_3d.view_perspective = "PERSP"
        space.shading.color_type = "MATERIAL"
        space.shading.show_cavity = True
        space.overlay.show_cursor = False
        window_region = next(
            region for region in area.regions if region.type == "WINDOW"
        )
        with bpy.context.temp_override(area=area, region=window_region):
            bpy.ops.view3d.view_selected(use_all_regions=False)
        area.tag_redraw()

bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = None

print("PARAMETERS =", dimensions)
print("SCENE_OBJECTS =", len(bpy.context.scene.objects))
print("HASH = ", result.part.hash(use_materials=True))
