from blender_cad import *
from tests.test_base import BaseCADTest


class TestMLStandardFlowNoText(BaseCADTest):
    """
    Category: Standard flow without text.
    Covers sizing, spacing, positioning, alignment, and basic box-model behavior.
    """

    def test_basic_block_sizing_and_padding(self):
        """Verify width, height, padding, and background material on a simple block."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=3,
                    padding=1,
                    background_mat=mat.blue,
                ),
                ml(style(width=2, height=1, background_mat=mat.red)),
            ).build()

        self.assertPart(
            result.part,
            "3a0ef5f42a2a691c42476c070c58b99aee767e66d516555de712944a7ce7479d",
            "test_basic_block_sizing_and_padding",
            use_materials=True,
        )

    def test_min_max_sizes_and_aspect_ratio(self):
        """Verify min/max constraints and aspect ratio driven sizing."""
        with BuildPart() as result:
            ml(
                style(
                    width=2,
                    min_width=4,
                    min_height=2,
                    max_width=8,
                    max_height=6,
                    aspect_ratio=2,
                    background_mat=mat.green,
                ),
            ).build()

        self.assertPart(
            result.part,
            "d7582c6678e9e1b98b98bb83637fbf669c204349dd3afa3056d7984424354e5f",
            "test_min_max_sizes_and_aspect_ratio",
            use_materials=True,
        )

    def test_relative_and_absolute_positioning(self):
        """Verify absolute positioning inside a relatively positioned container."""
        with BuildPart() as result:
            ml(
                style(
                    width=10,
                    height=6,
                    padding=1,
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        position="absolute",
                        left="50%",
                        top="50%",
                        width=2,
                        height=1,
                        background_mat=mat.red,
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "b3a9466b42eb6c567f7c611ae989c86e3ae007c0355bb66e609d97e95bf52b67",
            "test_relative_and_absolute_positioning",
            use_materials=True,
        )

    def test_alignment_axes(self):
        """Verify align and align_y on a container with simple children."""
        with BuildPart() as result:
            ml(
                style(
                    width=8,
                    height=4,
                    padding=0.5,
                    background_mat=mat.blue,
                    align="center",
                    align_y="end",
                ),
                ml(style(width=1, height=1, background_mat=mat.red)),
                ml(style(width=2, height=1, background_mat=mat.yellow)),
            ).build()

        self.assertPart(
            result.part,
            "11858429eef7acfedad0ad34e147cbfcfd24011a60b5282a7cfcf3728efccfd8",
            "test_alignment_axes",
            use_materials=True,
        )

    def test_margin_variants(self):
        """Verify margin shorthand and per-side overrides."""
        with BuildPart() as result:
            ml(
                style(width=10, padding=0.5, background_mat=mat.blue),
                ml(style(width=2, height=1, margin=1, background_mat=mat.red)),
                ml(style(width=2, height=1, margin_tb=0.5, margin_lr=1, background_mat=mat.yellow)),
                ml(style(width=2, height=1, margin_top=0.2, margin_right=0.4, margin_bottom=0.6, margin_left=0.8, background_mat=mat.green)),
            ).build()

        self.assertPart(
            result.part,
            "4a68c02f5892ad15bb8bfa36f403c362cbf31e3c97349d1aa99df70600875ddc",
            "test_margin_variants",
            use_materials=True,
        )

    def test_padding_variants(self):
        """Verify padding shorthand and per-side overrides."""
        with BuildPart() as result:
            ml(
                style(width=10, background_mat=mat.blue, padding_tb=1, padding_lr=0.5),
                ml(style(width=2, height=1, background_mat=mat.red)),
                ml(style(width=2, height=1, background_mat=mat.yellow)),
            ).build()

        self.assertPart(
            result.part,
            "741d5ba3e960e40878d2b4b3af595b94a68100ee7c54af2662d5c353d8f597bd",
            "test_padding_variants",
            use_materials=True,
        )

    def test_absolute_position_with_anchor(self):
        """Verify absolute positioning with custom anchor values."""
        with BuildPart() as result:
            ml(
                style(
                    width=8,
                    height=6,
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        position="absolute",
                        left="50%",
                        top="50%",
                        anchor_x=0.0,
                        anchor_y=1.0,
                        width=2,
                        height=2,
                        background_mat=mat.red,
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "a9632e4a99e6cad9cfff57261a50d6c52edb040c04b1c0d655ecc958e0c52fc4",
            "test_absolute_position_with_anchor",
            use_materials=True,
        )
    
    def test_nested_opacity_multiplication(self):
        """Verify that nested opacity values multiply correctly through the hierarchy."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=4,
                    opacity=0.8,
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        width=2,
                        height=2,
                        opacity=0.5,
                        background_mat=mat.red,
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "12e6e002912cd89512867c90ee6212d95791538ff57c4dc6f919f411ce86a55c",
            "test_nested_opacity_multiplication",
            use_materials=True,
        )

    def test_display_none_does_not_affect_layout(self):
        """Verify that display none elements do not participate in layout flow."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    margin_top=0.5,
                    padding=1,
                    background_mat=mat.blue,
                ),
                ml(style(display="none", width=2, height=1, background_mat=mat.red)),
                ml(style(width=1, height=1, background_mat=mat.red)),
            ).build()

        self.assertPart(
            result.part,
            "b5823d4055b6b76bcad7238adfa6d03bc8212e9f540ec9d3128967b682a89a6a",
            "test_display_none_does_not_affect_layout",
            use_materials=True,
        )

    def test_dynamic_lambda_dimensions(self):
        """Verify that element dimensions can be dynamically calculated using lambda functions based on other elements."""
        with BuildPart() as result:
            a = ml(
                style(
                    width=2,
                    height=1,
                    background_mat=mat.red,
                ),
            )
            b = ml(
                style(
                    width=lambda: a.width * 2,
                    height=1,
                    background_mat=mat.green,
                ),
            )
            c = ml(
                style(
                    width=lambda cur_node: b.width + a.width - cur_node.height,
                    height=1,
                    background_mat=mat.yellow,
                ),
            )
            ml(
                style(
                    width=15,
                    height=1,
                    background_mat=mat.blue,
                ),
                c,
                b,
                a
            ).build()

        self.assertPart(
            result.part,
            "590d7d21feee98cb54d81209891af7e6f0d0d318e5402b99b1b3872975261cc3",
            "test_dynamic_lambda_dimensions",
            use_materials=True,
        )

    def test_preset_shapes_with_absolute_centering(self):
        """Verify the rendering of preset shapes (square and circles) with absolute centering."""
        with BuildPart() as result:
            ml(
                style.square(size=7, mat=mat.green),
                ml(style.circle(radius=3, mat=mat.red), style.absolute_center()),
                ml(style.circle(radius=2, mat=mat.blue), style.absolute_center(z_index=1)),
                ml(style.circle(radius=1, mat=mat.red), style.absolute_center(z_index=2)),
            ).build()

        self.assertPart(
            result.part,
            "d8715528aa0e17b496c3143f689203c656a2cf26767913c029e94344130c6793",
            "test_preset_shapes_with_absolute_centering",
            use_materials=True,
        )

    def test_relative_position_offsets(self):
        """Verify that relative positioning offsets move elements without affecting flow."""
        with BuildPart() as result:
            ml(
                style(
                    width=8,
                    background_mat=mat.blue,
                    font_size=0.5
                ),
                style.align_center(),
                # Normal position
                ml(
                    style.circle(radius=1, mat=mat.red),
                ),
                # Shifted right and top
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style(right=1, top=1)
                ),
                # Normal position
                ml(
                    style.circle(radius=1, mat=mat.red),
                ),
                # Small shift left
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style(left=0.2)
                ),
                # Shifted up using bottom offset
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style(bottom=0.5)
                ),
                # Larger shift up using bottom offset
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style(bottom=1)
                ),
                "hello, world!",
            ).build()

        self.assertPart(
            result.part,
            "472327bda186bcc158d367e42570e95f0e47054c7a191505ab8c7d004f384122",
            "test_relative_position_offsets",
            use_materials=True,
        )
    
    def test_part_3d_integration_and_transformation(self):
        """Verify that 3D parts are correctly integrated, styled, and transformed within layout."""
        with BuildPart() as result:
            # Create a source cylinder in private mode
            with BuildPart(mode=Mode.PRIVATE) as cylinder:
                Cylinder(radius=1, height=2)
                
            ml(
                style(
                    width=6,
                    height=4,
                    background_mat=mat.blue,
                    mat=mat.red,
                    display="flex",
                    justify_content="center",
                    align_items="center",
                ),
                ml.from_part(
                    cylinder, 
                    style(adapt_transform=True)
                ),
                ml(
                    style(background_mat=mat.green),
                    ml.from_part(
                        cylinder, 
                        style(transform=Rot(Z=45) * Scale(0.5), adapt_transform=True)
                    )
                )
            ).build()

        self.assertPart(
            result.part,
            "b426246498b317399ca637444311bf6bca503790e3605fe63d4c0bb657118599",
            "test_part_3d_integration_and_transformation",
            use_materials=True,
        )

    def test_ml_transform_with_custom_origin(self):
        """Verify that transform with a custom Origin correctly affects the element's rotation and positioning."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=3,
                    padding=1,
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        width=2,
                        height=1,
                        align="right",
                        align_y="end",
                        background_mat=mat.red,
                        # Pivot point is shifted by Origin before applying rotation
                        transform=Origin(X=1, Y=1) * Rot(Z=45)
                    ),
                    # Small marker to visualize the transformed coordinate system
                    ml(style(width=0.1, height=0.1, background_mat=mat.yellow)),
                ),
            ).build()

        self.assertPart(
            result.part,
            "82a457475fe3f0416353833b7d5886614834d8e726a725bf90be20e1763bff53",
            "test_ml_transform_with_custom_origin",
            use_materials=True,
        )

    def test_top_scale_with_border_warp(self):
        """Verify that top_scale correctly warps the element along with its border and border_radius."""
        with BuildPart() as result:
            ml(
                style(
                    width=3,
                    height=2,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                ml(
                    style(
                        width=2,
                        height=1,
                        top_scale=0.5,
                        background_mat=mat.red,
                        border_width=0.1,
                        border_mat=mat.yellow,
                        border_radius="50%",
                    )
                ),
            ).build()

        self.assertPart(
            result.part,
            "d13eb95c1b335f509133e97099a06fefefaa79839a791f448732de177341188a",
            "test_top_scale_with_border_warp",
            use_materials=True,
        )

    def test_circular_primitive_overflow_clipping(self):
        """Verify that overflow='hidden' correctly masks and clips child elements to a circular boundary."""
        with BuildPart() as result:
            ml(
                style(
                    width=4,
                    height=4,
                    background_mat=mat.blue,
                    dissolve=0.0 # to stabilize hash of the result part
                ),
                # Target Container: A centered circle that acts as an overflow mask
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style.absolute_center(),
                    style(border_mat=mat.yellow, border_width=0.1),
                    style(overflow="hidden"),                      # Core masking trigger
                    
                    # Child 1: Full-width top horizontal strip
                    ml(
                        style(
                            width="100%",
                            height="10%",
                            background_mat=mat.green,
                        )
                    ),
                    # Child 2: Shifted green strip with its own blue border
                    ml(
                        style(
                            width="100%",
                            height="20%",
                            margin_top=0.5,
                            background_mat=mat.green,
                        ),
                        style(border_mat=mat.blue, border_width=0.1),
                    )
                ),
            ).build()

        self.assertPart(
            result.part,
            "6aeddeaadfed33ca27aa1a72a266955be1c83db0e1de978ab4e6484eca66373a",
            "test_circular_primitive_overflow_clipping",
            use_materials=True,
        )

    def test_container_multi_element_centering(self):
        """Verify that multiple distinct children are correctly centered horizontally and vertically."""
        with BuildPart() as result:
            ml(
                style(
                    name="background",
                    width=4,
                    height=4,
                    align="center",
                    align_y="center",
                    background_mat=mat.blue,
                ),
                # Child 1: Red circle primitive
                ml(
                    style(name="circle"),
                    style.circle(radius=1.5, mat=mat.red),
                ),
                # Child 2: Green rectangular strip
                ml(
                    style(name="rect"),
                    style(
                        width=2,
                        height=0.5,
                        background_mat=mat.green,
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "0afa40999dba692666ee0478d02619492de11848cfec58353f73c50c24a060c4",
            "test_container_multi_element_centering",
            use_materials=True,
        )

    def test_percentage_overflow_and_alignment(self):
        """Verify that a 200% height/width child correctly overflows its parent while remaining centered."""
        with BuildPart() as result:
            ml(
                ml(
                    style(
                        width=5,
                        height=5,
                        padding=1,
                        background_mat=mat.blue,
                        align="center",
                        align_y="center",
                    ),
                    ml(
                        style(
                            width="200%",
                            height=1,
                            background_mat=mat.red,
                        ),
                    ),
                ),
                ml(
                    style(
                        width=5,
                        height=5,
                        padding=1,
                        background_mat=mat.green,
                        align="center",
                        align_y="center",
                    ),
                    ml(
                        style(
                            width=1,
                            height="200%",
                            background_mat=mat.red,
                        ),
                    ),
                )
            ).build()

        self.assertPart(
            result.part,
            "d2b37699c697f9868c1f306025245c279ea7376f4c83042220f9d7d6018bf312",
            "test_percentage_overflow_and_alignment",
            use_materials=True,
        )

    def test_explicit_new_line_breaks(self):
        """Verify that ml.new_line() forces row breaks and structures the layout matrix correctly."""
        
        with BuildPart() as result:
            box = lambda: ml(
                style(
                    width=1,
                    height=1,
                    margin_right=0.1,
                    margin_bottom=0.1,
                    background_mat=mat.red,
                ),
            )

            ml(
                style(
                    width=5,
                    height=5,
                    background_mat=mat.blue,
                    align="center",
                    align_y="center",
                ),
                box(),
                box(),
                ml.new_line(),
                box(),
                ml.new_line(),
                box(),
                ml.new_line(),
                box(),
                box(),
                box(),
            ).build()

        self.assertPart(
            result.part,
            "eaed9662f07b66bf8f83362405657001b18c88815ce7bd4267b8a215b73ef5d3",
            "test_explicit_new_line_breaks",
            use_materials=True,
        )

    def test_layout_engine_geometry_caching(self):
        """Verify that identical child nodes hit the cache, yielding 4 unique items for 5 nodes."""
        with BuildPart() as result:
            root = ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                
                # Node 1: Red box with a margin -> Cache Entry #1
                ml(
                    style(
                        width=1,
                        height=1,
                        margin_right=1, # does not affect cache
                        extrude=-0.5,
                        background_mat=mat.red,
                    ),
                ),
                
                # Node 2: Plain red box -> Cache Entry #2
                ml(
                    style(
                        width=1,
                        height=1,
                        extrude=-0.5,
                        transform=Pos(Z=-0.25), # does not affect cache
                        background_mat=mat.red,
                    ),
                ),
                
                
                # Node 3: Yellow box (Different material) -> Cache Entry #3
                ml(
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.yellow, # affects cache
                    ),
                ),
            )
            # Note: The root container itself counts as Cache Entry #4
            build_ctx = ml.BuildContext()
            root.build(build_ctx=build_ctx)

        # Asset that the total unique geometry elements in the cache store equals exactly 4
        cache_length = len(build_ctx.part_cache)
        self.assertEqual(
            cache_length, 
            3 + 1, 
            f"Cache optimization failed! Expected 4 unique entries, got {cache_length}."
        )

        self.assertPart(
            result.part,
            "b1f7f7980e510733b2dfe0d0c310e59399d6301366e8a570c7afdbab415aef77",
            "test_layout_engine_geometry_caching",
            use_materials=True,
        )

    def test_generate_array_fill_mode_box(self):
        """1. Verify fill_mode='box' behavior where items wrap to fill multiple rows/columns inside parent bounds."""
        with BuildPart() as result:
            root = ml(
                style(
                    width=7,
                    height=2,
                    background_mat=mat.blue,
                ),
                ml.generate_array(
                    lambda i: ml(
                        style(
                            width=1,
                            height=1,
                            margin_left=1,
                            background_mat=mat.red if i % 2 == 0 else mat.green,
                        )
                    ),
                    fill_mode="box",
                ),
            )
            root.build()

        self.assertPart(result.part, "bd56d462be603347663b3dc97a4101d43eb1ca6c1cafbf3ecdc26ae404735740", "test_generate_array_fill_mode_box", use_materials=True)


    def test_generate_array_fill_mode_line(self):
        """2. Verify fill_mode='line' behavior where node generation halts strictly at the end of the first line."""
        with BuildPart() as result:
            root = ml(
                style(
                    width=7,
                    height=2,
                    background_mat=mat.blue,
                ),
                ml.generate_array(
                    lambda i: ml(
                        style(
                            width=1,
                            height=1,
                            margin_left=1,
                            background_mat=mat.red if i % 2 == 0 else mat.green,
                        )
                    ),
                    fill_mode="line",
                ),
            )
            root.build()

        self.assertPart(result.part, "d668c62b383f87b486104b6a77e7099ef3f62d3cb42468885bebc586fd56d63b", "test_generate_array_fill_mode_line", use_materials=True)


    def test_generate_array_nested_matrix(self):
        """3. Verify nested generation combining 'line' columns and 'box' rows to create a grid matrix."""
        with BuildPart() as result:
            root = ml(
                style(
                    width=7,
                    height=7,
                    background_mat=mat.blue,
                ),
                ml.generate_array(
                    lambda: ml(
                        style(
                            width=1,
                            height="100%",
                            margin_left=1,
                            background_mat=mat.red
                        ),
                        ml.generate_array(
                            lambda: ml(
                                style(
                                    width=1,
                                    height=1,
                                    margin_top=1,
                                    background_mat=mat.green
                                ),
                            ),
                            fill_mode="box",
                        ),
                    ),
                    fill_mode="line",
                ),
            )
            root.build()

        self.assertPart(result.part, "e15353787c2c44b225bf4f5cf828c7c36b9728da7a849cdfe45ec45579bc1bfd", "test_generate_array_nested_matrix", use_materials=True)


    def test_generate_array_dynamic_incremental_sizing(self):
        """4. Verify that generator tracks state index `i` correctly for dynamic layout item sizing and step termination."""
        with BuildPart() as result:
            root = ml(
                style(
                    width=11,
                    height=3,
                    padding=1,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                ml.generate_array(
                    lambda i: ml(
                        style(
                            width=i + 1,
                            height=0.5 + i * 0.2,
                            margin_left=1 if i > 0 else None,
                            background_mat=mat.red,
                        )
                    ),
                ),
            )
            root.build()

        self.assertPart(result.part, "c3139f7423af6bfefd8801d2779fb29c068428e2b5d2ed8647a72559a57ec1fa", "test_generate_array_dynamic_incremental_sizing", use_materials=True)

    def test_surface_mapping_with_nested_layout_and_materials(self):
        """
        Verify that layout elements in a loop are correctly mapped onto the 
        cylinder's parametric surface using UV coordinates, and that nested 
        styles, materials, and extrusions are properly applied.
        """
        with BuildPart() as result:
            # Setup base cylinder and its material
            Cylinder(1, 4)
            result.mat = mat.blue

            # 1. Base layout container positioned on the cylinder surface face.
            # It applies a 180-degree X-rotation and a small normal Z-offset.
            ml(
                style(
                    width=3,
                    height=1,
                    locations=Locations(faces().cylinders_only()[0].location(uv) * Rot(X=180) * Pos(Z=0.005))
                ),
                [
                    # 2. Generate 5 child layout elements distributed via UV coordinates
                    ml(
                        style(
                            width=1,
                            height=0.5,
                            margin_left=0.2,
                            margin_bottom=0.2,
                            background_mat=mat.red,
                            x_offset="-50%",
                            y_offset="-50%",
                        ),
                        # 3. Create extruded yellow circles at the absolute center of each cell
                        ml(
                            style.circle(radius=0.2, mat=mat.yellow),
                            style.absolute_center(),
                            style(extrude=0.1)
                        )
                    ) for _ in range(5)
                ]
            ).build()

        # Verification via hash.
        # This checks that:
        # - 5 distinct elements are generated and correctly projected onto the cylinder surface.
        # - Layout parameters (margins, percentages, absolute centering) resolve correctly.
        # - Material assignments (Blue base, Red background, Yellow circle) are preserved.
        # - The 0.1 extrusion is correctly applied along the local face normals.
        self.assertPart(
            result.part, 
            "1313c6d56da3c8bb737be5193067a1f4758af13d75eeecef1aabe68433ddd473", 
            "test_surface_mapping_with_nested_layout_and_materials"
        )

    def test_flex_layout_distribution_along_wire_path(self):
        """Verify that justify_content='space-around' evenly spaces elements along a 3D wire trajectory."""
        with BuildPart() as result:
            # Step 1: Establish the base 3D solid geometry reference
            Cylinder(1, 1)
            result.mat = mat.blue
            
            # Isolate the top circular wire boundary of the cylinder
            wire = wires().top()[0]

            # Step 2: Map the flat flex layout container onto the target 3D wire coordinates
            ml(
                style(
                    display="flex",
                    justify_content="space-around",     # Evenly spaces out the 5 children along the loop
                    width=wire.length(),               # Flat canvas matches the exact unrolled path perimeter
                    height=1,
                    locations=Locations(wire.location() * Rot(X=180) * Pos(Z=0.005)) # Project slightly above cap face
                ),
                # Generate 5 compound markers to distribute around the rim
                [
                    ml(
                        style(
                            width=0.5,
                            height=0.5,
                            background_mat=mat.red,
                            x_offset="-50%",           # Align local geometry origin to the tracking path center
                            y_offset="-10%",
                        ),
                        # Nested embossed yellow pin indicator
                        ml(
                            style.circle(radius=0.1, mat=mat.yellow),
                            style.absolute_center(),
                            style(extrude=0.1)
                        )
                    ) for _ in range(5)
                ]
            ).build()

        self.assertPart(
            result.part,
            "8786936723ebea9f7a52d281f81f295e3fba91a186b0b701f8a0b0a28e08ec06",
            "test_flex_layout_distribution_along_wire_path",
            use_materials=True,
        )

    def test_joint_and_attachment(self):
        """Verify that ml.joint registration and .to() connection correctly links two parts in 3D space."""
        with BuildPart() as result:
            # Step 1: Create the child part with a defined joint at its bottom center
            child = ml(
                (s := style(
                    width=0.3,
                    height=0.3,
                    background_mat=mat.green,
                    extrude=1
                )),
                # Register a target joint named 'bottom_center' on the child part
                ml.joint("bottom_center", X="50%", Y="50%", Z=-s.extrude, flip=True),
            ).part

            # Step 2: Create the main part with a nested circular sub-element and a top joint
            main = ml(
                style(
                    width=4,
                    height=4,
                    background_mat=mat.blue,
                    extrude=0.1
                ),
                ml(
                    style(
                        width="100%",
                        height="100%",
                        transform=Origin(Y=1) * Rot(X=90, Z=180),
                    ),
                    ml(
                        style.circle(radius=1, mat=mat.red),
                        style.absolute_center(),
                        style(border_mat=mat.yellow, border_width=0.1),
                        style(
                            extrude=0.1,
                            extrude_delete_source_faces=False,
                            border_extrude_delete_source_faces=False
                        ),
                        # Register a matching target joint named 'top_center' on the main part
                        ml.joint("top_center", X="50%", Y="50%"),
                    ),
                )
            ).part

            # Step 3: Add the main base part to the active build context
            add(main)
            
            # Step 4: Actively connect the child's joint to the main part's joint using JOIN mode
            child.joint_by_name("bottom_center").to(main.joint_by_name("top_center"), mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "dbc350740d33abfe99400250ba499175113db901ef85000d0a0bd6243e109cd5",
            "test_joint_and_attachment",
            use_materials=True,
        )

    def test_background_generation_from_curve_boundary(self):
        """Verify that background_from_curve correctly maps a layout container into the shape of a closed curve."""
        with BuildPart() as result:
            # Step 1: Define a closed 2D spline curve profile using a local curve context
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)
            
            # Step 2: Build the layout structure projecting dimensions onto the reference spline profile
            ml(
                style(
                    width=5,
                    # height is set automatically with the original curve height = 10
                    background_mat=mat.blue,
                    background_from_curve=bc,
                    border_mat=mat.red,
                    border_width=0.1,
                    extrude=1
                ),
                # Step 3: Embed a nested green component exactly at the center of the generated geometry
                ml(
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.green,
                        extrude=-0.5
                    ),
                    style.absolute_center()
                )
            ).build()

        self.assertPart(
            result.part,
            "2314cab1653632d09cb663941251214888543e39300defa405f6511f067c8fa3",
            "test_background_generation_from_curve_boundary",
            use_materials=True,
        )


class TestMLStandardFlowText(BaseCADTest):
    """
    Category: Standard flow with text.
    Covers text sizing, color, background, wrapping, alignment, spacing, and nested inline styles.
    """

    def test_simple_text_block(self):
        """Verify a text block with explicit size, background, and text material."""
        with BuildPart() as result:
            ml(
                style(
                    width=10,
                    padding=0.5,
                    font_size=1,
                    background_mat=mat.blue,
                    mat=mat.red,
                ),
                "Hello",
            ).build()

        self.assertPart(
            result.part,
            "f89551f39b02d7fb078ef40a477b6b08effb99bb33b9722e8fd434d9b8ce2269",
            "test_simple_text_block",
            use_materials=True,
        )

    def test_text_alignment_right(self):
        """Verify right-aligned multiline text."""
        with BuildPart() as result:
            ml(
                style(
                    width=12,
                    padding=0.5,
                    font_size=1,
                    text_align="right",
                    background_mat=mat.blue,
                    mat=mat.red,
                ),
                "Hello, world!\nI love you",
            ).build()

        self.assertPart(
            result.part,
            "4b9276306bf18b68ae8a6ef31ee0bd10cae3bb0ac3ad4c57896dead916d9c5fb",
            "test_text_alignment_right",
            use_materials=True,
        )

    def test_text_alignment_center_and_wrap_character(self):
        """Verify center alignment and character wrapping."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    padding=0.5,
                    font_size=1.5,
                    text_align="center",
                    wrap_mode="character",
                    background_mat=mat.blue,
                    mat=mat.yellow,
                ),
                "Hello, world!",
            ).build()

        self.assertPart(
            result.part,
            "069d40e4c07d1cd21f692c15f72972fa562cd50af4f97a00f88e82d80061e969",
            "test_text_alignment_center_and_wrap_character",
            use_materials=True,
        )

    def test_inline_bold_italic_and_nested_background(self):
        """Verify mixed inline styling, nested boxes, and text color overrides."""
        with BuildPart() as result:
            ml(
                style(
                    width=10,
                    padding=0.3,
                    font_size=1.5,
                    text_align="center",
                    background_mat=mat.blue,
                    mat=mat.red,
                ),
                "Hello my best",
                " world! ",
                ml.b("I love you", style(font_size=2)),
                ml.i("so much", style(mat=mat.yellow, letter_spacing=0.3)),
                "! ",
                "This my number: ",
                ml(
                    style(
                        background_mat=mat.yellow,
                        padding_lr=0.1,
                    ),
                    "12345",
                ),
            ).build()

        self.assertPart(
            result.part,
            "6ff62d50907105086fe8f7c6270cf9455b2454c5067b0d9c22053f6cfc99ac9d",
            "test_inline_bold_italic_and_nested_background",
            use_materials=True,
        )

    def test_font_weight_style_and_line_height(self):
        """Verify font weight, italic style, and line height behavior."""
        with BuildPart() as result:
            ml(
                style(
                    width=12,
                    padding=0.5,
                    font_size=1,
                    font_weight="bold",
                    font_style="italic",
                    line_height=1.5,
                    background_mat=mat.green,
                    mat=mat.red,
                ),
                "Bold italic\nwith extra spacing",
            ).build()

        self.assertPart(
            result.part,
            "ffbbc6d0650bada6e8fca7cc25444599eacb9da85d3877bf91224fcd68029142",
            "test_font_weight_style_and_line_height",
            use_materials=True,
        )

    def test_letter_and_word_spacing(self):
        """Verify letter and word spacing on a compact text block."""
        with BuildPart() as result:
            ml(
                style(
                    width=15,
                    padding=0.4,
                    font_size=1,
                    letter_spacing=0.3,
                    word_spacing=0.6,
                    background_mat=mat.blue,
                    mat=mat.red,
                ),
                "Spacing test with words",
            ).build()

        self.assertPart(
            result.part,
            "2eb8f12f41a2697d8bed92f6762c727c9bb767d4fc895fd8831dfeb257ac58a9",
            "test_letter_and_word_spacing",
            use_materials=True,
        )

    def test_multiple_words_and_nested_text_boxes(self):
        """Verify a longer sentence with nested text containers and per-word sizing."""
        with BuildPart() as result:
            ml(
                style(
                    width=16,
                    padding=0.5,
                    font_size=1.2,
                    text_align="left",
                    background_mat=mat.blue,
                    mat=mat.red,
                ),
                "One ",
                ml(style(font_size=1.8), "two "),
                ml(style(font_size=0.8, mat=mat.yellow), "three "),
                ml(style(background_mat=mat.green, padding_lr=0.15), "four"),
            ).build()

        self.assertPart(
            result.part,
            "131930999c205a2edd9f4fa8b15bb8cf4f05a173e55e22fbf9e0ec78b397c2c7",
            "test_multiple_words_and_nested_text_boxes",
            use_materials=True,
        )

    def test_centered_inline_layout_with_text_and_boxes(self):
        """Verify centered alignment with inline text and neighboring box elements."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=2,
                    margin_top=1,
                    font_size=0.5,
                    border_mat=mat.yellow,
                    background_mat=mat.blue,
                    mat=mat.red,
                    align="center",
                    align_y="center",
                ),
                ml(
                    style(
                        width=1,
                        height=1,
                        margin_right=0.5,
                        background_mat=mat.red,
                    ),
                ),
                ml.b("Hello", style(margin_top=0.2)),
                ml(
                    style(
                        width=1,
                        height=1,
                        margin_left=0.5,
                        background_mat=mat.red,
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "fb10f459ef50518547705f74781a2aef9271848e3588c2d6d9b73f5359438029",
            "test_centered_inline_layout_with_text_and_boxes",
            use_materials=True,
        )

    def test_text_stroke_rendering(self):
        """Verify that text stroke width and materials are applied correctly to text elements."""
        with BuildPart() as result:
            ml(
                style(
                    font_size=1,
                    background_mat=mat.blue,
                    mat=mat.red,                # Main text body color
                    text_stroke_width=0.05,     # Thickness of the outline
                    text_stroke_mat=mat.yellow, # Outline color
                    text_extrude=0.1,           # Depth of the main text
                    text_stroke_extrude=0.05,    # Depth of the stroke (offset for 3D effect)
                ),
                "Hello, world!"
            ).build()

        self.assertPart(
            result.part,
            "75d6c7fc50a5ae5150254d752270c5bf6dbda1182570f74ac7779ee51a06e42c",
            "test_text_stroke_rendering",
            use_materials=True,
        )

    def test_overflow_clipping_behavior(self):
        """Verify that content is clipped correctly by both parent and self overflow properties."""
        with BuildPart() as result:
            root = ml(
                style(background_mat=mat.blue, font_size=1, width=2, overflow="hidden"),
                # "Hello, world!" is clipped by its own style (overflow="hidden")
                ml(style(height=1, overflow="hidden"), "Hello, world!"),
                # "world" is clipped by the parent's style (root overflow="hidden")
                ml(style(height=1), "world")
            )
            root.build()

        self.assertPart(
            result.part,
            "0ee4b4958ea7c4163646dcd3d4e46e13007feaadf5804814b1cd7c15420ae65b",
            "test_overflow_clipping_behavior",
            use_materials=True,
        )

    def test_extrusion_mode_vs_z_offset_layering(self):
        """Verify solid 3D extrusion generation via mode='extrude' with nested child shapes."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=5,
                    background_mat=mat.blue,
                    mode='extrude'
                ),
                style.align_center(),
                ml(
                    style(
                        width=4,
                        height=4,
                        background_mat=mat.green,
                        border_width=0.2,
                        border_mat=mat.yellow,
                        z_index=20
                    ),
                    style.align_center(),
                    ml(
                        ml(
                            style.circle(radius=1, mat=mat.red),
                            style(
                                border_width=0.1,
                                border_mat=mat.yellow,
                                font_size=0.5,
                                text_stroke_mat=mat.yellow,
                                text_stroke_width=0.02,
                            ),
                            style.align_center(),
                            "Hello"
                        ),
                        ml(
                            style.circle(radius=1, mat=mat.red),
                            style(right=0.5, z_index=1),
                            ml(
                                style.circle(radius=0.5, mat=mat.yellow),
                                style.absolute_center(),
                                style(
                                    border_width=0.1,
                                    border_mat=mat.green,
                                    border_extrude=0.1,
                                    extrude=0.2
                                )
                            )
                        )
                    )
                ),
            ).build()

        self.assertPart(
            result.part,
            "d8fbf9b72d3562b853d09e53225d01a0c1bab7ef51aa9a90121bd701c8b3cd0d",
            "test_extrusion_mode_vs_z_offset_layering",
            use_materials=True,
        )


class TestMLFlexFlow(BaseCADTest):
    """
    Category: Flex flow.
    Covers row/column flex direction, wrapping, gaps, justify-content, align-items, and align-content.
    """

    def test_row_flex_auto_size(self):
        """Verify basic row flex sizing without explicit container width/height."""
        with BuildPart() as result:
            ml(
                style(
                    padding=1,
                    width=10,
                    background_mat=mat.green,
                ),
                ml(
                    style(
                        padding=1,
                        display="flex",
                        flex_direction="row",
                        background_mat=mat.blue,
                    ),
                    ml(style(width=1, height=1, background_mat=mat.red)),
                    ml(style(width=1, height=1, margin_left=1, background_mat=mat.red)),
                )
            ).build()

        self.assertPart(
            result.part,
            "6811f0ea30c9e9218b2f6099ff3e2b39f5921183ab5d37aa2a1bc96e4acbe6e9",
            "test_row_flex_auto_size",
            use_materials=True,
        )

    def test_row_flex_justify_space_between(self):
        """Verify space-between distribution in a row flex container."""
        with BuildPart() as result:
            ml(
                style(
                    width=12,
                    padding=1,
                    display="flex",
                    flex_direction="row",
                    justify_content="space-between",
                    background_mat=mat.blue,
                ),
                [ml(style(width=1, height=1, background_mat=mat.red)) for _ in range(4)],
            ).build()

        self.assertPart(
            result.part,
            "11528343e134652e17702e9e4c43a8021aac96d388960fd357e7fbdbca90f000",
            "test_row_flex_justify_space_between",
            use_materials=True,
        )

    def test_row_flex_center_with_gap(self):
        """Verify centered row flex with a fixed gap."""
        with BuildPart() as result:
            ml(
                style(
                    width=12,
                    padding=1,
                    display="flex",
                    flex_direction="row",
                    justify_content="center",
                    gap=1,
                    background_mat=mat.blue,
                ),
                [ml(style(width=1, height=1, background_mat=mat.red)) for _ in range(3)],
            ).build()

        self.assertPart(
            result.part,
            "7df7a13d72046352fe4619138639a5f7257165ef0c1c715500fb449c508e8dec",
            "test_row_flex_center_with_gap",
            use_materials=True,
        )

    def test_column_flex_space_between_and_center_items(self):
        """Verify column flex layout with vertical spacing and centered cross-axis alignment."""
        with BuildPart() as result:
            ml(
                style(
                    width=8,
                    height=8,
                    padding=1,
                    display="flex",
                    flex_direction="column",
                    justify_content="space-between",
                    align_items="center",
                    background_mat=mat.blue,
                ),
                [ml(style(width=1, height=1, background_mat=mat.red)) for _ in range(4)],
            ).build()

        self.assertPart(
            result.part,
            "3cfb7a4ebdbe6575959570518b7b1d36612f2e23d9fdf8d28d9ca00d87c8e52a",
            "test_column_flex_space_between_and_center_items",
            use_materials=True,
        )

    def test_wrapping_and_align_content(self):
        """Verify wrapping rows and align-content distribution."""
        with BuildPart() as result:
            ml(
                style(
                    width=12,
                    height=6,
                    padding=1,
                    display="flex",
                    flex_direction="row",
                    flex_wrap="wrap",
                    justify_content="center",
                    align_content="space-between",
                    gap=1,
                    background_mat=mat.blue,
                ),
                [ml(style(width=1, height=1, background_mat=mat.red)) for _ in range(6)],
            ).build()

        self.assertPart(
            result.part,
            "0c9219e73318065d16840fddb94ac54d2891d1b702c86d3877088bddbd5e7a71",
            "test_wrapping_and_align_content",
            use_materials=True,
        )

    def test_nested_flex_containers(self):
        """Verify nested flex containers with different alignment strategies."""
        with BuildPart() as result:
            ml(
                style(
                    width=14,
                    height=8,
                    padding=0.5,
                    display="flex",
                    flex_direction="row",
                    justify_content="space-between",
                    align_items="center",
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        width=5,
                        height=6,
                        padding=0.5,
                        display="flex",
                        flex_direction="column",
                        justify_content="space-between",
                        align_items="center",
                        background_mat=mat.red,
                    ),
                    [ml(style(width=1, height=1, background_mat=mat.yellow)) for _ in range(3)],
                ),
                ml(
                    style(
                        width=5,
                        height=6,
                        padding=0.5,
                        display="flex",
                        flex_direction="column",
                        justify_content="space-between",
                        align_items="center",
                        background_mat=mat.green,
                    ),
                    [ml(style(width=1, height=1, background_mat=mat.yellow)) for _ in range(3)],
                ),
            ).build()

        self.assertPart(
            result.part,
            "4b0cf7b528d75dfb08eeddbcc813cc964f16bce97703c3f4485e83add57632c5",
            "test_nested_flex_containers",
            use_materials=True,
        )

    def test_stretch_alignment_with_mixed_child_sizes(self):
        """Verify stretch alignment with children that have partial size information."""
        with BuildPart() as result:
            ml(
                style(
                    width=12,
                    height=6,
                    padding=1,
                    display="flex",
                    flex_direction="row",
                    justify_content="space-between",
                    align_items="stretch",
                    background_mat=mat.blue,
                ),
                [
                    ml(style(width=1, background_mat=mat.red)),
                    ml(style(width=1, height=2, background_mat=mat.red)),
                    ml(style(width=1, background_mat=mat.red)),
                ],
            ).build()

        self.assertPart(
            result.part,
            "3ce5bedcdf1dff0c576c91ba1179aeabc657512efe08b0481e190385dffa939f",
            "test_stretch_alignment_with_mixed_child_sizes",
            use_materials=True,
        )
    
    def test_column_stretch_alignment_with_partial_widths(self):
        """Verify stretch alignment in a column flex layout with partially sized children."""
        with BuildPart() as result:
            ml(
                style(
                    width=12,
                    height=6,
                    margin_top=0.5,
                    padding=1,
                    display="flex",
                    flex_direction="column",
                    justify_content="space-between",
                    align_items="stretch",
                    background_mat=mat.blue,
                ),
                [
                    ml(style(height=1, background_mat=mat.red)),
                    ml(style(height=1, width=2, background_mat=mat.red)),
                    ml(style(height=1, background_mat=mat.red)),
                ],
            ).build()

        self.assertPart(
            result.part,
            "7cb4e52131d25069d77505371fe8e9a9b99428a096f6e1f0e2bf34c80778049e",
            "test_column_stretch_alignment_with_partial_widths",
            use_materials=True,
        )

    def test_flex_shrink_behavior(self):
        """Verify that flex children shrink correctly based on their flex_shrink factors when space is limited."""
        with BuildPart() as result:
            ml(
                style(
                    display="flex",
                    flex_direction="row",
                    width=10,
                    height=3,
                    gap=0.2,
                    background_mat=mat.blue,
                ),
                ml(style(width=6, flex_shrink=1, background_mat=mat.red)),
                ml(style(width=4, flex_shrink=2, background_mat=mat.green)),
                ml(style(width=3, flex_shrink=0, background_mat=mat.yellow)),
            ).build()

        self.assertPart(
            result.part,
            "21ddba4b7f9c75357821e6e60c58ce4f715d13637aad8284013a8c86e101939c",
            "test_flex_shrink_behavior",
            use_materials=True,
        )

    def test_flex_grow_distribution(self):
        """Verify proportional space distribution among children using different flex_grow values."""
        with BuildPart() as result:
            ml(
                style(
                    display="flex",
                    flex_direction="row",
                    width=20,
                    height=4,
                    gap=0.2,
                    background_mat=mat.blue,
                ),
                ml(style(width=2, flex_grow=1, background_mat=mat.red)),
                ml(style(width=2, flex_grow=2, background_mat=mat.green)),
                ml(style(width=3, flex_grow=0, background_mat=mat.yellow)),
            ).build()

        self.assertPart(
            result.part,
            "636864ccd1f5f87488b09ba520b94b10bf9b38675e9b1fe86233585aee8f719c",
            "test_flex_grow_distribution",
            use_materials=True,
        )

    def test_flex_mixed_content_and_alignment(self):
        """Verify layout with mixed fixed-width blocks and growing text blocks with vertical alignment."""
        with BuildPart() as result:
            ml(
                style(
                    display="flex",
                    flex_direction="row",
                    width=10,
                    height=3,
                    gap=0.2,
                    background_mat=mat.blue,
                    font_size=1,
                    mat=mat.yellow,
                    text_align="center",
                ),
                ml(style(width=6, background_mat=mat.red)),
                ml(style(flex_grow=1, background_mat=mat.green, align_y="center"), "Hello!"),
            ).build()

        self.assertPart(
            result.part,
            "11919e996a994510f728c0ce92a788a47022404031ccac1bf4aa042900cefcba",
            "test_flex_mixed_content_and_alignment",
            use_materials=True,
        )

    def test_flex_layout_with_relative_offsets(self):
        """Verify that relative offsets work correctly within a wrapped flex container."""
        with BuildPart() as result:
            root = ml(
                style(
                    width=7,
                    height=10,
                    background_mat=mat.blue,
                ),
                style.flex_center(),
                style(
                    flex_wrap="wrap", 
                    align_items="baseline", 
                    align_content="center", 
                    gap=0.1
                ),
                # Row 1: Red circles
                ml(style.circle(radius=1, mat=mat.red)),
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style(right=1, top=1), # Shifted out of flow
                ),
                ml(style.circle(radius=1, mat=mat.red)),
                ml(style.circle(radius=1, mat=mat.red)),
                
                # Row 2: Yellow circles
                ml(style.circle(radius=1, mat=mat.yellow)),
                ml(
                    style.circle(radius=1, mat=mat.yellow),
                    style(right=1, bottom=2, z_index=1) # Dramatic upward shift
                ),
                ml(style.circle(radius=1, mat=mat.yellow)),
                
                # Row 3: Green circles
                ml(style.circle(radius=1, mat=mat.green)),
                ml(style.circle(radius=1, mat=mat.green)),
            )
            root.build()

        self.assertPart(
            result.part,
            "cceb23b2f8b8115a58547c0a6409e7b5334b9451d2992cb589f216931abcf72b",
            "test_flex_layout_with_relative_offsets",
            use_materials=True,
        )


class TestMLBorder(BaseCADTest):
    """
    Category: Border.
    Covers border width, style, radius variants, and material assignment.
    """

    def test_solid_border_with_radius(self):
        """Verify a basic solid border and uniform radius."""
        with BuildPart() as result:
            ml(
                style(
                    width=4,
                    height=4,
                    border_width=0.2,
                    border_style="solid",
                    border_radius=0.5,
                    border_mat=mat.yellow,
                    background_mat=mat.red,
                ),
            ).build()

        self.assertPart(
            result.part,
            "9362148dee524bea81c8d9854adef4a42b12d3702969940711e0b8d02b97a6dc",
            "test_solid_border_with_radius",
            use_materials=True,
        )

    def test_dashed_border(self):
        """Verify a dashed border with a rounded rectangle."""
        with BuildPart() as result:
            ml(
                style(
                    width=4,
                    height=4,
                    border_width=0.1,
                    border_style="dashed",
                    border_radius="30%",
                    border_mat=mat.yellow,
                    background_mat=mat.red,
                ),
            ).build()

        self.assertPart(
            result.part,
            "54b2e44ac1aea05d46205883fb5a310aa93b2c0ca8624ae04009da894d04003c",
            "test_dashed_border",
            use_materials=True,
        )

    def test_dotted_border(self):
        """Verify a dotted border with a simple background fill."""
        with BuildPart() as result:
            ml(
                style(
                    width=4,
                    height=4,
                    border_width=0.1,
                    border_style="dotted",
                    border_mat=mat.green,
                    background_mat=mat.blue,
                ),
            ).build()

        self.assertPart(
            result.part,
            "81adc3bd74be432e8596a38b09375e7514f5d23ee0b3490af213e3c4f05647a5",
            "test_dotted_border",
            use_materials=True,
        )

    def test_double_border(self):
        """Verify a double border with explicit border material."""
        with BuildPart() as result:
            ml(
                style(
                    width=4,
                    height=4,
                    border_width=0.15,
                    border_style="double",
                    border_mat=mat.yellow,
                    background_mat=mat.red,
                ),
            ).build()

        self.assertPart(
            result.part,
            "2395479fec1b0d8cd4cc4ef3a26737b36ef1d2bc97786a87b267b124b8c8cd1a",
            "test_double_border",
            use_materials=True,
        )

    def test_asymmetric_border_radii(self):
        """Verify separate corner radii and side radii."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=4,
                    border_width=0.1,
                    border_style="solid",
                    border_radius_tl="50%",
                    border_radius_tr="25%",
                    border_radius_bl="10%",
                    border_radius_br="40%",
                    border_mat=mat.yellow,
                    background_mat=mat.blue,
                ),
            ).build()

        self.assertPart(
            result.part,
            "f0ca6d8033a5b509052b0f78767b8398275bd8bcca1ecd8f75869481d613644f",
            "test_asymmetric_border_radii",
            use_materials=True,
        )

    def test_border_radius_sides(self):
        """Verify top/bottom/left/right radius group settings."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=4,
                    border_width=0.12,
                    border_style="solid",
                    border_radius_top="50%",
                    border_radius_bottom="20%",
                    border_radius_left="35%",
                    border_radius_right="15%",
                    border_mat=mat.green,
                    background_mat=mat.red,
                ),
            ).build()

        self.assertPart(
            result.part,
            "d30fc1199da3b8e15aa3561d5c624c554bafe5a66a0f42fa9f0efd263680bb98",
            "test_border_radius_sides",
            use_materials=True,
        )

    def test_negative_border_radii(self):
        """Verify negative (concave) border radii rendering."""

        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=4,
                    border_width=0.12,
                    border_style="solid",

                    # Concave corners.
                    border_radius_tl="-35%",
                    border_radius_tr="-15%",
                    border_radius_bl="-25%",
                    border_radius_br="-45%",

                    border_mat=mat.green,
                    background_mat=mat.red,
                ),
            ).build()

        self.assertPart(
            result.part,
            "ca2929f9b823c1438a8bc3ed5a194f6add7dba493970825a7ccaeaf779a39a8f",
            "test_negative_border_radii",
            use_materials=True,
        )

    def test_border_extrusion_and_depth_warp(self):
        """Verify independent extrusion depths for element body and its border under warp conditions."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=3,
                    background_mat=mat.blue,
                    dissolve=0.0 # to stabilize hash of the result part
                ),
                style.align_center(),
                ml(
                    style(
                        width=3,
                        height=2,
                        background_mat=mat.red,
                        border_width=0.1,
                        border_mat=mat.yellow,
                        extrude=0.5,            # Body depth
                        border_extrude=1,       # Border sticks out further
                        border_style="double",   # Double line border rendering
                        top_scale=0.5,          # Trapezoidal deformation
                        border_radius="20%"     # Rounded warped corners
                    )
                ),
            ).build()

        self.assertPart(
            result.part,
            "7bbca69c230af35985f7f394d9b2b57a1964218b995987b1fba30349dee3b1e6",
            "test_border_extrusion_and_depth_warp",
            use_materials=True,
        )

    def test_advanced_border_properties(self):
        """Verify advanced border configurations including dashed patterns, offsets, scaling, and Z-shifts."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=4,
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        border_width=0.1,
                        border_style="dashed",
                        border_offset=0.1,       # Expands the border bounds outward from the element
                        border_dash_length=0.2,  # Length of individual dashes
                        border_step_scale=0.2,   # Distance or scaling between dashes
                        border_z_index=10,     # Lifts the border slightly forward on the Z-axis
                        border_mat=mat.yellow,
                    ),
                    ml(
                        style.circle(radius=1, mat=mat.red),
                    ),
                )
            ).build()

        self.assertPart(
            result.part,
            "7fd3e654375dd0136c942d17f31252e82d02fea25c9376f882a1f07dca85598f",
            "test_advanced_border_properties",
            use_materials=True,
        )

    def test_border_on_intersecting_complex_shapes(self):
        """Verify border tracking and sizing around a complex composition of intersecting shapes."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=4,
                    padding=0.1, # need this to prevent border from being sticked outwards
                    background_mat=mat.blue,
                ),
                style.align_center(),
                ml(
                    style(
                        border_width=0.1,
                        border_mat=mat.yellow,
                    ),
                    # Left Red Circle
                    ml(
                        style.circle(radius=1, mat=mat.red),
                    ),
                    # Right Red Circle (shifted left by 1 unit via right offset)
                    ml(
                        style.circle(radius=1, mat=mat.red),
                        style(right=1)
                    ),
                    # Vertical Center Strip (spanning 100% height of parent container bounds)
                    ml(
                        style.absolute_center(),
                        style(height="100%", width=0.3, background_mat=mat.red),
                    ),
                    # Horizontal Center Strip (spanning 100% width of parent container bounds)
                    ml(
                        style.absolute_center(),
                        style(height=0.3, width="100%", background_mat=mat.red),
                    ),
                )
            ).build()

        self.assertPart(
            result.part,
            "1e5478fe6f46f6f3e9ffba8be4a9660ee87c7cea892016c7d3aff3289b37dbff",
            "test_border_on_intersecting_complex_shapes",
            use_materials=True,
        )

    def test_box_sizing_and_border_measurement_modes(self):
        """Verify that borders are correctly factored into layout dimensions unless excluded by border_in_measure."""
        with BuildPart() as result:
            ml(
                style(
                    width=4,
                    height=4,
                    background_mat=mat.green,
                ),
                # Case 1: Border with offset (Expands outwardly but stays measured)
                ml(
                    style(
                        width="100%",
                        height="50%",
                        background_mat=mat.blue,
                        border_mat=mat.yellow,
                        border_width=0.2,
                        border_offset=0.2
                    ),
                ),
                # Case 2: Standard Box Model Border (Included in measurements)
                ml(
                    style(
                        width="100%",
                        height="25%",
                        background_mat=mat.red,
                        border_mat=mat.blue,
                        border_width=0.1,
                    ),
                ),
                # Case 3: Decoupled Border (Excluded from calculations, overflows boundaries)
                ml(
                    style(
                        width="100%",
                        height="10%",
                        background_mat=mat.red,
                        border_mat=mat.yellow,
                        border_width=0.05,
                        border_in_measure=False  # Border bleeds out without shifting layout flow
                    ),
                )
            ).build()

        self.assertPart(
            result.part,
            "21f62c68825a48a1c21589df5ac5ae07ff3b9a3c3b293c4c51dc3a43dd07643b",
            "test_box_sizing_and_border_measurement_modes",
            use_materials=True,
        )

    def test_border_no_inner_space_contribution(self):
        """Verify that a container's border does not alter or contribute to its inner content dimensions."""
        with BuildPart() as result:
            ml(
                style(
                    border_width=0.1,
                    border_mat=mat.yellow,
                    background_mat=mat.blue,
                ),
                ml(
                    style.circle(radius=1, mat=mat.red),
                ),
            ).build()

        self.assertPart(
            result.part,
            "f9546688194a7656fb471a967e1f5fedb4a5a85992fdc8aed4dd02e1f3230feb",
            "test_border_no_inner_space_contribution",
            use_materials=True,
        )

    def test_recessed_frame_via_negative_border_extrusion(self):
        """Verify that a negative border_extrude creates a carved trench frame using only a single child."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                # A single child that draws a blue center and carves its own red border into the parent
                ml(
                    style(
                        width=4,
                        height=4,
                        background_mat=mat.blue,
                        border_radius="10%",
                        border_mat=mat.red,
                        border_width=0.5,
                        border_extrude=-0.5  # Directly carves a 0.5-deep trench under the border path
                    ),
                )
            ).build()

        self.assertPart(
            result.part,
            "2b0e4077d03ff317bef8cd2cdba80934919c6c2d90ff4d100f113868f64e7350",
            "test_recessed_frame_via_negative_border_extrusion",
            use_materials=True,
        )

    def test_concentric_triple_border_nesting(self):
        """Verify that border_around_background=True sequentially nests recursive border shells without overlapping."""
        with BuildPart() as result:
            ml(
                # Layer 1: Outermost blue border shell
                style(
                    width=5,
                    height=5,
                    border_width=0.1,
                    border_mat=mat.blue,
                    border_radius="10%",
                    border_around_background=True
                ),
                style.align_center(),
                ml(
                    # Layer 2: Intermediate green border shell (stretches to 100% of parent inner space)
                    style(
                        width="100%",
                        height="100%",
                        border_width=0.1,
                        border_mat=mat.green,
                        border_radius="10%",
                        border_around_background=True
                    ),
                    style.align_center(),
                    ml(
                        # Layer 3: Innermost yellow border shell
                        style(
                            width="100%",
                            height="100%",
                            border_width=0.1,
                            border_mat=mat.yellow,
                            border_radius="10%",
                            border_around_background=True
                        ),
                        style.align_center(),
                        # Core Content: Solid red center block occupying half the inner yellow frame space
                        ml(
                            style(
                                width="50%",
                                height="50%",
                                background_mat=mat.red,
                            ),
                        )
                    )
                )
            ).build()

        self.assertPart(
            result.part,
            "2d98ee0de77fadf312a7aaf89ce4efa20a4de1e084cf95da8717f32af9c4457a",
            "test_concentric_triple_border_nesting",
            use_materials=True,
        )

    def test_extreme_border_radius_bottom_left_saturation(self):
        """Verify that a single corner radius at 100% safely clamps to edge boundaries without breaking geometry."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=5,
                    border_radius_bl="100%",  # Corner saturation test (Bottom-Left)
                    background_mat=mat.blue,
                ),
            ).build()

        self.assertPart(
            result.part,
            "09c65bbb3460165c5de3beb0da9cd86d2f9bf010d8a0f99d838d5409d6d3a1fa",
            "test_extreme_border_radius_bottom_left_saturation",
            use_materials=True,
        )


    def test_over_saturated_side_radius_clamping(self):
        """Verify that border_radius_left='200%' on a rectangular profile clamps smoothly to the maximum physical limit."""
        with BuildPart() as result:
            ml(
                style(
                    width=10,
                    height=5,
                    border_radius_left="200%",  # Extreme side saturation test
                    background_mat=mat.blue,
                ),
            ).build()

        self.assertPart(
            result.part,
            "04b8bf0f79d2040eceb105ee13db8e627fd16a50ccab08816428f5bebf1387f7",
            "test_over_saturated_side_radius_clamping",
            use_materials=True,
        )

    def test_border_ml_array_generation_distribution(self):
        """Verify that border_ml successfully populates and distributes generated array objects along the perimeter."""
        with BuildPart() as result:
            ml(
                style(
                    width=10,
                    height=5,
                    border_radius_bl="100%",
                    background_mat=mat.blue,
                    extrude=1,
                ),
                style.border_ml(
                    ml.generate_array(
                        lambda: ml(
                            style.circle(radius=0.1, mat=mat.red),
                            style(
                                margin_left=2,
                                extrude=0.1,
                            ),
                        ),
                        fill_mode="line",
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "9bb19fe68b10ca12adae21990d3891bb844a8a667c76eb8f220262270c60eee0",
            "test_border_ml_array_generation_distribution",
            use_materials=True,
        )

    def test_wheel_rim_and_spokes_distribution(self):
        """Verify wheel-like layout structure with spokes distributed around a circular border and a central hub."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=5,
                    border_radius="50%",
                    border_radius_segments=20,
                    background_mat=mat.blue,
                    extrude=0.1,
                ),
                style.border_ml(
                    style(
                        display="flex",
                        justify_content="space-around",
                    ),
                    [
                        ml(
                            style(
                                width=0.5,
                                height=2,
                                background_mat=mat.red,
                                x_offset="-50%",
                                bottom_scale=0.3,
                                margin_left=2,
                                extrude=0.1,
                            )
                        ) for _ in range(10)
                    ],
                ),
                ml(
                    style.circle(radius=0.5, mat=mat.yellow),
                    style.absolute_center(),
                    style(extrude=0.2),
                ),
            ).build()

        self.assertPart(
            result.part,
            "917049addcbdd1b10030dc32659e75f2d2f3a27dba185f5926cf51c8785576b1",
            "test_wheel_rim_and_spokes_distribution",
            use_materials=True,
        )

    def test_border_flex_distribution_with_inner_recess(self):
        """Verify flex objects distribute along extruded borders inside a recessed cavity."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=5,
                    border_radius_bl="100%",
                    background_mat=mat.blue,
                    extrude=1,
                ),
                style.border_ml(
                    style(
                        display="flex",
                        justify_content="space-around"
                    ),
                    [
                        ml(
                            style(
                                width=1,
                                height=0.5,
                                background_mat=mat.red,
                                extrude=-0.2,
                                extrude_subtract_part_height_k=1,
                            ),
                            style.border_extrude_loc(Y=0.5),
                            ml(
                                style.circle(radius=0.1, mat=mat.yellow),
                                style.absolute_center(),
                                style(extrude=0.4),
                            ),
                        ) for _ in range(5)
                    ],
                ),
            ).build()

        self.assertPart(
            result.part,
            "e68a1e274e4556ce4920b1562c1a872af4a94ff06bb29fa6bc7e33322257ec48",
            "test_border_flex_distribution_with_inner_recess",
            use_materials=True,
        )

    def test_border_location_context_passthrough(self):
        """Verify that location context passes through nested ml containers to distribute objects along the border curve."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=5,
                    border_radius_bl="100%",
                    background_mat=mat.blue,
                    extrude=1,
                ),
                style.border_ml(
                    ml(
                        style(
                            loc_ctx_passthrough=True,
                            margin_left=1,
                        ),
                        [
                            ml(
                                style(
                                    width=1,
                                    height=0.5,
                                    background_mat=mat.red,
                                    x_offset="-50%",
                                    y_offset="-100%",
                                    extrude=0.2,
                                    margin_left=1,
                                ),
                                ml(
                                    style.circle(radius=0.1, mat=mat.yellow),
                                    style.absolute_center(),
                                    style(extrude=0.1),
                                ),
                            ) for _ in range(3)
                        ],
                    )
                ),
            ).build()

        self.assertPart(
            result.part,
            "dca3ff3e7e7b81362b2d2e1b09b33f7ac38cbffd413fee4109ffab2392170cef",
            "test_border_location_context_passthrough",
            use_materials=True,
        )

    def test_background_curve_tag_selection_and_border_mapping(self):
        """Verify that a curve used for background_from_curve maintains its tag system for specific border element mapping."""
        with BuildPart() as result:
            # Step 1: Build a complex layout where the background geometry follows a tagged curve profile
            ml(
                style(
                    width=5,
                    height=7,
                    background_mat=mat.blue,
                    background_from_curve=(
                        c := curve(
                            curve.smooth(radius=10),
                            curve.tag("A"),
                            curve.step(10, angle=0, tag="B"),
                            curve.step(10, angle=90, tag="C"),
                            curve.step(10, angle=90, tag="D"),
                        ).build()
                    ),
                ),
                # Step 2: Distribute red sub-components exclusively along the segment tagged as 'C'
                style.border_ml(
                    style(
                        display="flex",
                        justify_content="space-around",
                    ),
                    [
                        ml(
                            style(
                                width=1,
                                height=1,
                                background_mat=mat.red,
                            ),
                        ) for _ in range(3)
                    ],
                    selector=lambda: c.tagged("C"),
                ),
                # Step 3: Attach a green component to the segment tagged as 'D', excluding its first point
                style.border_ml(
                    style(
                        display="flex",
                        justify_content="center",
                    ),
                    ml(
                        style(
                            width=1,
                            height=1,
                            background_mat=mat.green,
                        ),
                    ),
                    selector=lambda: c.tagged("D").untagged(Curve.TAG_POINT_FIRST),
                ),
                # Step 4: Attach a yellow component to the segment tagged as 'B', excluding its last point
                style.border_ml(
                    style(
                        display="flex",
                        justify_content="center",
                    ),
                    ml(
                        style(
                            width=1,
                            height=1,
                            background_mat=mat.yellow,
                        ),
                    ),
                    selector=lambda: c.tagged("B").untagged(Curve.TAG_POINT_LAST),
                ),
            ).build()
            
            # Step 5: Append the reference path curve skeleton to the final part representation
            add(c)

        self.assertPart(
            result.part,
            "ed626dec11ed0a43987c869c20ce96a5a0fa3d356d5e6e9d6c4287807e28527b",
            "test_background_curve_tag_selection_and_border_mapping",
            use_materials=True,
        )


class TestMLThreeDOperations(BaseCADTest):
    """
    Category: 3D operations.
    Covers extrude and text_extrude behavior.
    """

    def test_block_extrude(self):
        """Verify extrusion on a plain block."""
        with BuildPart() as result:
            ml(
                style(
                    width=3,
                    height=2,
                    extrude=1,
                    background_mat=mat.blue,
                ),
            ).build()

        self.assertPart(
            result.part,
            "194d531e2e5fc7ac097ea774f76b8ce75201ce9beb40c0a1bc8ace9fcd9fcd29",
            "test_block_extrude",
            use_materials=True,
        )

    def test_text_extrude(self):
        """Verify extrusion on text geometry only."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    padding=0.5,
                    font_size=1.2,
                    text_extrude=0.5,
                    background_mat=mat.blue,
                    mat=mat.yellow,
                ),
                "H",
            ).build()

        self.assertPart(
            result.part,
            "8995466aa5e21d889d9143f093ebccad7b577421cdef8cf072e3841ab4c39a79",
            "test_text_extrude",
            use_materials=True,
        )

    def test_combined_block_and_text_extrude(self):
        """Verify a block that extrudes both the container and its text."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=3,
                    extrude=0.8,
                    text_extrude=0.3,
                    font_size=1,
                    background_mat=mat.red,
                    mat=mat.yellow,
                ),
                "3D",
            ).build()

        self.assertPart(
            result.part,
            "98f90863924594ce4fef9d6a67411af847617ebdac92cbfdba92c6273a9a3324",
            "test_combined_block_and_text_extrude",
            use_materials=True,
        )

    def test_nested_mirroring_and_reflection_scales(self):
        """Verify recursive mirroring behaviors across different sides with positive and negative scaling."""
        with BuildPart() as result:
            ml.mirror(
                lambda: ml(
                    style(
                        width=4 * 4,
                        height=4,
                    ),
                    # Block 1: Blue box with right-side mirror reflection
                    ml.mirror(
                        lambda: ml(
                            style(
                                width=4,
                                height=4,
                                background_mat=mat.blue,
                                font_size=1,
                                align="center",
                                align_y="end",
                            ),
                            ml(
                                style(
                                    width=1,
                                    height=1,
                                    top_scale=0.5,
                                    right_scale=0.5,
                                    background_mat=mat.red,
                                ),
                            )
                        ),
                        side="right",
                        flip=False
                    ),
                    # Block 2: Green box with left-side inverted mirror reflection
                    ml.mirror(
                        lambda: ml(
                            style(
                                width=4,
                                height=4,
                                background_mat=mat.green,
                                font_size=1,
                                align="right",
                                align_y="center",
                            ),
                            ml(
                                style(
                                    width=1,
                                    height=1,
                                    top_scale=0.5,
                                    right_scale=0.5,
                                    background_mat=mat.red,
                                ),
                            )
                        ),
                        side="left",
                        flip=True
                    )
                ),
                side="bottom",
                flip=True
            ).build()

        self.assertPart(
            result.part,
            "e2495a5a863922391bd622a3c65e07b56396706fe12a9fa099826a9034ebea8f",
            "test_nested_mirroring_and_reflection_scales",
            use_materials=True,
        )

    def test_mirrored_negative_extrusion_cutouts(self):
        """Verify that negative extrusions inside a mirrored layout cleanly cut holes in both symmetrical positions."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                # Mirror wrapper duplicating the lambda factory output symmetrically
                ml.mirror(
                    lambda: ml(
                        style(
                            width=1,
                            height=1,
                            extrude=-0.1,        # Subtractive cutter property
                            right_scale=0.5,     # Tapered geometry modifier
                            background_mat=mat.red,
                        ),
                    ),
                    side="bottom",
                    offset=0.1
                )
            ).build()

        self.assertPart(
            result.part,
            "5f5e7280cc77134fc341e6f3b286050a16a48d26728dc59367944c039b310c26",
            "test_mirrored_negative_extrusion_cutouts",
            use_materials=True,
        )

    def test_negative_extrusion_boolean_subtraction(self):
        """Verify that negative extrusions pierce through parent elements to create openings."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                # Middle container: Extrudes upward into solid space
                ml(
                    style(
                        width=5,
                        height=5,
                        background_mat=mat.green,
                        extrude=0.5,
                    ),
                    style.align_center(),
                    # Inner container: Uses negative extrusion to cut into the green block
                    ml(
                        style(
                            width=4,
                            height=4,
                            background_mat=mat.red,
                            extrude=-0.5,       # CSG Subtraction / Hole cutting operation
                            border_radius="10%"
                        ),
                        style.align_center(),
                        # Core elements sitting inside the opening
                        ml(
                            style(
                                width=3,
                                height=3,
                                background_mat=mat.green,
                                mat=mat.red
                            ),
                            # Deepest child: Cuts an additional circular hole through layers
                            ml(
                                style.circle(radius=1, mat=mat.yellow),
                                style.absolute_center(),
                                style.align_center(),
                                style(extrude=-0.5, font_size=0.5),  # Secondary cutout pass
                                "Hello"
                            )
                        ),
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "0f2e025d649b9809a3792352b94ea8e0827937c9dfd00c0df9e84087d65f8deb",
            "test_negative_extrusion_boolean_subtraction",
            use_materials=True,
        )

    def test_recessed_frame_via_inverse_extrusions(self):
        """Verify that alternating negative and positive extrusions create a recessed frame/trench effect."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                # Step 1: Cut a 5x5 trench pocket into the blue base plate
                ml(
                    style(
                        width=5,
                        height=5,
                        background_mat=mat.red,
                        extrude=-0.5,       # Recesses the geometry backward
                        border_radius="10%"
                    ),
                    style.align_center(),
                    # Step 2: Push a 4x4 inner core back forward to the surface level
                    ml(
                        style(
                            width=4,
                            height=4,
                            background_mat=mat.blue,
                            extrude=0.5,        # Extrudes forward inside the cutout
                            border_radius="10%"
                        ),
                    )
                )
            ).build()

        self.assertPart(
            result.part,
            "0badb62dca84b525eb3d8c8ba0f4067359ed12674722b4e894ba33735ab581f7",
            "test_recessed_frame_via_inverse_extrusions",
            use_materials=True,
        )

    def test_text_negative_extrusion_engraving(self):
        """Verify that text characters with negative text_extrude engrave directly into parent faces."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                # Target container: A green plate that will receive the engraved text
                ml(
                    style(
                        width=5,
                        height=5,
                        background_mat=mat.green,
                        mat=mat.red,            # Material color assigned to the text/cutout geometry
                        font_size=2,
                        text_extrude=-0.5       # Directly carves text shapes 0.5 units deep into the green plate
                    ),
                    style.align_center(),
                    "Hello"
                )
            ).build()

        self.assertPart(
            result.part,
            "cbad48de5ba1e6bc75af55538c47b52cc5829c32a6c0a92f71fca654d707ed36",
            "test_text_negative_extrusion_engraving",
            use_materials=True,
        )

    def test_proportional_directional_box_extrusion(self):
        """Verify that style.prop_box_extrude dynamically adjusts extrusion depths along specific edges."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                # Base container: Configured to carve a pocket 1 unit deep
                ml(
                    style(
                        width=5,
                        height=5,
                        background_mat=mat.green,
                        extrude=-1,
                        border_radius="30%"
                    ),
                    # Modulates the bottom edge's extrusion depth ratio to 0.2
                    style.prop_box_extrude(bottom=0.2)
                )
            ).build()

        self.assertPart(
            result.part,
            "8fdd67ad4afc7dc3763803cb2e8897ecad2c3f4cbc28e82b910155ce57f1a633",
            "test_proportional_directional_box_extrusion",
            use_materials=True,
        )

    def test_proportional_box_extrude_with_bevel(self):
        """Validate compound proportional box extrusion layered with bevel layout styles."""
        with BuildPart() as result:
            # 1. Outer layout boundary block (Blue base)
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                # 2. Inner layout nested block (Green extruded + beveled insert)
                ml(
                    style(
                        width=5,
                        height=5,
                        background_mat=mat.green,
                        extrude=-1,
                    ),
                    # Applies 4-side proportional mask (Left=0.2, others default to 1.0)
                    style.prop_box_extrude(left=0.2),
                    # Applies top beveling configurations
                    style.bevel_box_top(lr=0.5),
                )
            ).build()

        self.assertPart(
            result.part,
            "fd7126019a2fee25a4d855e3681758aea7dbe6023d9a0f2b459a4c3cc90bee79",
            "test_proportional_box_extrude_with_bevel"
        )

    def test_advanced_extrusion_transforms_and_face_retention(self):
        """Verify advanced extrusion behaviors including transformations, draft scales, and non-destructive source face tracking."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                # Target element executing complex dual-extrusion profile sweeps
                ml(
                    style(
                        width=5,
                        height=5,
                        background_mat=mat.green,
                        border_width=0.1,
                        border_mat=mat.yellow,
                        border_radius="10%",
                        
                        # Core Body Extrusion Settings
                        extrude=-2,
                        extrude_transform=Scale(XY=0.5) * Pos(XY=2.5),
                        extrude_delete_source_faces=False,     # Retains native cap geometry at Z=0
                        top_scale=0.5,                         # Applies a linear taper/draft angle
                        
                        # Border Cutout Extrusion Settings
                        border_extrude=-3,
                        border_extrude_delete_source_faces=False,  # Retains boundary profile geometry at Z=0
                        border_extrude_transform=Scale(XY=0.5) * Pos(XY=2.5),
                    ),
                )
            ).build()

        self.assertPart(
            result.part,
            "30ab8e9082ddde27249f69b3d2d9d721c8341dcb1b765eae5e74e91bafe0a815",
            "test_advanced_extrusion_transforms_and_face_retention",
            use_materials=True,
        )

    def test_boolean_union_additive_mode(self):
        """Verify that mode='add' fuses intersecting child geometries into a single flat topological face."""
        with BuildPart() as result:
            ml(
                style(
                    width=6,
                    height=3,
                    background_mat=mat.blue,
                    mode="add",         # Forces 2D Boolean Union merging
                    dissolve=0.0 # to stabilize hash of the result part
                ),
                style.flex_center(),
                
                # 1. Left wing element with rounded corners
                ml(
                    style(width=1, height=0.5, border_radius_left="50%", background_mat=mat.red),
                ),
                
                # 2. Central complex circular assembly (nested booleans)
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style.flex_center(),
                    style(right=0.5),
                    ml(
                        style.circle(radius=0.7, mat=mat.yellow),
                        style.flex_center(),
                        ml(
                            style(height=0.2, width="100%", background_mat=mat.red + mat.Name("red"))
                        )
                    )
                ),
                
                # 3. Right wing element with text engraving baked into the fused shape
                ml(
                    style(width=2, height=1, border_radius_right="50%", background_mat=mat.red),
                    style(right=0.5),
                    style(font_size=0.5, text_align="right", align_y="center", mat=mat.yellow),
                    "Hello"
                )
            ).build()

        self.assertPart(
            result.part,
            "bb93b7603fff72fb0d18c16ba36f29dc0a2c7613f9c1b7ab2d71eccffa5d93c9",
            "test_boolean_union_additive_mode",
            use_materials=True,
        )

    def test_component_reusability_with_bbox_joints(self):
        """Verify that 3D components can be generated once, tagged with joints, and instanced repeatedly."""
        
        def box_component():
            """Generates a closed, 4-segmented 3D hollow loop anchored to its X-axis bounding box joint."""
            return chain(
                chain.twist(
                    ml(
                        style(
                            width=2,
                            height=1,
                            background_mat=mat.red,
                        )
                    ),
                    angle=360,
                    segments=4,
                ),
                side="top",
            ).bbox_joint(Axis.X)

        with BuildPart() as result:
            box = box_component()  # Generate the component asset and its anchor reference
            
            ml(
                style(
                    width=6,
                    height=6,
                    background_mat=mat.blue,
                ),
                style.align_center(),
                
                # Instance 1: Placed at the default layout origin
                ml(box),
                
                # Instance 2: Shifted horizontally via layout flow margins
                ml(
                    style(margin_left=1), 
                    box
                ),
                
                # Instance 3: Shifted and locally rotated 45 degrees along the Z-axis
                ml(
                    style(margin_left=1, transform=Rot(Z=45)), 
                    box
                )
            ).build()

        self.assertPart(
            result.part,
            "fa222a527dccc5ae1da560fa5fb3c312b3ba694dbdb248a273ece9ed1834f77c",
            "test_component_reusability_with_bbox_joints",
            use_materials=True,
        )

    def test_bend_cylindrical_deformation(self):
        """Verify that bend_angle deforms the layer into a semi-cylinder while maintaining child attachments."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=5,
                    background_mat=mat.blue,
                    extrude=0.2,
                ),
                style.align_center(),
                ml(
                    style(
                        width=3,
                        height=3,
                        background_mat=mat.red,
                        bend_angle=180,
                        bend_segments=16,
                        extrude=0.2,
                        extrude_delete_source_faces=False
                    ),
                    style.align_center(),
                    ml(
                        style(
                            width=2,
                            height=1,
                            background_mat=mat.yellow,
                            extrude=0.1
                        )
                    )
                )
            ).build()

        self.assertPart(
            result.part,
            "c079ecb945e3181c856b3fc8ecb26a6749a60291667707b44138cdb1936d90c5",
            "test_bend_cylindrical_deformation",
            use_materials=True,
        )

    def test_curved_hole(self):
        """Verify that ml.hole correctly conforms to a curved surface when high-density cuts are applied to both parent and child."""
        with BuildPart() as result:
            # Step 1: Create a flexible substrate with dense segmentation (cuts) and a 180-degree bend
            ml(
                style(
                    width=3,
                    height=3,
                    background_mat=mat.red,
                    background_cuts=16,
                    bend_angle=180,
                    extrude=0.2,
                    extrude_delete_source_faces=False,
                ),
                style.align_center(),
                # Step 2: Embed a matching flexible hole component that follows the exact curved profile smoothly
                ml.hole(
                    width=2,
                    height=1,
                    mat=mat.yellow,
                    depth=0.2,
                    cuts=16
                ),
            ).build()

        self.assertPart(
            result.part,
            "4abe26743a3fa2bcc00a6e522259a1a5ffd6cbf074408e046e57d0411bc04d21",
            "test_curved_hole",
            use_materials=True,
        )


    def test_tag_subtraction_and_dependent_scaling(self):
        """Verify that subtracting child faces from a parent layout component allows targeted scaling that deforms the connected geometry into a frustum."""
        with BuildPart() as result:
            # Step 1: Build the nested markup layout tracking parent and child entities via tags
            ml(
                style(
                    width=1,
                    height=1,
                    background_mat=mat.red,
                    extrude=1,
                    tag="parent"
                ),
                style.align_center(),
                ml(
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.blue,
                        extrude=1,
                        tag="child",
                        z_index=-1 # to merge shared vertices
                    ),
                )
            ).build(remove_double_verts=True)
            
            # Step 2: Isolate the exclusive parent base faces by removing elements shared with or tagged as child
            parent_base = faces().split().tagged("parent").untagged("child")
            
            # Step 3: Scale the isolated base, pulling the shared vertices to generate a truncated pyramid (frustum)
            transform(parent_base, op=Origin(XY=0.5) * Scale(XY=1.5))

        self.assertPart(
            result.part,
            "3e6c2547b6dc96abe31ceac0827e98bfbfa843c8b82a3a9c1880f4dda50bd689",
            "test_tag_subtraction_and_dependent_scaling",
            use_materials=True,
        )

    def test_evaluation_box_set_overlay_generation(self):
        """Verify that building a layout with evaluate=True correctly generates transparent evaluation bounding boxes over the geometry."""
        with BuildPart() as result:
            # Step 1: Create a closed reference spline path for the background
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)

            # Step 2: Define a complex layout containing absolute positioning, nested borders, and custom transformations
            obj = ml(
                style(
                    width=10,
                    height=6,
                    padding=1,
                    background_mat=mat.green,
                    background_from_curve=bc
                ),
                style.border_ml(
                    style(
                        display="flex",
                        justify_content="space-around",
                    ),
                    [
                        ml(
                            style(
                                width=0.5,
                                height=0.5,
                                background_mat=mat.red,
                                extrude=0.5,
                                x_offset="-50%",
                                y_offset="-100%"
                            )
                        ) for _ in range(3)
                    ],
                ),
                ml(
                    style(
                        position="absolute",
                        left="50%",
                        top="50%",
                        width="70%",
                        height="80%",
                        background_mat=mat.red,
                        border_radius="20%",
                        display="flex",
                        justify_content="space-around",
                        align_items="center",
                        extrude=0.5
                    ),
                    ml(
                        style(
                            width=1.5,
                            height=1,
                            bottom_scale=0.5,
                            background_mat=mat.green,
                            border_mat=mat.yellow,
                            border_width=0.2,
                            border_extrude=2.5,
                            extrude=2,
                        ),
                    ),
                    ml(
                        style(
                            width=1,
                            height=1,
                            background_mat=mat.green,
                            border_mat=mat.yellow,
                            border_width=0.2,
                            border_extrude=1,
                            extrude=-2,
                            transform=Rot(Z=45),
                        ),
                    ),
                    ml.from_part(
                        ml(
                            style(
                                width=0.5,
                                height=0.5,
                                background_mat=mat.green,
                                border_radius="50%",
                                extrude=2,
                                transform=Rot(X=10, Y=-10, Z=10) * Scale(Y=2),
                            ),
                        ),
                        style(transform=Pos(Z=-1) * Rot(X=50, Z=-30) * Scale(0.5))
                    ),
                ),
            )

            # Step 3: Build the standard layout geometry first
            obj.build()

            # Step 4: Build the evaluation BoxSet structure on top and apply a semi-transparent material override
            with BuildPart(mode=Mode.JOIN):
                obj.build(evaluate=True)
                set_mat(mat.blue + mat.PBR(alpha=0.5))

        self.assertPart(
            result.part,
            "67174c72cbe8a846d3727b7d10d4d8db6241837ee8cfb0dce470a4ae6adb69a7",
            "test_evaluation_box_set_overlay_generation",
            use_materials=True,
        )


class TestRuleBasedLayout(BaseCADTest):
    """Test suite verifying rule-based layout solving (RL rules), degree-of-freedom (DOF) parameters, and solver integration."""

    def test_text_generation_and_evaluation_overlay(self):
        """Verify that multi-styled text blocks and strokes layout properly and accept transparent evaluation boxes."""
        with BuildPart() as result:
            # Step 1: Define a complex layout containing stylized text nodes and embedded strokes
            obj = ml(
                style(
                    width=10,
                    padding=0.5,
                    font_size=1,
                    text_align="left",
                    background_mat=mat.blue,
                    mat=mat.red,
                    text_extrude=0.5,
                ),
                "One ",
                # Text node with a custom expanded font size and positive outer stroke extrusion
                ml(
                    style(
                        font_size=1.8,
                        text_stroke_width=0.2,
                        text_stroke_mat=mat.green,
                        text_stroke_extrude=0.25
                    ),
                    "two!"
                ),
                # Text node with an identical font scale but featuring a negative stroke cut extrusion
                ml(
                    style(
                        font_size=1.8,
                        text_stroke_width=0.1,
                        text_stroke_mat=mat.green,
                        text_stroke_extrude=-0.5
                    ),
                    "three..."
                ),
                # Small trailing marker node with tight left-right inner padding bounds
                ml(style(font_size=0.5, padding_lr=0.15), "4"),
            )

            # Step 2: Build the core geometry structure including the extruded text mesh data
            obj.build()
            
            # Step 3: Compute and append the bounding box evaluation overlay using a transparent material shader
            with BuildPart(mode=Mode.JOIN):
                obj.build(evaluate=True)
                set_mat(mat.yellow + mat.PBR(alpha=0.5))

        self.assertPart(
            result.part,
            "794fdb1806aab31ac22695ddda47369db39b9fca0ac9e6e7f30c957366087565",
            "test_text_generation_and_evaluation_overlay",
            use_materials=True,
        )

    def test_gravity_rule_with_positional_dof(self):
        """Verify that absolute positioning with positional DOFs converges using a gravity rule towards a target position."""
        with BuildPart() as result:
            # Define layout container with an absolutely positioned child using positional DOFs
            ml(
                style(
                    width=10,
                    height=5,
                    padding=1,
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        position="absolute",
                        left=ml.dof_p(),
                        top=ml.dof_p(),
                        width=1,
                        height=1,
                        background_mat=mat.red,
                        border_radius="20%",
                        extrude=0.5,
                    ),
                    rl.gravity(Pos(X=8, Y=10)),
                ),
            ).build()

        self.assertPart(
            result.part,
            "748d38777c46fca7acc70c2d280082e2aa75885ab97f339f2613249b11bb42ae",
            "test_gravity_rule_with_positional_dof",
            use_materials=True,
        )

    def test_gravity_rule_parent_propagation(self):
        """Verify that parent-level rules using .on_each() properly propagate down to all child nodes."""
        with BuildPart() as result:
            # Define container with gravity rule applied to all children via .on_each()
            ml(
                style(
                    width=10,
                    height=5,
                    padding=1,
                    background_mat=mat.blue,
                ),
                rl.gravity(Pos(X=8, Y=10)).on_each(),
                ml(
                    style(
                        position="absolute",
                        left=ml.dof_p(),
                        top=ml.dof_p(),
                        width=1,
                        height=1,
                        background_mat=mat.red,
                        border_radius="20%",
                        extrude=0.5,
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "748d38777c46fca7acc70c2d280082e2aa75885ab97f339f2613249b11bb42ae",
            "test_gravity_rule_parent_propagation",
            use_materials=True,
        )

    def test_size_dof_with_center_alignment(self):
        """Verify solver behavior when parent container dimensions (width/height) are bounded DOFs and child target is centered."""
        with BuildPart() as result:
            # Define container with bounded dimension DOFs and centered alignment
            ml(
                style(
                    width=ml.dof(min=1, max=10),
                    height=ml.dof(min=1, max=10),
                    background_mat=mat.blue,
                    align="center",
                    align_y="center",
                ),
                ml(
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.red,
                        border_radius="20%",
                        extrude=0.5,
                    ),
                    # Center of the square should align with target coordinates
                    rl.gravity(Pos(XY=5)),
                ),
            ).build()

        self.assertPart(
            result.part,
            "1d78329a6ac529339a0ea4dcf323cd23c36768e4bc17415582859126876124e9",
            "test_size_dof_with_center_alignment",
            use_materials=True,
        )

    def test_padding_dof_optimization(self):
        """Verify that padding properties can be declared as DOFs and properly solved alongside child layout constraints."""
        with BuildPart() as result:
            # Define layout container with dynamic bounded padding DOFs
            ml(
                style(
                    width=10,
                    height=10,
                    padding=ml.dof(min=1, max=3),
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.red,
                        border_radius="20%",
                        extrude=0.5,
                    ),
                    rl.gravity(Pos(XY=2)),
                ),
            ).build()

        self.assertPart(
            result.part,
            "7f93bba42bd91c31d3692251e9bf96a6cd7b7ef641afb5129fc15323414187aa",
            "test_padding_dof_optimization",
            use_materials=True,
        )

    def test_inter_object_gravity_push_dependencies(self):
        """Verify layout solving across multiple interconnected objects with push-gravity interdependencies."""
        with BuildPart() as result:
            # Define tree structure with reference node assignments and push gravity dependencies
            ml(
                style(
                    width=10,
                    height=5,
                    padding=1,
                    background_mat=mat.blue,
                ),
                obj_1 := ml(
                    style(
                        position="absolute",
                        left="10%",
                        top="100%",
                        width=1,
                        height=1,
                        background_mat=mat.green,
                        border_radius="20%",
                        extrude=0.5,
                    ),
                ),
                obj_2 := ml(
                    style.dof_abs_pos_p(),  # Preset shortcut for absolute positional DOFs
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.red,
                        border_radius="20%",
                        extrude=0.5,
                    ),
                    rl.gravity(obj_1, push=True),
                ),
                ml(
                    style.dof_abs_pos_p(),
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.yellow,
                        border_radius="20%",
                        extrude=0.5,
                    ),
                    rl.gravity(obj_1, push=True),
                    rl.gravity(obj_2, push=True),
                ),
            ).build()

        self.assertPart(
            result.part,
            "a1414825989feed180c18bb71a4f8776c4d7630effe211ced134f6384c3a4b64",
            "test_inter_object_gravity_push_dependencies",
            use_materials=True,
        )

    def test_dof_in_transform_callback(self):
        """Verify that DOFs can be queried within transform callbacks during the build phase."""
        with BuildPart() as result:
            # Define node using ml.dof_get inside transform callback lambda
            ml(
                style(
                    width=10,
                    height=5,
                    padding=1,
                    background_mat=mat.blue,
                ),
                ml(
                    style(
                        position="absolute",
                        width=1,
                        height=1,
                        background_mat=mat.red,
                        border_radius="20%",
                        top_scale=0.2,
                        extrude=0.5,
                        transform=lambda: Pos(
                            X=ml.dof_get(max=10),
                            Y=ml.dof_get(max=5)
                        ),
                    ),
                    rl.gravity(Pos(X=8, Y=3)),
                ),
            ).build()

        self.assertPart(
            result.part,
            "f7a5885658200cdc95cdd4022d8a1e71c2073e60485a25325bb096b2355c4ef8",
            "test_dof_in_transform_callback",
            use_materials=True,
        )

    def test_tag_propagation_and_chained_rule_selectors(self):
        """Verify non-root tag propagation to child nodes and rule evaluation using chained tag selectors."""
        with BuildPart() as result:
            # Step 1: Create tagged tree structure and group rules using chained tag queries
            ml(
                style(
                    width=10,
                    height=5,
                    padding=1,
                    background_mat=mat.blue,
                    tag="all",  # Propagates to all descendants
                ),
                ml(
                    style.dof_abs_pos_p(),
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.red,
                        border_radius="20%",
                        extrude=0.5,
                        root_tag="red",
                    ),
                ),
                ml(
                    style.dof_abs_pos_p(),
                    style(
                        width=1,
                        height=1,
                        background_mat=mat.green,
                        border_radius="20%",
                        extrude=0.5,
                        root_tag="green",
                    ),
                ),
                rl.group(
                    rl.tagged("all").tagged("red")
                    | rl.gravity(rl.tagged("green"), push=True),
                    rl.untagged("red").tagged("green")
                    | rl.gravity(Pos(X=2, Y=2), pull=True),
                ),
            ).build()

        self.assertPart(
            result.part,
            "c1b3247e99cedac6e2224ca67f11bb2c5441992a8a9b7f78b773f73daf1084c5",
            "test_tag_propagation_and_chained_rule_selectors",
            use_materials=True,
        )

    def test_inside_rule_with_curve_boundary_and_evaluation(self):
        """Verify the 'inside' layout rule against custom curve boundaries and visualize bounding evaluation boxes."""
        with BuildPart(mode=Mode.JOIN) as result:
            # Step 1: Define boundary curve path
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)

            # Step 2: Define element constrained to stay inside the defined curve
            ml(
                style(
                    width=5,
                    height=6,
                    background_mat=mat.blue,
                    background_from_curve=bc,
                    border_mat=mat.red,
                    border_width=0.5,
                    extrude=1,
                    top_scale=0.5,
                    transform=Pos(X=2) * Rot(Z=45),
                    show_eval_box=mat.yellow
                ),
                rl.inside().on_each(),
                ml(
                    style.dof_abs_pos_p(),
                    style(
                        anchor_x=0.5,
                        anchor_y=0.5,
                        width=1,
                        height=1,
                        background_mat=mat.green,
                        extrude=-0.5,
                        show_eval_box=mat.red
                    ),
                    rl.gravity(Pos(X=100)),  # Pull outwards to challenge inside constraint
                ),
            ).build()

        self.assertPart(
            result.part,
            "10258e8c412f6f1784a4786b02286dd3772e375448e270a068164d9bc42d6188",
            "test_inside_rule_with_curve_boundary_and_evaluation",
            use_materials=True,
        )

    def test_parent_geometry_deformation_via_scale_dof(self):
        """Verify parent geometry deformation using scale DOFs (e.g. left_scale) to allow inner element optimization towards a target."""
        with BuildPart() as result:
            # Define parent circle with dynamic left-side scaling DOF and child pulling towards bottom-right
            ml(
                style(
                    width=5,
                    height=5,
                    border_radius="50%",
                    background_mat=mat.blue,
                    border_mat=mat.red,
                    border_width=0.5,
                    extrude=1,
                    left_scale=ml.dof(min=0.5, max=1.0),
                ),
                rl.inside().on_each(),
                ml(
                    style.dof_abs_pos_p(),
                    style(
                        anchor_x=0.5,
                        anchor_y=0.5,
                        width=1,
                        height=1,
                        background_mat=mat.green,
                        extrude=-0.5,
                    ),
                    # Strong directional pull forces parent geometry deformation via left_scale optimization
                    rl.gravity(Pos(X=100, Y=100)),
                ),
            ).build()

        self.assertPart(
            result.part,
            "af4ea1c2205057cd9c5891a348f4986146318296f396174f78d759ceb71bab64",
            "test_parent_geometry_deformation_via_scale_dof",
            use_materials=True,
        )

    def test_size_rule_expansion_in_concave_geometry(self):
        """Verify that rl.size pushes a child square element to grow until its boundary hits the parent container's concave walls (created by negative border_radius)."""
        with BuildPart() as result:
            # Step 1: Define parent container with inward-curved concave corners via negative border_radius
            ml(
                style(
                    width=5,
                    height=5,
                    border_radius="-30%",
                    background_mat=mat.blue,
                    border_mat=mat.red,
                    border_width=0.5,
                    extrude=1,
                ),
                rl.inside().on_each(),
                # Step 2: Define centered child node constrained to remain a square (height bound to width)
                ml(
                    style.absolute_center(),
                    style(
                        width=ml.dof_p(),
                        height=lambda n: n.width,  # Maintain 1:1 square aspect ratio based on width DOF
                        background_mat=mat.green,
                        extrude=-0.5,
                    ),
                    # Size rule attempts to expand dimension
                    rl.grow(),
                ),
            ).build()

        self.assertPart(
            result.part,
            "c3c8557666eb91183120094c11da9cf0d184887a938fa8a84819e9eaf3c71377",
            "test_size_rule_expansion_in_concave_geometry",
            use_materials=True,
        )

    def test_border_element_layout_solver_isolated_execution(self):
        """Verify that border elements execute layout solving in an isolated builder with a separate solver, consuming already solved parent curve results."""
        with BuildPart() as result:
            # Step 1: Define a mirrored curve-backed container with embedded border markup (border_ml)
            obj = ml.mirror(
                lambda: ml(
                    style(
                        background_mat=mat.blue,
                        background_from_curve=(
                            c := curve(
                                curve.move_to(Pos(Y=0.7)),
                                curve.move_to(Pos(X=10, Y=3)),
                                curve.step(1, angle=-80),
                                curve.clear_rot(),
                                curve.step(2),
                                curve.move_to(Pos(X=14, Y=4)),
                                curve.step(1),
                                curve.tag("back"),
                                curve.move_to_X(),
                            ).build()
                        ),
                        extrude=0.7,
                        root_tag="curve",
                    ),
                    # Step 2: Attach border layout targeting specific curve segment selector
                    style.border_ml(
                        style(
                            transform=Pos(Z=-0.7 - 0.001) * Rot(Y=-90),
                        ),
                        ml(
                            style(
                                width=1.0,
                                margin_left=0.5,
                                margin_top=0.1,
                                display="flex",
                                justify_content="space-between",
                                evaluate=False,
                            ),
                            [
                                ml(
                                    style(
                                        width=0.3,
                                        height=ml.dof(max=2),
                                        background_mat=mat.red,
                                    ),
                                    # Constrain border elements inside parent curve boundary with 0.1 margin
                                    rl.inside(rl.tagged("curve"), margin=0.1),
                                    # Force element to maximize height DOF within boundary
                                    rl.grow(),
                                )
                                for _ in range(2)
                            ],
                        ),
                        selector=lambda: c.tagged("back"),
                    ),
                ),
                side="top",
                style=style(
                    transform=Pos(Z=-1) * Rot(X=20, Y=20, Z=45)
                ),
            )

            # Step 3: Execute root solver first, then secondary solver for sub-builder border elements
            obj.build()

        self.assertPart(
            result.part,
            "0d91bf00726130210078b3b41153e83f5fa47cb697d87bbff859764b51f57ef2",
            "test_border_element_layout_solver_isolated_execution",
            use_materials=True,
        )


class TestMLCombinedCases(BaseCADTest):
    """
    Category: Complex combined cases.
    These tests intentionally mix layout, text, flex, borders, and 3D features.
    """

    def test_context_styles_and_on_build_callbacks(self):
        """Verify ML context stacking, style composition, and on_build callbacks."""
        with BuildPart() as result:
            # Reusable style helpers.
            align_center = lambda: style(
                align="center",
                align_y="center",
            )
            move_to_center = lambda: style(
                position="absolute",
                left="50%",
                top="50%",
            )
            # Helper for circular blocks with enforced aspect ratio.
            circle = lambda r: (
                style(width=r, height=r)
                + style(border_radius="50%", aspect_ratio=1)
            )

            # Root ML context.
            with ml() as root:
                # Root container styles.
                style(
                    width=10,
                    height=5,
                    padding=1,
                    background_mat=mat.blue,
                )
                align_center()

                # First nested block.
                with ml():
                    style(
                        background_mat=mat.red,
                        extrude=0.1,
                    )
                    circle(r=3)

                    # Nested centered circle.
                    with ml():
                        style(
                            background_mat=mat.yellow,
                        )
                        # Style composition should merge correctly.
                        circle(r=2) + move_to_center()

                        # Build callback for local geometry modification.
                        @ml.on_build
                        def build(ctx: BuildPart):
                            extrude(
                                faces().bottom(),
                                op=Pos(Z=-0.1),
                            )

            # Root-level build callback.
            @root.on_build
            def build(ctx: BuildPart):
                # Add additional topology.
                subdivide(
                    faces().top(),
                    cuts=4,
                )
                # Bend the final geometry.
                bend(
                    angle=30,
                    axis=Axis.Y,
                )

            root.build()

        self.assertPart(
            result.part,
            "1ce7c0c5ecffa2fe213beead3526e405e2ff5280fe26833f31c45b9f6d742676",
            "test_context_styles_and_on_build_callbacks",
            use_materials=True,
        )

    def test_complex_card_with_badge_and_flex_row(self):
        """Verify a card layout that combines border, flex, text, extrude, and absolute badge positioning."""
        with BuildPart() as result:
            with ml() as root:
                style(
                    width=8,
                    height=3,
                    padding=0.75,
                    border_width=0.1,
                    border_style="dashed",
                    border_radius=0.8,
                    border_mat=mat.yellow,
                    background_mat=mat.blue,
                    display="flex",
                    flex_direction="row",
                    justify_content="space-between",
                    align_items="center",
                )
                with ml():
                    style(
                        width=3,
                        font_size=1,
                        text_align="left",
                        text_extrude=0.1,
                        mat=mat.red,
                    )
                    ml("Mr. ")
                    ml("Smith")

                with ml():
                    style(
                        width=2,
                        height=1,
                        margin_right=0.2,
                        border_radius="50%",
                        background_mat=mat.green,
                        extrude=0.1
                    )
                    ml(style(width="100%", height=0.1, position="absolute", top="50%", left="50%", background_mat=mat.red)),
                    ml(style(width=0.1, height="100%", position="absolute", top="50%", left="50%", background_mat=mat.red)),
                    def build(c):
                        faces().group_by(Axis.Z)[-2].mat = mat.red
                        extrude(faces().group_by(Axis.Z, tolerance=1e-4)[0], op=Pos(Z=-0.05))
                    ml.on_build(build)
            root.build()

        self.assertPart(
            result.part,
            "99b0fec1900c563b334bc7ca66fc5d6a119462781a99180bf4e4451c570c74b4",
            "test_complex_card_with_badge_and_flex_row",
            use_materials=True,
        )

    def test_complex_text_panel_with_border_and_extrusion(self):
        """Verify a text-heavy panel with mixed inline styling and extrusion."""
        with BuildPart() as result:
            ml(
                style(
                    width=18,
                    padding=0.5,
                    border_width=0.15,
                    border_style="double",
                    border_radius_tl="50%",
                    border_radius_br="30%",
                    extrude=0.5,
                    background_mat=mat.green,
                    mat=mat.red,
                    font_size=1,
                    line_height=1.2,
                    letter_spacing=0.1,
                    word_spacing=0.3,
                ),
                "Mixing ",
                ml.b("everything", style(font_size=1.3, mat=mat.yellow)),
                " in one ",
                ml(style(background_mat=mat.blue, padding_lr=0.15), "test"),
            ).build()

        self.assertPart(
            result.part,
            "16e268f4aa6a41c4484d0fb94deb597e3c62d1ad7431997e0016b617aafb3726",
            "test_complex_text_panel_with_border_and_extrusion",
            use_materials=True,
        )

    def test_complex_flex_wrapped_dashboard(self):
        """Verify a wrapped dashboard-like layout with mixed child sizes and alignments."""
        with BuildPart() as result:
            ml(
                style(
                    width=20,
                    height=10,
                    padding=1,
                    display="flex",
                    flex_direction="row",
                    flex_wrap="wrap",
                    justify_content="space-around",
                    align_content="space-between",
                    align_items="stretch",
                    gap=0.5,
                    border_width=0.1,
                    border_style="dotted",
                    border_radius=0.5,
                    border_mat=mat.yellow,
                    background_mat=mat.blue,
                ),
                [
                    ml(style(width=3, height=1, background_mat=mat.red)),
                    ml(style(width=2, height=2, background_mat=mat.green)),
                    ml(style(width=4, height=1, background_mat=mat.red)),
                    ml(style(width=2, height=3, background_mat=mat.yellow)),
                    ml(style(width=3, height=1, background_mat=mat.red)),
                    ml(style(width=2, height=2, background_mat=mat.green)),
                ],
            ).build()

        self.assertPart(
            result.part,
            "eb20231d5a437d3b64b6e8f85adf5c1b199e3b699c97651d4ad82608a987736f",
            "test_complex_flex_wrapped_dashboard",
            use_materials=True,
        )

    def test_complex_absolute_overlay_on_text_block(self):
        """Verify an absolute overlay on top of a centered text block."""
        with BuildPart() as result:
            ml(
                style(
                    width=14,
                    height=5,
                    padding=0.5,
                    background_mat=mat.blue,
                    mat=mat.red,
                    font_size=1,
                    align="center",
                    align_y="center",
                ),
                "Overlay",
                ml(
                    style(
                        position="absolute",
                        left="50%",
                        top="50%",
                        width=4,
                        height=1,
                        background_mat=mat.yellow,
                        opacity=0.7,
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "1ffedf8cb9f12dedacc375ea5f0413dce5d3d85b6db035f6301d18d59423189b",
            "test_complex_absolute_overlay_on_text_block",
            use_materials=True,
        )

    def test_complex_alignment_and_text_distribution(self):
        """Verify combined layout engine behavior for structural block alignment and inline text alignment."""
        with BuildPart() as result:
            ml(
                style(
                    width=5,
                    height=6,
                    background_mat=mat.blue,
                    mat=mat.yellow,
                    font_size=1,
                    align="center",
                    align_y="center"
                ),
                # Item 1: Circle with "hello" text using inherited default text alignment
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style(font_size=0.5),
                    "hello"
                ),
                # Item 2: Circle with "world" forced to right-aligned text rendering
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style(font_size=0.5, text_align="right"),
                    "world"
                ),
                # Raw text strings injected into the parent layout flow
                "333", 
                "444",
                # Item 3: Circle with centered multi-axis adjustments for text "!!!"
                ml(
                    style.circle(radius=1, mat=mat.red),
                    style(align_y="center", text_align="center"),
                    "!!!"
                ),
            ).build()

        self.assertPart(
            result.part,
            "bd18c6691827c0a5d3d6f9a29958699e88b7b8e6ea694c141fd296fe346e8513",
            "test_complex_alignment_and_text_distribution",
            use_materials=True,
        )

    def test_complex_multi_layered_shape_composition(self):
        """Verify integration of dynamic sizing, absolute centering, and nested circular overflow clipping."""
        with BuildPart() as result:
            ml(
                style(
                    width=4,
                    height=4,
                    background_mat=mat.yellow,
                    dissolve=0.0 # to stabilize hash of the result part
                ),
                # 1. Background Stripes: 5 blue strips using dynamic lambda sizing expressions
                *[
                    ml(
                        style(
                            width="100%",
                            height=lambda c: (c.parent.height - 0.05) * 0.2 - 0.05,
                            background_mat=mat.blue,
                            margin_top=0.05,
                        ),
                    ) for i in range(5)
                ],
                # 2. Main Centered Artwork Container
                ml(
                    style.align_center(),
                    style.absolute_center(),
                    style(
                        border_mat=mat.yellow,
                        border_width=0.05,
                    ),
                    # Sub-element A: Upper standalone red rectangular block
                    ml(
                        style(
                            width=2,
                            height=0.5,
                            background_mat=mat.red,
                        ),
                    ),
                    # Sub-element B: Large masked circular graphic assembly
                    ml(
                        style.circle(radius=1.5, mat=mat.yellow),
                        style(bottom=0.4),
                        style(overflow="hidden"),  # Clips the internal red rows to a circular arc
                        ml(
                            style(
                                width="100%",
                                height="35%",
                                background_mat=mat.red,
                            ),
                        ),
                        ml(
                            style(
                                width="100%",
                                height="30%",
                                background_mat=mat.red,
                                margin_top=0.05
                            ),
                        ),
                        ml(
                            style(
                                width="100%",
                                height="35%",
                                background_mat=mat.red,
                                margin_top=0.05
                            ),
                        ),
                        # Concentric Targets nested dead-center inside the masked circle
                        ml(
                            style.absolute_center(),
                            style.circle(radius=1, mat=mat.green),
                            style(
                                border_mat=mat.yellow,
                                border_width=0.05,
                            ),
                            ml(
                                style.absolute_center(),
                                style.circle(radius=0.5, mat=mat.blue),
                                style(
                                    border_mat=mat.yellow,
                                    border_width=0.05,
                            ),
                            )
                        )
                    ),
                    # Sub-element C: Lower compound red container with a slightly raised blue sub-strip
                    ml(
                        style(
                            width=2,
                            height=0.5,
                            bottom=0.4,
                            background_mat=mat.red,
                            align="center"
                        ),
                        ml(
                            style(
                                width="50%",
                                height="100%",
                                background_mat=mat.blue,
                            ),
                        ),
                    ),
                ),
            ).build()

        self.assertPart(
            result.part,
            "00ae64d0d99f63e4c35a13200555ba38ffc7d506754fbfee2e814164ff317b28",
            "test_complex_multi_layered_shape_composition",
            use_materials=True,
        )

    def test_scifi_console_station_with_monitors_and_stepped_ladder_access(self):
        """Verify the creation of a hollowed sci-fi console station featuring angled screen extensions and stepped ladder access."""
        with BuildPart() as result:
            # Build the outer housing containing the nested inner cavity, screen arrays, and ladder structure
            (wall_outer := ml(
                style(
                    width=5,
                    height=5,
                    background_mat=mat.blue,
                    border_radius_top="50%",
                    padding=0.2,
                    padding_bottom=0,
                    extrude=1
                ),
                (wall_inner := ml(
                    style(
                        width="100%",
                        height="100%",
                        background_mat=mat.red,
                        border_radius_top="50%",
                        border_radius_segments=30,
                        border_mat=mat.red,
                        padding=0.4,
                        padding_bottom=0,
                        extrude=2
                    ),
                    # Attach the tilted 3-monitor display console array along the bottom edge wire
                    style.border_ml(
                        style(
                            display="flex",
                            justify_content="space-around",
                        ),
                        [
                            ml(
                                style(
                                    width=2.5,
                                    height=1.5,
                                    background_mat=mat.green,
                                    pivot_x="-50%",
                                    y_offset=lambda n: -n.height + wall_inner.style.padding * 0.8,
                                    z_offset="-30%",
                                    transform=Origin(Y=1) * Rot(X=60),
                                    bottom_scale=0.8,
                                    extrude=0.15,
                                    extrude_delete_source_faces=False,
                                ),
                                style.prop_box_extrude(bottom=0.7)
                            ) for _ in range(3)
                        ],
                        selector=lambda: (edges().bottom() - edges().max_y()).to_wire(),
                    ),
                    # Create the core room cutout and the stepped multi-tiered entry ladder
                    ml(
                        style(
                            width="100%",
                            height="100%",
                            background_mat=mat.yellow,
                            border_radius_top="50%",
                            top=1e-2,
                            extrude=lambda: -(wall_outer.style.extrude + wall_inner.style.extrude),
                        ),
                        style.extrude_delete_face(side_bottom=True),
                        # Ladder Tier 3 (Highest platform step)
                        ml(
                            style(
                                width="100%",
                                height="70%",
                                background_mat=mat.yellow,
                                border_radius_top="50%",
                                extrude=0.25 * 3,
                            ),
                        ),
                        # Ladder Tier 2 (Intermediate platform step)
                        ml(
                            style(
                                width="100%",
                                height="15%",
                                background_mat=mat.yellow,  
                                extrude=0.25 * 2
                            ),
                        ),
                        # Ladder Tier 1 (Lowest entrance step)
                        ml(
                            style(
                                width="100%",
                                height="15%",
                                background_mat=mat.yellow,
                                extrude=0.25 * 1
                            ),
                        ),
                    )
                ))
            )).build()

        self.assertPart(
            result.part,
            "c844ac57da0066f9649a4c4fdded55182c2f037d60e4aafea6589645a7019cf6",
            "test_scifi_console_station_with_monitors_and_stepped_ladder_access",
            use_materials=True,
        )
