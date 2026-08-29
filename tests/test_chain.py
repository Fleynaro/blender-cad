from blender_cad import *
from tests.test_base import BaseCADTest


class TestChain(BaseCADTest):
    """
    Category: Chain
    Tests for the chain operation.
    """

    def test_chain_operation(self):
        """Verify the chain operation sequentially combining elements with rotations and material assignments."""
        with BuildPart() as result:
            # Perform the flow chain operation to sequentially align and connect objects
            chain(
                # 1. Base multilayer element (Red material)
                ml(
                    style(
                        width=2,
                        height=1,
                        background_mat=mat.red,
                    )
                ),
                # 2. Rotate the coordinate system for the next element
                Rot(X=90),
                # 3. Second multilayer element (Blue material)
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.blue,
                    )
                ),
                # 4. Rotate again to change the direction of the chain
                90,
                # 5. Third multilayer element (Red material)
                ml(
                    style(
                        width=2,
                        height=1,
                        background_mat=mat.red,
                    )
                ),
                # 6. Rotate for the final element
                90,
                # 7. Fourth multilayer element (Blue material)
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.blue,
                    )
                ),
                # Specify the flow direction axis for alignment
                axis=-Axis.Y,
                rot_axis=Axis.X,
            ).build()

        self.assertPart(
            result.part,
            "53257b056cf092a2e67e21db8674965ad5776715ede6175e4719ab11951b668a",
            "test_chain_operation",
            use_materials=True,
        )

    def test_chained_layout_branching_and_forks(self):
        """Verify that chain branch sprouts side-glances/forks off the main 3D folding sequence."""
        with BuildPart() as result:
            chain(
                # Base Segment 1: Red root plane
                ml(
                    style(
                        width=2,
                        height=1,
                        background_mat=mat.red,
                    )
                ),
                # Fork 1: Green branch sprouting to the right (+X) rotated 90° on Y
                chain(
                    Rot(Y=90),
                    ml(
                        style(
                            width=2,
                            height=1,
                            background_mat=mat.green,
                        )
                    ),
                    axis=Axis.X,
                ),
                # Fork 2: Complex nested yellow branches splitting off to the left (-X)
                chain(
                    chain(
                        Rot(Y=-45),
                        ml(
                            style(
                                width=1,
                                height=1,
                                background_mat=mat.yellow,
                            )
                        ),
                        axis=Axis.X,
                    ),
                    Rot(Y=-90),
                    ml(
                        style(
                            width=1,
                            height=1,
                            background_mat=mat.yellow,
                        )
                    ),
                    axis=-Axis.X,
                ),
                # Main Sequence continues: Fold 90° on X into a blue plate
                Rot(X=90),
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.blue,
                    )
                ),
                # Main Sequence: Fold another 90° on X into a red plate
                Rot(X=90),
                ml(
                    style(
                        width=2,
                        height=1,
                        background_mat=mat.red,
                    )
                ),
                # Main Sequence End: A final blue plate attached to a list wrapper
                [
                    Rot(X=90),
                    ml(
                        style(
                            width=2,
                            height=2,
                            background_mat=mat.blue,
                        )
                    ),
                ],
                # Global chain overrides for attachment direction and boolean strategy
                axis=-Axis.Y,
            ).build()

        self.assertPart(
            result.part,
            "0265f66376d520fe529bab5a85f60313be836c2ff021b5326441bd38a6c7fc87",
            "test_chained_layout_branching_and_forks",
            use_materials=True,
        )

    def test_chained_layout_folding_3d_box(self):
        """Verify that ml.chain sequence rotates and hinges consecutive segments into a 3D box structure."""
        with BuildPart() as result:
            chain(
                # Segment 1: Red wall with a tapered bottom edge
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.red,
                        bottom_scale=0.5,
                    )
                ),
                90,  # Fold 90 degrees upward to start the box walls
                # Segment 2: Blue wall with an embossed green center square
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.blue,
                    ),
                    style.align_center(),
                    ml(
                        style(width=1, height=1, background_mat=mat.green, extrude=0.1),
                    ),
                ),
                90,  # Fold another 90 degrees (forming the top/opposite face)
                # Segment 3: Simple flat red wall
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.red,
                    )
                ),
                90,  # Final 90-degree fold to complete the 4-sided loop
                # Segment 4: Blue wall with a tapered top and an engraved green pocket
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.blue,
                        top_scale=0.5,
                    ),
                    style.align_center(),
                    ml(
                        style(
                            width=1,
                            height=1,
                            background_mat=mat.green,
                            extrude=-0.1,
                            top_scale=0.5,
                        ),
                    ),
                ),
                side="top",
            ).build()

        self.assertPart(
            result.part,
            "65797c0fd93c5a6b0f0e5efad581e395cf07c6bf20b08e0454b1098e92763905",
            "test_chained_layout_folding_3d_box",
            use_materials=True,
        )

    def test_chain_automatic_dimension_propagation_to_wall(self):
        """Verify that item_width and item_height from chain automatically propagate down to nested wall components."""
        with BuildPart() as result:
            # Step 1: Define a generic wall component without explicit width or height
            wall = ml(
                style(
                    background_mat=mat.red,
                )
            )

            # Step 2: Build a chain where dimensions are passed down implicitly to each wall instance
            chain(
                wall, Rot(Y=30), wall, axis=Axis.X, item_width=2, item_height=2
            ).build()

        self.assertPart(
            result.part,
            "5ffbd28f71fc6ca1fe0f4260f26502415d456cecd4b286e4fb89aca78f11d87a",
            "test_chain_automatic_dimension_propagation_to_wall",
            use_materials=True,
        )

    def test_twist_chain_operation(self):
        """Verify the operation by wrapping multiple layers around an axis with a specified angle and segments."""
        with BuildPart() as result:
            # Perform the twist chain operation to deform and segment the multilayer elements
            chain(
                chain.twist(
                    # 1. First multilayer element (Red material)
                    ml(
                        style(
                            width=2,
                            height=1,
                            background_mat=mat.red,
                        )
                    ),
                    # 2. Second multilayer element (Blue material)
                    ml(
                        style(
                            width=2,
                            height=1,
                            background_mat=mat.blue,
                        )
                    ),
                    axis=Axis.X,
                    angle=360,
                    segments=5,
                ),
                axis=Axis.Y,
            ).build()

        self.assertPart(
            result.part,
            "69ba82d4f0b74c08e590e1b3e2fcd70e79decc5acf1e838c7bc63c69d7de409d",
            "test_twist_chain_operation",
            use_materials=True,
        )

    def test_twist_chain_with_radial_branching(self):
        """Verify that chain branch nodes correctly inherit radial offsets and tilt within a twist context."""
        with BuildPart() as result:
            chain(
                chain.twist(
                    # Segment 1: Red base tile
                    ml(
                        style(
                            width=1,
                            height=1,
                            background_mat=mat.red,
                        )
                    ),
                    # Radial Fork 1: Green branch sprouting outward along +X from the first curved segment
                    chain(
                        Rot(Y=0),
                        ml(
                            style(
                                width=1,
                                height=1,
                                background_mat=mat.green,
                            )
                        ),
                        axis=Axis.X,
                    ),
                    # Segment 2: Blue intermediate tile
                    ml(
                        style(
                            width=1,
                            height=1,
                            background_mat=mat.blue,
                        )
                    ),
                    # Radial Fork 2: Yellow branch sprouting outward along +X from the subsequent curved segment
                    chain(
                        Rot(Y=0),
                        ml(
                            style(
                                width=1,
                                height=1,
                                background_mat=mat.yellow,
                            )
                        ),
                        axis=Axis.X,
                    ),
                    axis=Axis.X,
                    angle=360,
                    segments=5,
                ),
                axis=Axis.Y,
            ).build()

        self.assertPart(
            result.part,
            "58c0fea3102fbb15f5f37e39691ce473c219bd2f76419779858c518800f4dcf5",
            "test_twist_chain_with_radial_branching",
            use_materials=True,
        )

    def test_chain_twist_with_lambda_cap_generation_and_deformation(self):
        """Verify that a conditional ml.chain cap generated via lambda correctly attaches to a twisted sequence and stretches the entire body when scaled."""
        with BuildPart() as result:
            # Step 1: Build a twisted chain using a lambda function to conditionally add a capped end
            chain(
                chain.twist(
                    lambda i: [
                        ml(
                            style(
                                width=2,
                                height=2,
                                background_mat=mat.red,
                            )
                        ),
                        # Dynamically inject the nested cap structure only at the starting index (i == 0)
                        chain(
                            Rot(Y=90),
                            ml(
                                style(
                                    background_mat=mat.blue,
                                ),
                                ml(
                                    style.circle(radius=0.5, mat=mat.yellow),
                                    style.absolute_center(),
                                    style(extrude=1),
                                ),
                            ),
                            axis=-Axis.X,
                            clip_by_parent=True,
                            tag="cap",
                        )
                        if i == 0
                        else None,
                    ],
                    angle=360,
                    segments=5,
                ),
                axis=Axis.Y,
                rot_axis=Axis.X,
                transform=Rot(Y=90),
            ).build()

            # Step 2: Scale the tagged cap geometry
            # Due to shared topology, the entire twisted structure deforms along with the cap, creating a frustum-like effect
            transform(faces().split().tagged("cap"), op=Origin(0.5) * Scale(XY=0.5))

        self.assertPart(
            result.part,
            "e1ad16e59a2ae8b73df4ecc033508a60a899503f94a050db2e4e9d9db69eef93",
            "test_chain_twist_with_lambda_cap_generation_and_deformation",
            use_materials=True,
        )

    def test_chain_bend_sequences(self):
        """Verify complex chaining with multiple bend operations across different axes."""
        with BuildPart() as result:
            # Define a reusable straight road element
            road = ml(
                style(
                    width=2,
                    height=2,
                    background_mat=mat.red,
                )
            )

            # Construct a continuous chain using straight segments and bends
            chain(
                road,
                # First bend: 90 degrees around Z axis (Blue)
                chain.bend(angle=90, axis=Axis.Z, segments=5),
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.blue,
                    )
                ),
                road,
                # Second bend: -180 degrees around Z axis with wider width (Green)
                chain.bend(angle=-180, axis=Axis.Z, segments=10),
                ml(
                    style(
                        width=4,
                        height=2,
                        background_mat=mat.green,
                    )
                ),
                # Third bend: 180 degrees counter-bend around Z axis (Green)
                chain.bend(angle=180, axis=Axis.Z, segments=10),
                ml(
                    style(
                        width=4,
                        height=2,
                        background_mat=mat.green,
                    )
                ),
                road,
                # Fourth bend: 90 degrees around Z axis (Blue)
                chain.bend(angle=90, axis=Axis.Z, segments=5),
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.blue,
                    )
                ),
                road,
                # Fifth bend: 20 degrees vertical incline around Y axis (Yellow)
                chain.bend(angle=20, axis=Axis.Y, segments=5),
                ml(
                    style(
                        width=2,
                        height=2,
                        background_mat=mat.yellow,
                    )
                ),
                road,
                # Set the primary chaining direction along the X axis
                axis=Axis.X,
            ).build()

        # Verify the final geometry hash to ensure all bends deformed and aligned correctly
        self.assertPart(
            result.part,
            "9587c8b9888e883dab1eb0a361ba562091311d948daed839a3315b17b70d1188",
            "test_chain_bend_sequences",
            use_materials=True,
        )

    def test_nested_chain_bends(self):
        """Verify nested bend operations deforming geometry along multiple axes simultaneously."""
        with BuildPart() as result:
            # Define a reusable straight road element
            road = ml(
                style(
                    width=2,
                    height=2,
                    background_mat=mat.red,
                )
            )

            # Construct a chain with a nested bend to test multi-axis deformation
            chain(
                road,
                # Inner bend: First deforms the flat style 50 degrees around Z axis
                chain.bend(angle=50, axis=Axis.Z, segments=5),
                # Outer bend: Deforms the already bent geometry -10 degrees around X axis
                chain.bend(angle=-10, axis=Axis.X, segments=5),
                ml(
                    style(
                        width=2,
                        height=4,
                        background_mat=mat.blue,
                    )
                ),
                road,
                # Set the primary chaining direction along the negative Y axis
                axis=-Axis.Y,
            ).build()

        # Verify the final geometry hash to ensure nested deformations aligned correctly
        self.assertPart(
            result.part,
            "f3586e8ea5621dff8747b4f521bdd1f4ab844740160f02df68a635ddc7ea480f",
            "test_nested_chain_bends",
            use_materials=True,
        )

    def test_chain_bend_loop(self):
        """Verify repetitive bend operations within an ml.chain list multiplication."""
        with BuildPart() as result:
            # Construct a looping chain by repeating a pattern of a straight road and a 90-degree bend 4 times
            chain(
                [
                    # Straight segment (Red)
                    ml(
                        style(
                            width=2,
                            height=2,
                            background_mat=mat.red,
                        )
                    ),
                    # 90-degree bend segment (Blue)
                    chain.bend(angle=90, segments=5),
                    ml(
                        style(
                            width=2,
                            height=2,
                            background_mat=mat.blue,
                        ),
                    ),
                ]
                * 4,
                side="top",
            ).build()

        # Verify the final closed or looping geometry hash to ensure correct transformation accumulation
        self.assertPart(
            result.part,
            "698563a83bc393a696227ea3616eef81551d8ab181a6182695bfa9eb812a7663",
            "test_chain_bend_loop",
            use_materials=True,
        )

    def test_chain_bend_on_solid_3d_corridor_with_windows(self):
        """Verify that chain.bend correctly deforms complex 3D solid corridor geometries containing window inserts."""
        with BuildPart() as result:
            # Step 1: Define a plain solid wall component
            wall = ml(
                style(
                    background_mat=mat.red,
                )
            )

            # Step 2: Define a composite wall component featuring an embedded yellow window panel
            wall_with_window = ml(
                style(
                    background_mat=mat.red,
                ),
                style.align_center(),
                ml(
                    style(
                        background_mat=mat.yellow,
                        width="50%",
                        height="50%",
                        extrude=0.05,
                    )
                ),
            )

            # Step 3: Assemble a single hollow 3D corridor segment profile using sub-chains
            corridor = chain(
                wall,
                90,
                wall_with_window,
                90,
                wall,
                90,
                wall_with_window,
                axis=Axis.X,
                rot_axis=Axis.Y,
                item_width=2,
                item_height=4,
            ).part

            # Step 4: String multiple corridor segments together, applying sequential 3D bend operations
            chain(
                corridor,
                corridor,
                # Deform the third section with a positive 90-degree curve
                chain.bend(angle=90, segments=5),
                corridor,
                corridor,
                # Deform the fifth section with a negative 45-degree curve
                chain.bend(angle=-45, segments=5),
                corridor,
                axis=Axis.Y,
                rot_axis=Axis.Z,
            ).build()

        self.assertPart(
            result.part,
            "ed57e0382df7152d9c7928fc8d66b6469b2bbb5ee01bdbfd3c6a9c08d126d09a",
            "test_chain_bend_on_solid_3d_corridor_with_windows",
            use_materials=True,
        )

    def test_chain_t_junction_corridor_branching_and_attachments(self):
        """Verify that a complex T-junction corridor layout branches correctly using conditional sub-chains and specific joint attachments."""
        with BuildPart() as result:
            # Step 1: Define a standard twisted corridor segment along the X-axis
            corridor = chain(
                chain.twist(
                    ml(
                        style(
                            background_mat=mat.red,
                        )
                    ),
                    angle=360,
                    segments=4,
                ),
                axis=Axis.X,
                rot_axis=Axis.Y,
                item_width=2,
                item_height=4,
            ).part

            # Step 2: Define a T-junction corridor segment that conditionally opens up a side path at the second index
            corridor_T = chain(
                chain.twist(
                    lambda i: [
                        ml(
                            style(
                                background_mat=mat.blue,
                            ),
                            style.align_center(),
                            ml(
                                style(
                                    background_mat=mat.red,
                                    width=2,
                                    height="100%",
                                    extrude=-1,
                                    subtract=True,
                                )
                            )
                            if i == 1
                            else None,
                        ),
                    ],
                    angle=360,
                    segments=4,
                ),
                axis=-Axis.Y,
                rot_axis=Axis.X,
                item_width=6,
                item_height=2,
            ).part

            # Step 3: Construct the main linear path and split into two distinct branches at the T-junction
            chain(
                corridor,
                chain.bend(angle=90, segments=5),
                corridor,
                corridor_T,
                # Branch 1: Attach to the positive X-axis joint of the T-junction and continue with a curved path
                chain(
                    chain.attach(to_joint=Axis.X),
                    corridor,
                    chain.bend(angle=90, segments=5),
                    corridor,
                    corridor,
                ),
                # Branch 2: Attach to the negative X-axis joint with a 180-degree twist and break off in the opposite direction
                chain(
                    chain.attach(to_joint=-Axis.X, twist=180),
                    chain.bend(angle=-90, segments=5),
                    corridor,
                    corridor,
                ),
                axis=Axis.Y,
                rot_axis=Axis.Z,
            ).build()

        self.assertPart(
            result.part,
            [
                "e926123061dab3cad38c1d421b447272ec17ca214a4e4424077be17010ad496f",
                "f54dafd6ccadbd5b75081086bda463ba020cdb1ddb8d77542ec67fcc36232a16",
            ],
            "test_chain_t_junction_corridor_branching_and_attachments",
            use_materials=True,
        )

    def test_chain_clip_by_parent_and_automatic_dimension_matching(self):
        """Verify that nested child chains automatically inherit dimensions and are correctly clipped by their parent components."""
        with BuildPart() as result:
            # Step 1: Define a generic base wall layout component
            wall = ml(
                style(
                    background_mat=mat.red,
                )
            )

            # Step 2: Build the first complex layout combining multiple nested child chains with parent clipping
            chain(
                wall,
                # Child chain automatically adopts parent dimensions and clips to the parent geometry context
                chain(Rot(X=90), wall, axis=Axis.Y, clip_by_parent=True),
                # Child chain clips using an explicit SUBTRACT boolean mode against the parent
                chain(Rot(X=-90), wall, axis=-Axis.Y, clip_by_parent=Mode.SUBTRACT),
                Rot(Y=120),
                wall,
                Rot(Y=120),
                wall,
                axis=-Axis.X,
                item_width=2,
                item_height=2,
            ).build()

            # Step 3: Verify the same clipping and auto-dimension scaling behavior within a shifted spatial location
            with Locations(Pos(X=5)):
                chain(
                    wall,
                    # Dimension-less nested chain automatically matches the item_width/item_height of this parent block
                    chain(Rot(Y=-90), wall, axis=Axis.X, clip_by_parent=True),
                    chain(Rot(Y=90), wall, axis=-Axis.X, clip_by_parent=Mode.SUBTRACT),
                    Rot(X=120),
                    wall,
                    Rot(X=120),
                    wall,
                    axis=Axis.Y,
                    item_width=1,
                    item_height=3,
                ).build()

        self.assertPart(
            result.part,
            "1b008905a5df77d2db474e05013077f9951adbc5de4e9b308c6ed6e05ad1fdec",
            "test_chain_clip_by_parent_and_automatic_dimension_matching",
        )

    def test_geometry_deformation_by_tag_with_shared_vertices(self):
        """Verify that scaling a tagged middle section deforms adjacent connected sections due to shared vertices."""
        with BuildPart() as result:
            # Step 1: Define a reusable factory function for twisted wall segments with custom tagging
            walls = lambda tag: (
                chain(
                    chain.twist(
                        ml(
                            style(
                                width=2,
                                height=2,
                                background_mat=mat.red,
                            )
                        ),
                        axis=Axis.X,
                        angle=360,
                        segments=5,
                    ),
                    axis=Axis.Y,
                    transform=Rot(Y=90),
                    tag=tag,
                ).part
            )

            # Step 2: Stack three sequential wall segments along the Z-axis, tagging each distinctly
            chain(
                walls("walls_1"),
                walls("walls_2"),
                walls("walls_3"),
                axis=Axis.Z,
            ).build()

            # Step 3: Scale only the middle segment ('walls_2')
            # The adjacent segments ('walls_1' and 'walls_3') must stretch and deform because they share topology
            transform(faces().split().tagged("walls_2"), op=Origin(0.5) * Scale(XY=0.5))

            # Step 4: Extract part for top wall ('walls_3') and shift it to the right
            add(faces().split().tagged("walls_3").part(), offset=Pos(X=5))

        self.assertPart(
            result.part,
            "b8225c832e4b39bd5c266cc33dc0a434cd28fc4b0d497b04fa6c8818f8bb311a",
            "test_geometry_deformation_by_tag_with_shared_vertices",
            use_materials=True,
        )
