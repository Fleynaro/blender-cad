from blender_cad import *
from tests.test_base import BaseCADTest


class TestRuleBasedLayoutSolver(BaseCADTest):
    """
    Category: Rule-Based Layout Solver constraints & interactions.
    Covers gravity, transforms, rule grouping, priorities, curve boundaries,
    look-at/along orientation, stack rules, DOF configuration and more.
    """

    box_style = style(width=1, height=1, x_offset="-50%", y_offset="-50%")

    def test_basic_gravity_rule(self):
        """Verify basic application of the gravity rule pushing an object towards a target point."""
        with BuildPart() as result:
            (
                ml(
                    self.box_style,
                    style(background_mat=mat.red),
                )
                | rl.gravity(Pos(X=10, Y=5))
            ).resolve()

        self.assertPart(
            result.part,
            "7c522fa3949515219ad6620f38ac4b60cecc9bbf5c50a164d8de1dd59ed3dbc1",
            "test_basic_gravity_rule",
            use_materials=True,
        )

    def test_combined_gravity_and_transform_rule(self):
        """Verify combining gravity with fixed transform overrides (e.g., locking Y coordinate)."""
        with BuildPart() as result:
            (
                ml(
                    self.box_style,
                    style(background_mat=mat.red),
                )
                | rl.gravity(Pos(X=10, Y=5))
                | rl.transform(y=3.0, rz=45)
            ).resolve()

        self.assertPart(
            result.part,
            "24ecd4c5100d4630b98b598d07c3ef68caf56abb9e34ff3051788630d0df997e",
            "test_combined_gravity_and_transform_rule",
            use_materials=True,
        )

    def test_initial_transform_and_gravity_resolution(self):
        """Verify that an initial transform sets the starting state before gravity pulls the component to its final resolved position."""
        with BuildPart() as result:
            (
                rl.group(
                    ml(
                        self.box_style,
                        style(background_mat=mat.red),
                    )
                    # Set the initial spatial state prior to simulation physics
                    | rl.transform(x=-5, y=-5, rz=45, init=True)
                    # Apply a localized gravity source to attract the component to its final destination
                    | rl.gravity(Pos(X=5, Y=5))
                )
            ).resolve()

        self.assertPart(
            result.part,
            "cc99ab8c0a86274a1821ff35e12083df26094fcd9fde462c5eda34233c1d2f4d",
            "test_initial_transform_and_gravity_resolution",
            use_materials=True,
        )

    def test_rule_group_application(self):
        """Verify aggregating multiple rules into a RuleGroup and applying them together."""
        rules = rl.gravity(Pos(X=5)) | rl.transform(rz=45, sx=2)
        with BuildPart() as result:
            (
                ml(
                    self.box_style,
                    style(background_mat=mat.red),
                )
                | rules
            ).resolve()

        self.assertPart(
            result.part,
            "30970eb40b5c873d05671478a1aa3d9aeabfaeccfb3fa714d1af8b4d6b522e59",
            "test_rule_group_application",
            use_materials=True,
        )

    def test_rule_priority_weighting(self):
        """Verify solver behavior when competing rules use custom priority weights."""
        with BuildPart() as result:
            (
                ml(
                    self.box_style,
                    style(background_mat=mat.red),
                )
                | rl.gravity(Pos(X=-5))
                | rl.gravity(Pos(X=5)).with_priority(2.0)
            ).resolve()

        self.assertPart(
            result.part,
            "d66ff0a736db68b957838ccf4dfba672742e52018e7e1ace415ea3af646c2452",
            "test_rule_priority_weighting",
            use_materials=True,
        )

    def test_group_layout_multiple_elements(self):
        """Verify creating a group with distinct rules per layout element."""
        with BuildPart() as result:
            rl.group(
                ml(
                    self.box_style,
                    style(background_mat=mat.red),
                )
                | rl.gravity(Pos(X=10, Y=5)),
                ml(
                    self.box_style,
                    style(background_mat=mat.blue),
                )
                | rl.gravity(Pos(X=-5, Y=10)),
            ).resolve()

        self.assertPart(
            result.part,
            "3d1f89721687e3dc9a0fb1b19eec94ade71b979a7397339539b881502d48524f",
            "test_group_layout_multiple_elements",
            use_materials=True,
        )

    def test_group_rules_applied_on_self(self):
        """Verify applying transform and gravity rules directly to the group container space via on_self()."""
        with BuildPart() as result:
            (
                rl.group(
                    ml(
                        self.box_style,
                        style(background_mat=mat.red),
                    )
                    | rl.transform(x=-1),
                    ml(
                        self.box_style,
                        style(background_mat=mat.blue),
                    )
                    | rl.transform(x=1),
                )
                | rl.transform(rz=45).on_self()
                | rl.gravity(Pos(X=5, Y=5)).on_self()
            ).resolve()

        self.assertPart(
            result.part,
            "77424c9cb245fdb011536d8096994de6d9195ac5ace1a88b1d3861a8a701fc96",
            "test_group_rules_applied_on_self",
            use_materials=True,
        )

    def test_group_rules_applied_on_each_child(self):
        """Verify broadcasting transform rules across all group elements individually (default scope)."""
        with BuildPart() as result:
            (
                rl.group(
                    ml(
                        self.box_style,
                        style(background_mat=mat.red),
                    )
                    | rl.gravity(Pos(X=1, Y=1)),
                    ml(
                        self.box_style,
                        style(background_mat=mat.blue),
                    )
                    | rl.gravity(Pos(X=3, Y=1)),
                )
                | rl.transform(rz=45)
            ).resolve()

        self.assertPart(
            result.part,
            "f926d961795cbd208c7ddc91f423c63cc516353dedb4d5a0dff4768768313bd9",
            "test_group_rules_applied_on_each_child",
            use_materials=True,
        )

    def test_dynamic_lambda_target_gravity(self):
        """Verify setting dynamic target points in gravity using lambda references to other layout objects."""
        with BuildPart() as result:
            rl.group(
                obj_1 := rl.object(
                    ml(
                        self.box_style,
                        style(background_mat=mat.red),
                    )
                ),
                obj_2 := rl.object(
                    ml(
                        self.box_style,
                        style(background_mat=mat.blue),
                    )
                ),
                obj_1 | rl.gravity(Pos(X=10, Y=5)),
                obj_2 | rl.gravity(lambda: Pos(rl.get_position(obj_1)) * Pos(Y=1)),
            ).resolve()

        self.assertPart(
            result.part,
            "3eac7aa4cb0f632daf6737ea0ca0ce991162e8d52a032ab377126dcb1b170a0f",
            "test_dynamic_lambda_target_gravity",
            use_materials=True,
        )

    def test_size_constraints_and_dynamic_evaluation(self):
        """Verify that components correctly handle static sizing, dynamic functions, and soft size limits."""
        box_style = style(width=1, height=1, x_offset="-50%", y_offset="-50%")

        with BuildPart() as result:
            rl.group(
                # Step 1: Define a red object and explicitly override its initial base width dimension
                red := ml(
                    box_style,
                    style(background_mat=mat.red),
                )
                | rl.transform(y=0)
                | rl.size(x=2, init=True),
                # Step 2: Define a blue object whose size evaluates dynamically based on the red element's bounds
                ml(
                    box_style,
                    style(background_mat=mat.blue),
                )
                | rl.transform(y=2)
                | rl.size(x=lambda: rl.get_size(red).x * 2),
                # Step 3: Define a green object with a soft size constraint allowing flexibility during collision tracking
                ml(
                    box_style,
                    style(background_mat=mat.green),
                )
                | rl.transform(y=4)
                | rl.size(x=6, soft=True),
            ).resolve()

        self.assertPart(
            result.part,
            "8800bc84d3f765a69284d5bd03b7fd132625269cc9dc8a218b1e9189884e7000",
            "test_size_constraints_and_dynamic_evaluation",
            use_materials=True,
        )

    def test_outside_constraint_and_gravity_resolution(self):
        """Verify that the outside constraint prevents geometry overlap during gravity resolution between two objects."""
        with BuildPart() as result:
            # Step 1: Define two distinct layout objects with specific materials and initial extrusions
            obj_a = rl.object(
                ml(
                    self.box_style,
                    style(background_mat=mat.red, transform=Scale(0.5) * Pos(Z=-0.05)),
                )
            )
            obj_b = rl.object(
                ml(self.box_style, style(background_mat=mat.blue, extrude=0.1))
            )

            # Step 2: Resolve the physics group where mutual gravity pulls them together, bounded by the outside rule
            (
                rl.group(
                    # Object A starts transformed and is strictly constrained to remain outside Object B's boundary
                    obj_a
                    | rl.gravity(obj_b)
                    | rl.transform(x=-5.0, rz=45.0, init=True)
                    | rl.outside(obj_b),
                    # Object B starts on the opposite side and gravitates toward Object A
                    obj_b | rl.gravity(obj_a) | rl.transform(x=5.0, init=True),
                )
            ).resolve(mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "9c7a45f88ca31f9380095f1d6101894c8deb5631132fde6736808a868c0ed267",
            "test_outside_constraint_and_gravity_resolution",
            use_materials=True,
        )

    def test_inside_curve_boundary_constraint(self):
        """Verify that the inside constraint restricts an object's simulated movement strictly within a curve profile."""
        with BuildPart() as result:
            # Step 1: Define a closed boundary path using a transformed spline curve
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)
                bc.curve.transform = Pos(X=2) * Rot(Z=45)
            add(bc)

            # Step 2: Resolve layout simulation keeping the red block strictly inside the curve boundary
            (
                rl.group(
                    ml(
                        self.box_style,
                        style(background_mat=mat.red),
                    )
                    | rl.transform(rz=60.0, init=True)
                    | rl.gravity(Pos(X=10, Y=10)),
                )
                # Apply the containment rule to the entire physics resolution group
                | rl.inside(bc)
            ).resolve(mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "5f7b7f861846016449de0456343b15228c65f47a60be2cad12ef2a6b3766960b",
            "test_inside_curve_boundary_constraint",
            use_materials=True,
        )

    def test_inside_curve_constraint_multiple_points(self):
        """Verify the constraint enforcing multiple point-like objects to stay bounded inside a transformed curve boundary."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)
                bc.curve.transform = Pos(X=2) * Rot(Z=45)
            add(bc)

            (
                rl.group(
                    ml(
                        self.box_style,
                        style(background_mat=mat.red),
                    )
                    | rl.gravity(Pos(X=10, Y=5)),
                    ml(
                        self.box_style,
                        style(background_mat=mat.blue),
                    )
                    | rl.gravity(Pos(X=5, Y=10)),
                    ml(
                        self.box_style,
                        style(background_mat=mat.green),
                    )
                    | rl.gravity(Pos(X=-10, Y=5)),
                    ml(
                        self.box_style,
                        style(background_mat=mat.yellow),
                    )
                    | rl.gravity(Pos(X=5, Y=-10)),
                )
                | rl.inside(
                    bc, shell_override=rl.point()
                )  # make collision shell a point
            ).resolve(mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "e44f73790a8c07a2952fd752568b82a01bfe6f45a37bfa270ce639d6b84d0f3a",
            "test_inside_curve_constraint_multiple_points",
            use_materials=True,
        )

    def test_at_curve_constraint(self):
        """Verify at_curve constraint pinning object parameters strictly onto 1D parametric curve coordinates."""
        with BuildPart() as result:
            with BuildCurve() as bc:
                Spline((0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), close=True)
                bc.curve.transform = Pos(X=4, Y=-1) * Rot(Z=45) * Scale(X=0.8)
            add(bc)

            (
                rl.group(
                    ml(
                        self.box_style,
                        style(background_mat=mat.red),
                    )
                    | rl.gravity(Pos(X=10, Y=5)),
                    ml(
                        self.box_style,
                        style(background_mat=mat.blue),
                    )
                    | rl.gravity(Pos(X=5, Y=10)),
                    ml(
                        self.box_style,
                        style(background_mat=mat.green),
                    )
                    | rl.gravity(Pos(X=-10, Y=5)),
                    ml(
                        self.box_style,
                        style(background_mat=mat.yellow),
                    )
                    | rl.gravity(Pos(X=5, Y=-10)),
                )
                | rl.at_curve(bc)
            ).resolve(mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "f7eff1f72193d2166c75e35c1c7ea1a4920e57c31690948c3bf4d2f340015994",
            "test_at_curve_constraint",
            use_materials=True,
        )

    def test_look_at_and_look_along_orientation(self):
        """Verify orientation rules including soft look_at targets and explicit look_along directional vectors."""
        from itertools import product

        with BuildPart() as result:
            box_style = self.box_style + style(
                top_scale=0.1,
                transform=Rot(X=0, Z=90),
            )
            rl.group(
                red_objs := [
                    ml(
                        box_style,
                        style(
                            background_mat=mat.red,
                        ),
                    )
                    | rl.transform(x=x, y=y, z=0)
                    | rl.look_at(Pos(X=0, Y=0), soft=True)
                    for x, y in product([5, -5], repeat=2)
                ],
                [
                    ml(
                        box_style,
                        style(
                            background_mat=mat.blue,
                        ),
                    )
                    | rl.gravity(Pos(X=x, Y=y))
                    | rl.look_at(target)
                    for target, (x, y) in zip(red_objs, product([2, -2], repeat=2))
                ],
                ml(
                    box_style,
                    style(
                        background_mat=mat.green,
                    ),
                )
                | rl.gravity(Pos(X=3, Y=3))
                | rl.transform(rx=0)
                | rl.look_along(Axis.X),
                ml(
                    box_style,
                    style(
                        background_mat=mat.green,
                    ),
                )
                | rl.gravity(Pos(X=3, Y=-3))
                | rl.transform(rx=0)
                | rl.look_along(-Axis.Y),
            ).resolve()

        self.assertPart(
            result.part,
            "aa128f7ec772f17319d9036a3b9c729f29ec5503f9f41813e85b4a1fec725386",
            "test_look_at_and_look_along_orientation",
            use_materials=True,
        )

    def test_stack_rule_with_dynamic_gap_function(self):
        """Verify linear stack layout along an axis using dynamic step functions for gap calculation."""
        with BuildPart() as result:
            (
                rl.group(
                    [
                        ml(
                            self.box_style,
                            style(background_mat=mat.red),
                        )
                    ]
                    * 5
                )
                | rl.stack(-Axis.X, gap=lambda idx: idx + 1)
                | rl.gravity(Pos(X=5, Y=5)).on_self()
            ).resolve()

        self.assertPart(
            result.part,
            "c852c9ae0a43d701e06b08754ee1cd555c1f2f33138296c5c7ac8767423ae435",
            "test_stack_rule_with_dynamic_gap_function",
            use_materials=True,
        )

    def test_configure_dofs_axis_lock(self):
        """Verify locking specific DOFs (disabling Y translation) via configure_dofs rule."""
        with BuildPart() as result:
            (
                ml(
                    self.box_style,
                    style(background_mat=mat.red),
                )
                | rl.configure_dofs(y=False)
                | rl.gravity(Pos(X=5, Y=5))
            ).resolve()

        self.assertPart(
            result.part,
            "69b4fd8a7ea18fd0c042d444e3c52e60347c260c4139f09c0eefb7f1c17def80",
            "test_configure_dofs_axis_lock",
            use_materials=True,
        )

    def test_tag_system(self):
        """Verify the tag system, including scoped root lookups, untagged filtering, and structural transform targets."""
        with BuildPart() as result:
            rl.group(
                # Step 1: Define a nested group structure containing tagged objects
                red_blue := rl.group(
                    rl.object(
                        ml(
                            self.box_style,
                            style(background_mat=mat.red),
                        ),
                        tag="red",
                    ),
                    rl.object(
                        ml(
                            self.box_style,
                            style(background_mat=mat.blue),
                        ),
                        tag=["blue", "shared_tag"],
                    ),
                    tag="red_blue",
                ),
                rl.object(
                    ml(
                        self.box_style,
                        style(background_mat=mat.green),
                    ),
                    tag=["green", "shared_tag"],
                ),
                # Step 2: Apply precise transform rules using selectors, scoped roots, and tags
                # Select by direct tag string matching
                rl.tagged("red") | rl.transform(x=1),
                # Use a specific root element to restrict the tag search scope
                rl.root(red_blue).tagged("shared_tag") | rl.transform(x=-1),
                # Nest a tag-based root query inside another scoped root lookup
                rl.root(rl.tagged("red_blue")).tagged("shared_tag")
                | rl.transform(z=-0.1),
                # Chain sequential root scopes and subtract specific tags using untagged filtering
                rl.root(rl.root(rl.tagged("red_blue")).tagged("red")).untagged("blue")
                | rl.transform(z=0.1),
                # Apply procedural transformations down across immediate children and deep physical leaves
                rl.tagged("red_blue")
                | rl.transform(
                    rz=45
                ).on_each()  # Apply to the groups named "red_blue" (there is the only ONE group)
                | rl.transform(
                    rz=45
                ).on_deep_physical(),  # Apply to all physical children (red, blue) of the groups "red_blue"
                rl.tagged("green") | rl.transform(sx=2),
                # Combine systemic tags with untagged criteria to filter exclusive physical objects (rl.object)
                rl.tagged(rl.TAG_OBJECT).untagged("green") | rl.transform(sy=2),
            ).resolve()

        self.assertPart(
            result.part,
            "df1b69325e74b42ef7eed1df8ab01e6b6332494b2fdae37b683ff5ab81a8e117",
            "test_tag_system",
            use_materials=True,
        )
