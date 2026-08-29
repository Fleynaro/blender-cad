from blender_cad import *
from tests.test_base import BaseCADTest


class TestComponents(BaseCADTest):
    """
    Category: Component Systems
    Tests for BoxComp, custom joints, and complex assembly logic.
    """

    def test_imperial_sci_fi_throne(self):
        """Verify the construction of a complex sci-fi throne using BoxComp and advanced joint mapping."""
        with BuildPart() as result:
            # 1. Initialize core components
            base = BoxComp(7, 18, 7)
            back = BoxComp(7, 18, 10, taper=0.5)
            cap = BoxComp(10, 9, 1, taper=0.9)
            seat = BoxComp(10, 15, 1)
            side = BoxComp(10, 4, 4)
            hole = BoxComp(10, 4, 4)
            stand = BoxComp(7, 7, 4, taper=0.7)

            # 2. Refine geometry of individual components using their local contexts
            with base:
                # Scale bottom edges on the max X side to create a tapered look
                transform(edges().bottom().max_x(), op=Scale(X=0.7))
                base.update_freezed_joints()

            with side:
                # Bevel specific edges for the side armrests
                bevel(edges().top().min_y(), radius=0.5)

            with hole:
                # Round the top face of the decorative hole/inset
                bevel(faces().top(), radius=0.4)

            # 3. Assemble the throne using joints
            add(base)

            # Stack the backrest and the decorative cap
            back.j_bottom().to(base.j_top())
            cap.j_bottom(uv.max_x()).to(back.j_top(uv.max_x()))

            # Attach seat and side panels with rotations and flips
            seat.j_face(Axis.X, uv.bottom()).to(base.j_face(-Axis.X, uv.bottom()))

            # First side panel
            side.j_face(Axis.X, uv.bottom().max_y()).to(
                base.j_face(-Axis.X, uv.bottom().max_y())
            )

            # # Second side panel using FlipY for symmetry
            side.j_face(Axis.X, uv.bottom().min_y()).to(
                base.j_face(-Axis.X, uv.bottom().min_y()), op=FlipY
            )

            # # 4. Final boolean operations and foundation
            # # Subtract the decorative hole from the base
            hole.j_bottom().to(base.j_top().offset(Pos(Z=1.5)), mode=Mode.SUBTRACT)

            # # Attach the stand to the bottom of the base
            stand.j_top(uv.max_x()).to(base.j_bottom(uv.min_x()))

        self.assertPart(
            result.part,
            "8954af1eafb4dea7c269f732ff9642a3484c6263238146153e040fd205347e24",
            "test_imperial_sci_fi_throne",
        )

    def test_sci_fi_console_with_monitors(self):
        """Verify a modular sci-fi console assembly with custom joints and repeating elements."""
        with BuildPart() as result:
            # 1. Create a base with a cut-out and rounded edges
            with BoxComp(10, 10, 1) as base:
                # Bevel side edges on the min_x side
                bevel(edges().min_x().side(), radius=0.5, segments=3)
                # Subtract a block to shape the console base
                with Locations(Pos(X=5)):
                    Box(10, 10, 10, mode=Mode.SUBTRACT)
                # Update joints after geometry modification
                base.update_freezed_joints()

            # 2. Create a stand assembly as a private part
            with BuildPart(mode=Mode.PRIVATE) as stand:
                width = 2.5
                height = 8
                thickness = 0.2
                depth = 1.5

                vert_block = BoxComp(thickness, width, height / 2)
                hor_block = BoxComp(depth, width, thickness)

                add(vert_block)
                # Define custom joints for the stand assembly
                stand_j_bottom = Joint(vert_block.faces().max_x()[0].at(uv.bottom()))

                # Chain the blocks together
                hor_block.j_bottom(uv.max_x()).to(vert_block.j_top(uv.max_x()))
                vert_block.j_bottom(uv.min_x()).to(hor_block.j_top(uv.min_x()))

                # Top joint for monitor attachment
                stand_j_top = Joint(vert_block.j_top().loc)

            # 3. Define the monitor component
            with BoxComp(thickness, width, 3, taper=2) as monitor:
                pass

            # 4. Final Assembly
            add(base)
            # Filter side faces of the base for mounting stands
            attach_faces = base.faces().side() - base.faces().max_x()

            for face in attach_faces:
                stand_j_bottom.to(face.at(uv.bottom()))
                monitor.j_bottom().to(stand_j_top.offset(Rot(Y=60)), twist=0)

        self.assertPart(
            result.part,
            "9582434f3747789ac819442caa39ebb24e45d8c1a1f74940aadc18e21c294435",
            "test_sci_fi_console_with_monitors",
        )

    def test_parametric_lamp_with_multi_joint_stand(self):
        """Validate a parametric lamp assembly with a multi-part stand, angled supports, and repeated joint-driven placement."""

        with BuildPart() as result:
            # 1. Create lamp body with beveled top and bottom edges
            with BoxComp(0.2, 0.8, 3) as lamp:
                top_edges = edges().top().min_y() + edges().top().max_y()
                bottom_edges = edges().bottom().min_y() + edges().bottom().max_y()
                bevel(top_edges + bottom_edges, radius=0.2)

                # Joint for attaching lamp to supports
                lamp_j_back = lamp.j_face(-Axis.X)

            # 2. Build top stand assembly with multiple angled arms
            with BuildPart(mode=Mode.PRIVATE) as stand_top:
                b1 = BoxComp(0.3, 0.3, 1)
                b2 = BoxComp(0.1, 2, 0.2)
                b3 = BoxComp(1, 0.2, 0.1)

                b3_j_lamp = b3.j_face(Axis.X)

                # Base joint to connect to middle stand
                stand_top_j_bottom = Joint(b1.j_bottom().loc)
                stand_top_j_lamps: list[Joint] = []

                # Assemble vertical and horizontal supports
                add(b1)
                b2.j_face(-Axis.X).to(b1.j_face(-Axis.X), move_only=True)

                # Create multiple angled arms (symmetrical placement)
                b3.j_face(-Axis.X, uv.max_y()).to(
                    b2.j_face(Axis.X, uv.max_y()).offset(Rot(X=-30))
                )
                stand_top_j_lamps.append(Joint(b3_j_lamp.loc))

                b3.j_face(-Axis.X, uv.max_y()).to(
                    b2.j_face(Axis.X, uv.max_y()).offset(Rot(X=30))
                )
                stand_top_j_lamps.append(Joint(b3_j_lamp.loc))

                b3.j_face(-Axis.X, uv.min_y()).to(
                    b2.j_face(Axis.X, uv.min_y()).offset(Rot(X=-30))
                )
                stand_top_j_lamps.append(Joint(b3_j_lamp.loc))

                b3.j_face(-Axis.X, uv.min_y()).to(
                    b2.j_face(Axis.X, uv.min_y()).offset(Rot(X=30))
                )
                stand_top_j_lamps.append(Joint(b3_j_lamp.loc))

                b3.j_face(-Axis.X, uv.top()).to(
                    b1.j_face(Axis.X, uv.top()).offset(Rot(X=-30))
                )
                stand_top_j_lamps.append(Joint(b3_j_lamp.loc))

                b3.j_face(-Axis.X, uv.top()).to(
                    b1.j_face(Axis.X, uv.top()).offset(Rot(X=30))
                )
                stand_top_j_lamps.append(Joint(b3_j_lamp.loc))

            # 3. Middle vertical connector (simple pass-through element)
            with BoxComp(0.2, 0.2, 3) as stand_middle:
                pass

            # 4. Bottom stand with solver-based alignment and deformation
            with BuildPart(mode=Mode.PRIVATE) as stand_bottom:
                with BoxComp(0.2, 1.5, 2) as b1:
                    bevel(
                        edges().top().min_y() + edges().top().max_y(),
                        radius=0.4,
                        segments=1,
                    )

                    # Shape deformation
                    transform(faces().bottom(), op=Scale(Y=0.3))
                    transform(faces().top(), op=Pos(Z=-0.4))

                    b1.update_freezed_joints()

                    stand_bottom_j_top = b1.j_top().offset(Pos(Z=-0.1))
                    b1_j_back = b1.j_face(-Axis.X).offset(Pos(Z=-0.1))

                    # Tilt base
                    b1.loc = Rot(Y=-30)

                    # Align bottom to ground
                    front_edges = edges().max_x()
                    b1_min_z = edges().bottom()[0].center().z
                    transform(
                        front_edges, op=Pos(Z=b1_min_z - front_edges[0].center().z)
                    )

                with BoxComp(0.3, 0.3, 0.3) as b2:
                    bevel(edges().top().max_x(), radius=1, segments=1)
                    transform(edges().top(), op=Pos(X=b2.size.x * 0.3))

                    b2.update_freezed_joints()
                    b2_j = b2.j_face(Axis.X).offset(Pos(Z=-0.05))

                with BoxComp(1, 0.1, 0.2) as b3:
                    transform(faces().max_x(), op=Scale(Z=1.5))

                    b3_j1 = b3.j_face(Axis.X)
                    b3_j2 = b3.j_face(-Axis.X)

                # Solve alignment between base and connector
                add(b1)
                for s in Solver():
                    Y = s.param(0.0)
                    b3_j1.to(b1_j_back.offset(Pos(Y=Y) * Rot(Y=30)), mode=s.mode())
                    b2_j.to(b3_j2, move_only=True, mode=s.mode())

                    if s.is_final:
                        b3_j1.to(b1_j_back.offset(Pos(Y=Y) * Rot(Y=-30)))
                        b2_j.to(b3_j2, move_only=True)

                    s.aim_equal(b1_min_z, b2.bbox.min.z)

            # 5. Final assembly of all components
            add(stand_bottom)

            stand_middle.j_bottom().to(stand_bottom_j_top, move_only=True)
            stand_top_j_bottom.to(stand_middle.j_top(), move_only=True)

            # Attach lamps between paired joints (centered between supports)
            for i in range(0, len(stand_top_j_lamps), 2):
                j1 = stand_top_j_lamps[i]
                j2 = stand_top_j_lamps[i + 1]

                target = Pos((j1.loc.position + j2.loc.position) / 2) * Pos(
                    X=-lamp.size.x / 2
                )
                lamp_j_back.to(target, move_only=True)

        self.assertPart(
            result.part,
            "48e4a8a172e4cdaef460c00a0fcb14e757574bf9f680e2bc2795472280d85a23",
            "test_parametric_lamp_with_multi_joint_stand",
        )

    def test_staircase_solver_alignment(self):
        """
        Verify that the StaircaseBuilder solver (Nelder-Mead) correctly calculates
        parameters to bridge two disconnected floor components.
        Checks if the top joint of the generated staircase aligns with the target floor joint.
        """
        import math

        class StaircaseBuilder(Component):
            def __init__(
                self,
                step: BoxComp,
                count: int,
                turn_angle: float,
                turn_decay: float = 1.0,
            ):
                """
                A parametric staircase component.
                :param step: The Component instance to be used as a step.
                :param count: Total number of steps.
                :param turn_angle: Initial rotation angle for the turn.
                :param turn_decay: Coefficient to change the rotation intensity (1.0 = constant turn).
                """
                super().__init__()
                self.count = int(count)

                # We get current active solver to minimize 'Add' boolean operations (see s.mode() below)
                s = solver()

                step_j_bottom = step.j_bottom(uv.local().min_x())
                step_j_top = step.j_top(uv.local().max_x())
                step.transform = Transform()

                with self:
                    current_angle = turn_angle

                    for i in range(self.count):
                        step.transform *= SizeAlongAxis(
                            Axis.X,
                            step.size.x
                            + step.size.y
                            * 0.5
                            * math.sin(math.radians(abs(current_angle))),
                        )

                        if i == 0:
                            add(step, mode=s.mode())
                            self.bottom_j = Joint(step.j_face(-Axis.X).loc)
                        else:
                            step_j_bottom.to(
                                step_j_top, twist=current_angle, mode=s.mode()
                            )
                            current_angle *= turn_decay

                        step.transform *= ScaleAlongAxis(Axis.X, 1, reset=True)

                    self.top_j = Joint(step_j_top.loc)

            def j_bottom(self):
                """Returns the lower joint of the staircase."""
                return self.bottom_j

            def j_top(self):
                """Returns the upper joint of the staircase."""
                return self.top_j

            @staticmethod
            def build_from_to(
                start_point: Location, end_point: Location, step_comp: Component
            ):
                """
                Static solver method to build a staircase that perfectly connects two points.
                """
                for s in Solver(sm.brute().with_polish(), max_steps=20):
                    num = s.param(1, min=5, max=12, step=1.0)
                    angle = s.param(0, min=0, max=40, steps=5)
                    decay = s.param(1.0, min=0.5, max=1.0, steps=3)
                    if s.is_final:
                        print(f"num={num}, angle={angle}, decay={decay}")
                    stair = StaircaseBuilder(
                        step_comp, count=int(num), turn_angle=angle, turn_decay=decay
                    )
                    stair.j_bottom().to(start_point, mode=s.mode())
                    s.aim_equal(stair.j_top().loc.position, end_point.position)
                return stair

        with BuildPart() as result:
            # 1. Setup the environment: Two floors at different heights and orientations
            floor = BoxComp(5, 5, 1)
            stair_start = floor.j_face(Axis.Y).loc
            add(floor)
            floor.loc = Pos(10, 10, 5)
            stair_end = floor.j_face(-Axis.X).loc
            add(floor)

            # 2. Define the step prototype for the builder
            step_proto = BoxComp(1, 5, 0.5)

            # 3. Invoke the solver to generate the staircase
            # The solver optimizes 'count', 'turn_angle', and 'turn_decay'
            # to minimize the distance between stair.j_top() and stair_end
            StaircaseBuilder.build_from_to(
                stair_start,
                stair_end,
                step_proto,
            )

        self.assertPart(
            result.part,
            "4455f08ff64c5df8dcd9243cf0a6736744172fc3f876b5a6d62c70a5d2b8326b",
            "test_staircase_solver_alignment",
        )

    def test_semicircular_room(self):
        """Verify part extraction and material assignment in a complex architectural scenario."""
        with BuildPart() as result:
            # Create base floor
            Box(100, 80, 5)

            # Bevel the side edges of the front face to create a rounded profile
            bevel(faces().max_x().edges().side(), radius=0.5, segments=16)

            # Extrude the top face upwards to create walls
            extrude(faces().top(), op=Pos(Z=30))

            # Select the newly created cylindrical face from the bevel/extrusion
            cyl_face = faces().filter_by(GeomType.CYLINDER)[0]

            x_spacing = 28
            windows_size = 20

            # Place windows along the curved surface using UV coordinates
            with Locations(cyl_face.location(uv) * Pos(Y=1, Z=-1)):
                for loc in GridLocations(
                    x_spacing=x_spacing, y_spacing=0, x_count=8, y_count=1
                ):
                    with Locations(loc):
                        with BuildPart():
                            # Create a box for the window frame/glass volume
                            Box(windows_size, windows_size, 10)
                            window_top_face = faces().top()
                            # Delete the face from the frame to create a hole
                            delete(window_top_face)

                            # Use .part() to create a separate entity for the glass from the deleted face
                            glass = window_top_face.part()
                            # Add the glass part back to the main assembly
                            add(glass, mat=mat.blue + mat.PBR(alpha=0.5))

                # Add decorative piers/pillars between the windows
                with GridLocations(
                    x_spacing=x_spacing, y_spacing=0, x_count=9, y_count=1
                ):
                    with Locations():
                        with BuildPart(mode=Mode.JOIN) as pier:
                            Box(5, 45, 5)
                            pier.mat = mat.red

            transform(op=Scale(XY=1.2), prop_edit=LinearPropEdit())
            roof = faces().group_by(Axis.Z)[-2]
            extrude(roof, op=Pos(Z=10) * Scale(XY=0.8))
            delete(roof)

        self.assertPart(
            result.part,
            "ea95deefa98b06bdd2c7a70e4b40368740b64630f57fdb36436a1093480c6fbd",
            "test_semicircular_room",
        )

    def test_random_tiled_pyramid(self):
        """Verify the creation of a truncated pyramid with randomized tiling on its side faces."""

        def random_cube_grid(
            loc=Location(), grid_size=(10, 10), step=1.0, h_range=(1.0, 5.0), seed=1
        ):
            import random

            rng = random.Random(seed)
            nx = int(grid_size[0] / step)
            ny = int(grid_size[1] / step)
            for ceil_loc in GridLocations(step, step, nx, ny):
                h = rng.uniform(*h_range)
                # Offset each tile based on its generated height to keep base aligned
                with BuildPart(offset=loc * ceil_loc * Pos(0, 0, h / 2)):
                    Box(step - 0.01, step - 0.01, h)

        with BuildPart() as result:
            # Create the base volume of the pyramid
            Box(2, 2, 3)
            # Taper the shape by scaling the top face to create a truncated pyramid effect
            transform(faces().top(), Scale(XY=0.5))

            # Select only the side faces for tiling
            side_faces = faces().side()

            for i in range(4):
                # Create a solid volume from the side face to use as an intersection mask (clipper)
                # This ensures tiles do not protrude beyond the pyramid's original boundaries
                clipper = solidify_faces(side_faces[i], height=1.5)

                with BuildPart():
                    # Map the local coordinate system to the current side face using UV mapping
                    with Locations(side_faces[i].at(uv) * Pos(Z=0.01)):
                        # Generate a grid of boxes with randomized heights on the face surface
                        random_cube_grid(grid_size=(2, 3), step=0.5, h_range=(0.1, 0.3))

                    # Intersect the generated tiles with the clipper to trim them perfectly to the face shape
                    add(clipper, mode=Mode.INTERSECT)

        # Validate the resulting geometry against the expected hash
        self.assertPart(
            result.part,
            "ca837ad094b7a84a80b5de7e123c996881f352b9b9b3dbc92e11e8ee06922924",
            "test_random_tiled_pyramid",
        )
