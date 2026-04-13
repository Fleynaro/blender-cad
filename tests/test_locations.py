from blender_cad import *
from tests.test_base import BaseCADTest

class TestLocationsAndPatterns(BaseCADTest):
    """
    Category: Transformations & Distributions
    Tests for complex positioning, coordinate systems, and pattern generation.
    """

    def test_nested_locations_with_child_offset(self):
        """
        Verify that nested BuildPart contexts correctly isolate Locations 
        and properly combine the child's local 'loc' with the parent's context.
        
        Expected behavior: 
        1. The Box(0.1) is created once in its local identity context.
        2. Its local offset X=0.5 is applied.
        3. The parent context Z=0.5 is applied only once during the final add.
        Resulting position should be (0.5, 0, 0.5).
        """
        with BuildPart() as result:
            Box(1, 1, 1)
            with Locations(Pos(Z=0.5)):
                with BuildPart():
                    with BuildPart():
                        with BuildPart() as child:
                            Box(0.1, 0.1, 0.1)
                            # Child's manual offset
                            child.loc = Pos(X=0.5)

        self.assertPart(
            result.part,
            "750d6968d809b0441f90dcc7d6bff12c6b6b7b1d99ec1df03a07a831fdb1d176",
            "test_nested_locations_with_child_offset"
        )

    def test_multiple_locations_with_nested_assembly(self):
        """
        Verify that multiple parent locations correctly multiply a deeply 
        nested assembly without internal duplication.
        
        Setup: 
        - Locations at X=0 and X=2.
        - Deeply nested Box(1,1,1).
        - A small Box(0.1) offset to Z=0.5 (placed on top of the large box).
        
        Expected behavior:
        Total of 4 objects in the final Part:
        - 2 Large Boxes at (0,0,0) and (2,0,0)
        - 2 Small Boxes at (0,0,0.5) and (2,0,0.5)
        """
        with BuildPart() as result:
            with Locations(Pos(X=0), Pos(X=2)):
                # This context captures the entire "Box + Small Box" logic
                with BuildPart():
                    with BuildPart():
                        Box(1, 1, 1)
                        with BuildPart():
                            with BuildPart() as child:
                                Box(0.1, 0.1, 0.1)
                                child.loc = Pos(Z=0.5)

        # Verification via hash. 
        # If duplication was broken, we'd see 4 small boxes or 4 large boxes.
        self.assertPart(
            result.part,
            "63628e8fbfd4bf1ae3f5b8ddc55315360dc9fa281b60d2b4e4deae4b58c46861",
            "test_multiple_locations_with_nested_assembly"
        )

    def test_nested_locations_multiplication(self):
        """
        Verify that nested Locations across different BuildPart contexts 
        properly multiply to create a grid.
        
        Logic Flow:
        1. Outer Locations defines X=[0, 2].
        2. Inner Locations defines Y=[0, 2].
        3. The multiplication [X0, X2] * [Y0, Y2] creates 4 points:
           (0,0,0), (0,2,0), (2,0,0), (2,2,0).
        
        Expected behavior:
        Exactly 4 Box(1,1,1) instances should be created at the grid points.
        """
        with BuildPart() as result:
            with Locations(Pos(X=0), Pos(X=2)):
                with BuildPart():
                    with Locations(Pos(Y=0), Pos(Y=2)):
                        with BuildPart():
                            Box(1, 1, 1)

        # Hash verification for the 4-box grid assembly
        self.assertPart(
            result.part,
            "3fb78c4ecdd62f976a932fcf8afeb95befb35e23e5482d575c12645b7a55e857",
            "test_nested_locations_multiplication"
        )

    def test_cone_flip_z_transformation(self):
        """
        Verify that the FlipZ transformation correctly inverts the part 
        along the local Z axis. 
        For a Cone, this should swap the base and the apex positions.
        """
        with BuildPart() as result:
            # Create a cone: bottom radius 1, top radius 0.1, height 3
            Cone(1, 0.1, 3)
            
            # Apply FlipZ transformation using the multiplication operator
            # This should invert the geometry along the Z axis
            result.transform *= FlipZ
            
            # Verification via hash to ensure all vertices are mirrored correctly
            self.assertPart(
                result.part, 
                "cc4dcd7ab4e01ede260e5f8b8f2586b90776204e280d160c07641fba837a4cb3", 
                "test_cone_flip_z_transformation"
            )

    def test_independent_loc_and_scale(self):
        """Verify that loc and scale properties independently update the world matrix."""
        with BuildPart() as result:
            Box(2, 2, 2)
            # Apply a chamfer/bevel to check if topology scales correctly
            bevel(edges().top().min_y(), radius=0.5)
            
            # Initial scale along Z axis
            result.scale = (1, -1, 2)
            
            # Apply first combined position and rotation offset
            result.loc *= Pos(X=1, Y=2, Z=-3) * Rot(X=30, Y=40, Z=50)
            
            # Re-applying or updating scale (should remain independent of loc)
            result.scale = (1, -1, 2)
            
            # Apply additional position offset
            result.loc *= Pos(Y=5)

        # Hash check to verify the final transformed geometry
        self.assertPart(result.part, "faa5220b1bb0d325a9ba1af6030441f6b5fe263cdb5f7fe4be16bcf6caf4bd9c", "test_independent_loc_and_scale")


    def test_grid_and_rotational_locations(self):
        """Verify nested Location transformations combined with Grid patterns."""
        with BuildPart() as result:
            # Shift center by 10 on Z and rotate around X
            with Locations(Pos(Z=10) * Rot(X=90)):
                # Create a 2x2 grid with 5.0 spacing
                with GridLocations(5, 5, 2, 2):
                    # Local rotation for each instance
                    with Locations(Rot(Y=45)):
                        Box(1, 1, 1)
        
        self.assertPart(result.part, "6c61662c324de2a317d069f31043a6e5290cde8896e0314f222c5424c480c3cb", "test_grid_and_rotational_locations")

    def test_polar_distribution(self):
        """Verify circular distribution of objects using PolarLocations."""
        with BuildPart() as result:
            with PolarLocations(radius=8, count=6):
                Sphere(1.5)
                
        self.assertPart(result.part, "6c270f31de94289dcada98cbb334326ae9e2a88a94a20aace75a66e372539d1a", "test_polar_distribution")

    def test_hexagonal_distribution(self):
        """Verify hexagonal grid layout for parts."""
        with BuildPart() as result:
            # Position the grid 5 units below origin
            with Locations((0, 0, -5)):
                with HexLocations(apothem=1, x_count=2, y_count=2):
                    Box(0.5, 0.5, 2)
                    
        self.assertPart(result.part, "9bd253f01cc649637642bf3325a0d19f1a68a985a2e41405e9f729ff15375ac9", "test_hexagonal_distribution")

    def test_surface_coordinate_chaining_and_compensation(self):
        """
        Verify the 'Sandwich' logic of SurfaceLocation:
        1. Left of surface: Global parent_loc (Pos(Z=2)).
        2. Right of surface: Local U, V, Z offsets.
        3. Compensation: Multiple Pos calls should accumulate (X=1 + X=1 = U=2).
        4. Result Type: Ensure the final object is still a SurfaceLocation.
        """
        with BuildPart() as result:
            # Setup base geometry
            Cylinder(2, 5, segments=64)
            cyl_face = faces().filter_by(GeomType.CYLINDER)[0]
            
            # The complex chain:
            # Global Z+1 -> Z+1 -> Surface -> U+1, Z+1 -> U+1, Z-1 -> V+2, Z+0.2 -> Rot Y90
            target_loc: SurfaceLocation = (
                Pos(Z=1) * Pos(Z=1) * cyl_face.surface(uv) * Pos(X=1, Z=1) * Pos(X=1, Z=-1) * Pos(Y=2, Z=0.2) * Rot(Y=90)
            )
            
            # Place the marker
            Marker(loc=target_loc)

            # 1. Type Verification: The math should preserve the SurfaceLocation class
            self.assertTrue(
                isinstance(target_loc, SurfaceLocation), 
                f"Expected SurfaceLocation, got {type(target_loc)}"
            )

            # 2. Logic Verification: 
            # - U should be 2.0 (X=1 + X=1)
            # - V should be 2.0 (Y=2)
            # - Z (Normal offset) should be 0.2 (1 - 1 + 0.2)
            self.assertAlmostEqual(target_loc.u_offset, 2.0)
            self.assertAlmostEqual(target_loc.v_offset, 2.0)
            self.assertAlmostEqual(target_loc.z_offset, 0.2)

        # Final Hash Verification
        self.assertPart(
            result.part, 
            "4748ea7d430fe770fd4559fc91fec6c9f2ac4016aabc705089e87f5063d7ab56", 
            "test_surface_coordinate_chaining_and_compensation"
        )

    def test_surface_to_global_transition_via_loc(self):
        """
        Verify that calling .loc on a SurfaceLocation converts it to a 
        static Location, shifting subsequent operations to global space.
        
        Logic:
        1. Create a Cone.
        2. Create a SurfaceLocation with local offsets (U=0.5, Normal=0.1).
        3. Call .loc to 'freeze' the evaluation into a static global matrix.
        4. Multiply by Pos(X=2). This should be a global shift, 
           NOT a further U-offset on the cone.
        """
        with BuildPart() as result:
            Cone(2, 0.1, 2, segments=64)
            cone_face = result.faces().filter_by(GeomType.CONE)[0]
            
            # Step-by-step transformation:
            # a) Evaluate point on cone with local offsets
            surf_loc: SurfaceLocation = cone_face.surface(uv) * Pos(X=0.5, Z=0.1)
            
            # b) Convert to static Location (Global Matrix)
            static_loc = surf_loc.loc
            
            # c) Apply global translation
            target_loc = static_loc * Pos(X=2)

            # Place Marker
            Marker(loc=target_loc)

            # Type Verification: 
            # After .loc, target_loc must be a standard Location, not SurfaceLocation
            self.assertIs(
                type(target_loc), Location, 
                f"Expected static Location after .loc, but got {type(target_loc)}"
            )

        # Final Hash Verification
        self.assertPart(
            result.part, 
            "5527967de8e2014705f812473f4aa5a337deba4ed5fe2956909e44df0306e622", 
            "test_surface_to_global_transition_via_loc"
        )

    def test_surface_with_grid_and_cumulative_rotation(self):
        """
        Verify that GridLocations correctly offset the U and V parameters of a 
        SurfaceLocation, and that nested rotations (10 + 20 = 30 degrees) 
        are accumulated correctly.
        """
        # Setup: A cylinder with radius 10, height 20
        with BuildPart() as result:
            Cylinder(2, 5, segments=64)
            cyl_face = faces().filter_by(GeomType.CYLINDER)[0]
            
            # 1. Apply a global scale to the result (Verify scale persistence)
            result.scale.x = 1.5
            
            # 2. Start from a surface location at V=0.5
            with Locations(cyl_face.surface(uv.set(v=0.5))):
                # 3. Apply a local Z-offset (Normal) and partial rotation
                with Locations(Pos(Z=0.1) * Rot(X=10)):
                    # 4. Generate a 2x2 grid (spacing 3, 3) 
                    # Grid X/Y will be treated as U/V offsets on the surface
                    with GridLocations(2, 2, 3, 3):
                        # 5. Apply the remaining rotation and place a marker
                        # Total Rotation should be Rot(X=30)
                        Marker(Rot(X=20))

        # Verification via hash.
        # This checks that:
        # - 4 Markers exist.
        # - Markers are projected onto the cylinder surface.
        # - Markers have combined X-rotation of 30 degrees.
        # - Global X-scale of 1.5 is preserved.
        self.assertPart(
            result.part, 
            "2b9ddf05af55e67638ac1645e8117bafec8bd52822fb315662c9f3bc169f7f9c", 
            "test_surface_with_grid_and_cumulative_rotation"
        )

    def test_align_joints(self):
        """Verify the align function works correctly for joint-like positioning."""
        with BuildPart(mat=mat.blue) as result:
            Box(10, 10, 2)
            top_face = faces().sort_by(Axis.Z)[-1]
            top_face.mat = mat.green
            
            # Apply initial offset and rotation to the parent part
            result.loc = Pos(X=5, Y=6, Z=2) * Rot(X=30, Y=40)
            
            with BuildPart(mode=Mode.JOIN) as child:
                Box(5, 7, 1)
                bottom_face = child.faces().sort_by(Axis.Z)[0]
                bottom_face.mat = mat.red
                
                # Define connection points on child and parent faces
                from_loc = bottom_face.at(0.5, 0.5)
                to_loc = top_face.at(1.0, 0.5)
                
                # Align the child part to the parent's face location
                child.loc = align(from_port=from_loc, to_port=to_loc)

        self.assertPart(result.part, "6b067a1072e5af18e25e3fedbbf49960969c2679254c60b3a9c615de4965f67e", "test_align_joints")

    def test_align_joints_with_twist(self):
        """Verify the align function works correctly with the twist parameter for joint rotation."""
        with BuildPart(mat=mat.blue) as result:
            Box(10, 10, 2)
            top_face = faces().sort_by(Axis.Z)[-1]
            top_face.mat = mat.green
            
            # Apply initial offset and rotation to the parent part
            result.loc = Pos(X=5, Y=6, Z=2) * Rot(X=30, Y=40)
            
            with BuildPart(mode=Mode.JOIN) as child:
                Box(5, 7, 1)
                bottom_face = child.faces().sort_by(Axis.Z)[0]
                bottom_face.mat = mat.red
                
                # Define connection points on child and parent faces using .at()
                from_loc = bottom_face.at(0.5, 0.5)
                to_loc = top_face.at(1.0, 0.5)
                
                # Align the child part to the parent's face location with an additional twist
                # twist=45 rotates the child part 45 degrees around the joint axis
                child.loc = align(from_port=from_loc, to_port=to_loc, twist=45)

        self.assertPart(result.part, "40511e9e4740266a7e85c80032ec7d3ff41ed194ac74272ca1b4852477910cab", "test_align_joints_with_twist")

    def test_joint_connection(self):
        """Verify part alignment using the Joint class and connect_to method."""
        with BuildPart() as result:
            # Create the base part
            Box(5, 5, 5)
            top_face = faces().top()[0]
            top_face.mat = mat.green
            # Define some offset for the main part
            result.loc = Pos(X=5, Y=6, Z=7) * Rot(X=30, Y=40, Z=50)
            # Define a joint at the center of the top face
            top_joint = Joint(top_face.at(0.5, 0.5))
            
            with BuildPart(mode=Mode.JOIN) as child:
                # Create the second part to be attached
                Box(1, 1, 2)
                bottom_face = faces().bottom()[0]
                bottom_face.mat = mat.red
                # Define some offset for the child part
                child.loc = Pos(X=2, Y=3, Z=4) * Rot(X=50, Y=60, Z=70)
                # Define a joint on the child's bottom face
                bottom_joint = Joint(bottom_face.at(0.5, 0.5))
                
                # Move the child joint to the parent joint with a Z offset
                bottom_joint.to(top_joint.offset(Pos(Z=1)), mode=Mode.PRIVATE)

        self.assertPart(result.part, "825e732dc0cd32fd07ff1b13283b0ad545ce2faf6709bcb519d771ca2a991851", "test_joint_connection")
