from blender_cad import *
from tests.test_base import BaseCADTest

class TestSelectorsAndLocations(BaseCADTest):
    """
    Category: Selectors & Topology Mapping
    Tests for selecting specific entities and verifying UV/parameter mapping 
    via at for Faces, Edges, and Wires.
    """

    def iter(self, steps, callback):
        """Helper to iterate from 0.0 to 1.0 with a fixed number of steps."""
        for i in range(steps):
            callback(i / (steps - 1))

    def test_geometry_type_identification(self):
        loc = Locations(Pos(X=5,Y=6,Z=7) * Rot(X=50,Y=60,Z=70))

        # 1. Check Box (Should be all Planes)
        with BuildPart() as box:
            with loc:
                Box(1, 2, 3)
        # A box has 6 faces, all must be planar
        planar_faces = box.part.faces().filter_by(GeomType.PLANE)
        self.assertEqual(len(planar_faces), 6, "Box should have 6 planar faces")
        
        # 2. Check Sphere (Should be a Spherical surface)
        with BuildPart() as sphere:
            with loc:
                Sphere(1)
        spherical_faces = sphere.part.faces().filter_by(GeomType.SPHERE)
        self.assertEqual(len(spherical_faces), 1, "Sphere must have at least 1 spherical face")

        # 3. Check Cylinder (Should have 2 Planes and 1 Cylinder surface)
        with BuildPart() as cyl:
            with loc:
                Cylinder(radius=1, height=2)
        cyl_planes = cyl.part.faces().filter_by(GeomType.PLANE)
        cyl_sides = cyl.part.faces().filter_by(GeomType.CYLINDER)
        self.assertEqual(len(cyl_planes), 2, "Cylinder should have 2 planar caps")
        self.assertEqual(len(cyl_sides), 1, "Cylinder should have 1 cylindrical side")

        # 4. Check Cone (Should have 1 or 2 Planes and 1 Cone surface)
        with BuildPart() as cone:
            with loc:
                Cone(radius_bottom=2, radius_top=0.01, height=2)
        cone_planes = cone.part.faces().filter_by(GeomType.PLANE)
        cone_sides = cone.part.faces().filter_by(GeomType.CONE)
        self.assertEqual(len(cone_planes), 2, "Cone should have 2 planar caps")
        self.assertEqual(len(cone_sides), 1, "Cone should have 1 conical side")

        # 5. Check Backwards Cone (Should have 1 or 2 Planes and 1 Cone surface)
        with BuildPart() as cone:
            with loc:
                Cone(radius_bottom=1, radius_top=2, height=10)
        cone_planes = cone.part.faces().filter_by(GeomType.PLANE)
        cone_sides = cone.part.faces().filter_by(GeomType.CONE)
        self.assertEqual(len(cone_planes), 2, "Cone should have 2 planar caps")
        self.assertEqual(len(cone_sides), 1, "Cone should have 1 conical side")

        # 6. Check Thin Cone (Should have 1 or 2 Planes and 1 Cone surface)
        # Use a custom topology config with a small angle to ensure correct geometry analysis
        with BuildPart(topology=TopologyConfig(smooth_angle=1)) as cone:
            with loc:
                Cone(radius_bottom=20, radius_top=1, height=0.5)
        cone_planes = cone.part.faces().filter_by(GeomType.PLANE)
        cone_sides = cone.part.faces().filter_by(GeomType.CONE)
        self.assertEqual(len(cone_planes), 2, "Cone should have 2 planar caps")
        self.assertEqual(len(cone_sides), 1, "Cone should have 1 conical side")

        # 7. Multiple Geometries
        with BuildPart() as complex:
            with Locations(Pos(X=5,Y=6,Z=7) * Rot(X=50,Y=60,Z=70)):
                Cone(radius_bottom=1, radius_top=2, height=2)
            with Locations(Pos(X=50,Y=60,Z=70) * Rot(X=150,Y=160,Z=170)):
                Cone(radius_bottom=5, radius_top=0.01, height=20)
            Sphere(0.1)
        planes = complex.part.faces().filter_by(GeomType.PLANE)
        cone_sides = complex.part.faces().filter_by(GeomType.CONE)
        spheres = complex.part.faces().filter_by(GeomType.SPHERE)
        self.assertEqual(len(planes), 4, "Cones should have 4 planar caps")
        self.assertEqual(len(cone_sides), 2, "Cones should have 2 conical side")
        self.assertEqual(len(spheres), 1, "Sphere should have 1 spherical face")

    def test_at_with_offset(self):
        """Verify that at works correctly after modifying the object's location (loc)."""
        with BuildPart() as result:
            Box(5, 5, 1)
            # Offset the entire part along the Z axis
            result.loc = Pos(Z=3)
            
            # Select the bottom face (Z-min)
            bottom_face = result.faces().sort_by(Axis.Z)[0]
            
            # Place a marker at the center of the face.
            # Due to the offset, the Z coordinate should be 2.5 (3 - 0.5)
            Marker(bottom_face.at(0.5, 0.5))

        self.assertPart(result.part, "e041c96775995207b21c21b804230cdca7103a4338531a4ed8cd8133fe35dcf2", "test_at_with_offset")

    def test_plane_at_with_complex_transform(self):
        """Verify that selectors work on a single-face Plane after complex transformations."""
        with BuildPart() as result:
            # Create a 2x2 Plane (1 face)
            Plane(2)
            
            # Apply a complex transformation: Translation, Rotation, and Scaling
            result.transform = Pos(X=1, Y=2, Z=3) * Rot(X=30, Y=40, Z=50) * Scale(X=1.5)
            
            # Select the only face available
            face = result.faces()[0]
            
            # Place markers using UV coordinates and local edge selectors
            # 1. Center of the face
            Marker(face.at(uv))
            
            # 2. Top-center edge (max Y in local space)
            Marker(face.at(uv.local().max_y()))
            
            # 3. Bottom-right corner (max X, min Y in local space)
            Marker(face.at(uv.local().max_x().min_y()))

        self.assertPart(
            result.part, 
            "3ec517876bf803cae8ec16515adc3e639f6fc770f5610c1cea15c9c6f98f29bb", 
            "test_plane_at_with_complex_transform"
        )

    def test_box_selectors_and_grids(self):
        """Verify face selection and coordinate mapping on a Box."""
        with BuildPart() as result:
            Box(10, 12, 10)

            box_faces = faces()
            
            # 1. Mark centers of all faces
            for face in box_faces:
                Marker(face.at(0.5, 0.5), size=5)
            
            # 2. Create a 3x3 grid of markers on the TOP face
            top_face = box_faces.sort_by(Axis.Z)[-1]
            self.iter(3, lambda u: 
                self.iter(3, lambda v: 
                    Marker(top_face.at(u, v))
                )
            )

            # 3. Mark points along the longest edges
            long_edges = box_faces.sort_by(Axis.Z)[0].edges().sort_by(SortBy.LENGTH)[-2:]
            for edge in long_edges:
                with CurveLocations(edge, count=5):
                    Marker(Rot(Y=90))

        self.assertPart(result.part, "7ff1913b1c27e0789da4da9c83bf9bb7905e441d92a430c157486e72f269544c", "test_box_selectors_and_grids")

    def test_sphere_surface_mapping(self):
        """Verify UV mapping on a spherical surface."""
        with BuildPart() as result:
            Sphere(5)
            # Spheres usually have one face in CAD kernels
            main_face = faces().filter_by(GeomType.SPHERE)[0]
            Marker(main_face.at(0.9, 0.9))
            
            # Create a diagonal 'stitch' of markers across the sphere
            self.iter(8, lambda t: Marker(main_face.at(t, t)))
            
            # Mark the 'equator' by fixing V and moving U
            self.iter(8, lambda u: Marker(main_face.at(u, 0.5), size=2))

        self.assertPart(result.part, "fa556fcd13e93371531ec74ee4be8a0deb5a6b340da96bf4ab106463c9c44880", "test_sphere_surface_mapping")

    def test_cone_rectangularization(self):
        """
        Verify that the conical surface is correctly unwrapped 
        (Rectangular vs Fan mapping).
        """
        with BuildPart() as result:
            Cone(radius_bottom=2, radius_top=1, height=10)
            
            # Filter the side surface from the caps (Planes)
            side_face = faces().filter_by(GeomType.CONE)[0]
            caps = faces().filter_by(GeomType.PLANE)
            
            # Mark centers of caps
            for cap in caps:
                Marker(cap.at(0.5, 0.5))

            # Draw 'meridians' on the side surface
            # If rectangularization works, u=const should be a straight vertical line
            for u in [0.0, 0.25, 0.5, 0.75]:
                self.iter(5, lambda v: Marker(side_face.at(u, v)))
            
            # Draw a 'parallel' (circle) at 70% height
            self.iter(8, lambda u: Marker(side_face.at(u, 0.7)))

        self.assertPart(result.part, "e3e8120bb9a9042de836d50b8ae101650f9ec044d9ab4bb2918c0c03447b5b2d", "test_cone_rectangularization")

    def test_rotated_cylinder_wire(self):
        """
        Verify that the wire of a rotated cylinder retains correct 
        parameterization (u=0.0, 0.25, 0.5, 0.75 mapping).
        """
        with BuildPart() as result:
            with Locations(Rot(X=30, Y=40)):
                Cylinder(radius=5, height=1)

            top_face = faces().sort_by(Axis.Z)[-1]
            wire = top_face.wires()[0]
            with CurveLocations(wire, count=5):
                Marker(Rot(Y=90))

        self.assertPart(result.part, "6bf49b4670524c94c9fec7d4446949d266390ffc898233d8fb5d0a11eda19ff7", "test_rotated_cylinder_wire")

    def test_rotated_annulus_wire(self):
        """
        Verify that both outer and inner wires of a subtracted cylinder 
        retain correct parameterization after a 90-degree rotation.
        """
        with BuildPart() as result:
            # Initialize context and create a hollow cylinder (annulus)
            with Locations(Rot(X=90)):
                Cylinder(radius=5, height=1)
                # Subtraction mode to create a hole
                Cylinder(radius=3, height=1, mode=Mode.SUBTRACT)

            # Select the 'top' face by filtering the highest along the Y-axis
            # (Since it was rotated 90 deg around X, the original Z-cap is now on Y)
            top_face = faces().sort_by(Axis.Y)[-1]
            
            # Retrieve all wires (expected: 1 outer + 1 inner)
            wires = top_face.wires()
            
            # Verify mapping for each wire by placing markers at specific parameters
            for wire in wires:
                with CurveLocations(wire, count=5):
                    Marker(Rot(Y=90))

        self.assertPart(result.part, "52e81aa80f66ed42db7d1895152ad3b943d2f2f1d1f79dcee6e4bab66df679e8", "test_rotated_annulus_wire")

    def test_internal_cylinder_surface(self):
        """
        Verify that markers are correctly placed on the internal cylindrical surface 
        of a subtracted cylinder after a 90-degree rotation along the Y-axis.
        """
        with BuildPart() as result:
            # Initialize context and create a hollow cylinder with 90deg Y-rotation
            with Locations(Rot(Y=90)):
                Cylinder(radius=5, height=1)
                # Creating a hole with a smaller radius
                Cylinder(radius=4, height=1, mode=Mode.SUBTRACT)

            # Filter only cylindrical surfaces and sort them by Y-axis position
            # Expected: Two concentric cylindrical faces
            cylinders = faces().filter_by(GeomType.CYLINDER).sort_by(Axis.Y)
            
            # Place markers on the second (internal) cylindrical face 
            # at 50% height (v=0.5) across the full circular perimeter (u)
            for t in [0.0, 0.25, 0.5, 0.75]:
                # Accessing cylinders[1] assumes the internal face index in the collection
                Marker(cylinders[1].at(t, 0.5))

        self.assertPart(result.part, "820f59cd9e5fbaecf31cba132fedeb88711a734c9181e765f9625689cb83940d", "test_internal_cylinder_surface")

    def test_grouping_and_filtering(self):
        """Verify advanced grouping logic."""
        with BuildPart() as result:
            Box(10, 10, 2)
            # Create a pattern to test grouping
            with Locations(faces().sort_by(-Axis.Z)[0]):
                with GridLocations(4, 4, 2, 2):
                    Box(1, 1, 1)
            
            # Group faces by their Z coordinate
            z_groups = faces().group_by(Axis.Z)
            
            # Mark the center of every face in the group with the highest Z
            highest_faces = z_groups[-1] # The top-most faces of the small boxes
            for f in highest_faces:
                Marker(f.at(0.5, 0.5))

        self.assertPart(result.part, "da49faa0ed9bc48d221d46a46edf38d42c36bdcf9e3961f6c319007a30a1bd8c", "test_grouping_and_filtering")

    def test_complex_selection_logic(self):
        """
        Verify complex ShapeList arithmetic: intersection (&), union (+), 
        difference (-), and inversion (~) using Z and X axis groupings.
        """
        with BuildPart() as result:
            Box(10, 10, 2)
            # Create a pattern to test grouping
            with Locations(faces().sort_by(-Axis.Z)[0]):
                with GridLocations(4, 4, 2, 2):
                    Box(1, 1, 2)
            x_groups = faces().group_by(Axis.X)
            z_groups = faces().group_by(Axis.Z)
            selected_faces = (z_groups[-1] & x_groups[-3]) + (z_groups[-2] - ~(x_groups[-2] + x_groups[1]))
            for f in selected_faces:
                Marker(f.at(0.5, 0.5), size=3)
            
        self.assertPart(result.part, "6b59c6b0b942029004bc74d014d7ab4fe305f8eb25588b622acb9c0c391f468c", "test_complex_selection_logic")

    def test_sphere_cone_attachment_and_selection(self):
        """
        Verify that a cone can be attached to a sphere's surface location
        and that surface filtering (including inverse selection) works correctly.
        """
        with BuildPart() as result:
            # Create the base sphere
            Sphere(radius=2)
            
            # Attach a cone to a specific UV coordinate on the sphere's surface
            # This tests if Locations can accept a Location object from a face
            with Locations(faces().filter_by(GeomType.SPHERE)[0].at(0.3, 0.3)):
                Cone(radius_bottom=1, radius_top=0.2, height=2)
            
            # Group faces by geometry type
            sphere_faces = faces().filter_by(GeomType.SPHERE)
            cone_faces = faces().filter_by(GeomType.CONE)
            
            # Use inverse selection (~) to find faces that are NEITHER sphere NOR cone
            # In this case, it should be the flat circular cap (Plane) of the cone
            cone_caps = ~(sphere_faces + cone_faces)
            
            # Place markers at the center of each distinct surface type
            Marker(sphere_faces[0].at(0.5, 0.5))
            Marker(cone_faces[0].at(0.5, 0.5))
            
            # Use a custom size for the marker on the flat cap to verify property support
            Marker(cone_caps[0].at(0.5, 0.5), size=0.5)

        self.assertPart(result.part, "17c0177f117709f1f53c3e4fa28a65ef0f5988afa3897444ca3770e9d2835b9f", "test_sphere_cone_attachment_and_selection")

    def test_uv_selector_offset_with_wrap_around(self):
        """
        Verify that the .offset() method in UVSelector correctly applies 
        UV shifts and handles boundary wrap-around (modulo arithmetic).
        """
        with BuildPart() as result:
            # 1. Create a simple Box
            Box(1, 1, 1)
            
            # 2. Select the face with minimum X.
            # By default, uv.bottom() sets pref_v = 0.
            # We apply offset u = 0.2 and v = -0.1.
            # Calculation: 
            # final_u = (0.5 + 0.2) % 1.0 = 0.7
            # final_v = (0.0 - 0.1) % 1.0 = 0.9
            target_face = faces().min_x()[0]
            Marker(target_face.at(uv.bottom().offset(u=0.2, v=-0.1)))

            self.assertPart(
                result.part, 
                "a146bf6215dbc73826c47a17c1a2d0a4a1aaf276a530cb9921777477a29f3377",
                "test_uv_selector_offset_with_wrap_around"
            )
    
    def test_surface_at_mapping_with_complex_deformation(self):
        """
        Verify that the .at(u, v) method correctly maps coordinates on a surface
        even after the part has been translated, rotated, flipped, and non-uniformly scaled.
        The Marker must follow the deformed surface precisely.
        """
        with BuildPart() as result:
            # 1. Create initial Cone
            Cone(1, 0.1, 3)
            
            # 2. Apply a heavily distorted transformation stack:
            # - Position shift
            # - Rotation on all axes
            # - Mirroring along Z
            # - Non-uniform scaling (stretching the cone into an elliptical one)
            result.transform *= Pos(1, 2, 3) * Rot(X=30, Y=40, Z=50) * FlipZ * Scale(1, 2, 3)
            
            # 3. Locate the conical face and place a Marker at specific UV coordinates
            # .at(0.7, 0.8) should stay locked to the surface regardless of the scale/flip
            conical_face = faces().filter_by(GeomType.CONE)[0]
            Marker(conical_face.at(0.7, 0.8))
            
            # Verification via hash. 
            # If the Marker position calculation doesn't account for 'result.transform', 
            # the final combined geometry hash will fail.
            self.assertPart(
                result.part, 
                "87fee55494e8d3ca198fb9c1620ed43818acba33ebcbf9460068534f519c28c4", 
                "test_surface_at_mapping_with_complex_deformation"
            )

    def test_uv_metric_offsets_on_scaled_cylinder(self):
        """
        Verify that .offset_m() correctly calculates metric distances on a 
        non-uniformly scaled cylinder, handling the Jacobian transformation,
        quad diagonals (0.5, 0.5), and boundary transitions.
        """
        with BuildPart() as result:
            # 1. Create a cylinder with a reasonable segment count for curvature
            Cylinder(2, 5, segments=64)
            cyl_face = faces().filter_by(GeomType.CYLINDER)[0]
            
            # 2. Apply non-uniform scale (squashing the cylinder into an oval)
            # This is the critical part: X-axis is stretched, so 1.0 units of U 
            # covers different metric distances depending on the position.
            result.scale = (2, 1, 2)
            
            # 3. Test various walk scenarios:
            
            # Case A: Diagonal walk from origin
            for i in range(20):
                Marker(cyl_face.at(uv.set(u=0.0, v=0.0).offset_m(u=i, v=i * 0.2)))
            
            # Case B: Pure horizontal walk from the center point (u=0.5)
            # This checks the logic of crossing quad diagonals.
            for i in range(3):
                Marker(cyl_face.at(uv.set(u=0.5, v=0.0).offset_m(u=i)))
                
            # Case C: Pure vertical walk near the seam (u=0.9)
            for i in range(3):
                Marker(cyl_face.at(uv.set(u=0.9, v=0.0).offset_m(v=i)))
                
            # Case D: Walk from exact quad center (0.5, 0.5) with negative offset
            # Verifies floating point precision and 'nudge' logic on boundaries.
            for i in range(3):
                Marker(cyl_face.at(uv.set(u=0.5, v=0.5).offset_m(u=i, v=-i)))

            # 4. Assert geometry state (assuming standard internal hash check)
            self.assertPart(
                result.part, 
                "2fdaff82c24e5ec4cb18210fb64dada7f76692a1e86b08a00e2877e7f5d90ff6",
                "test_uv_metric_offsets_on_scaled_cylinder"
            )

    def test_stretched_sphere_surface_with_grid(self):
        """
        Verify that SurfaceLocation works correctly on a non-uniformly scaled part.
        
        Logic:
        1. Create a Sphere(3).
        2. Apply a global Y-scale of 1.5 (making it an ellipsoid).
        3. Place a primary Marker at a specific UV offset (V=0.2).
        4. Place a 3x3 Grid of markers centered at that same UV point.
        
        The test confirms that the 'parent_loc' (containing the scale) 
        correctly transforms the evaluated surface points.
        """
        with BuildPart() as result:
            Sphere(3)
            sphere_face = result.faces().filter_by(GeomType.SPHERE)[0]
            
            # Apply non-uniform scaling to the whole part context
            result.scale.y = 1.5
            
            # Use a SurfaceLocation with a base UV offset
            with Locations(sphere_face.surface(uv.offset(v=0.2))):
                # This marker sits exactly at the evaluated (0, 0.2) UV point
                Marker(size=5)
                
                # Grid X/Y (1, 1) with 3x3 spacing (offsets -1, 0, 1)
                # These metric offsets are applied to the U/V of the surface
                with GridLocations(1, 1, 3, 3):
                    Marker()

        self.assertPart(
            result.part,
            "4b0ce6f0edc9d416f3efd61e864256d6e65ae34bbc7cb46dad1125a3922ff6d6",
            "test_stretched_sphere_surface_with_grid"
        )
    
    def test_stretched_box_surface_with_grid(self):
        """
        Verify SurfaceLocation behavior on a Box with non-uniform scaling.
        
        Logic:
        1. Create a 2x2x2 Box.
        2. Scale the context by Y=2 (Box becomes 2x4x2).
        3. Identify the top face.
        4. Place a center Marker (size 5).
        5. Place a 2x2 grid (spacing 1, 1) of standard Markers.
        """
        with BuildPart() as result:
            Box(2, 2, 2)
            # Identify the top face (Z-max)
            top_face = faces().top()[0]
            
            # Stretch the box along the Y axis
            result.scale.y = 2
            
            # Use the default center UV for the top face
            with Locations(top_face.surface(uv)):
                # Primary marker at the face center
                Marker(size=5)
                
                # 2x2 Grid with 1.0 spacing in both directions
                # These X/Y grid offsets map to U/V on the planar top_face
                with GridLocations(1, 1, 2, 2):
                    Marker()

        # Verification:
        # The grid should produce markers at (+/- 0.5, +/- 0.5) relative to center.
        # Total markers: 5.
        # Position: All should be at Z=1.0 (original) or Z=1.0 (scaled) 
        # depending on if Z was scaled (here it wasn't).
        
        self.assertPart(
            result.part,
            "c196ab736050aa08d2715db2f1d101b87a43add9e047727706fbfb73ec635ced",
            "test_stretched_box_surface_with_grid"
        )

    def test_half_cylinder_surface_mapping(self):
        """Verify UV mapping on a partial cylindrical surface (semi-cylinder)."""
        with BuildPart() as result:
            Box(5, 5, 5)
            with Locations(Pos(X=2)):
                with BuildPart() as cyl:
                    Cylinder(1, 2.5)
                    cyl.loc = Rot(X=40)
                    cyl.size.y = 4
            cyl_faces = faces().filter_by(GeomType.CYLINDER)
            self.assertTrue(len(cyl_faces) > 0, "Should detect cylindrical geometry")
            
            cyl_face = cyl_faces[0]
            cyl_face.mat = mat.blue
            faces().filter_by(GeomType.PLANE).mat = mat.green
            Marker(cyl_face.at(uv.set(u=0.0, v=0.0)))
            Marker(cyl_face.at(uv.set(u=0.5, v=0.0)))
            Marker(cyl_face.at(uv.set(u=0.0, v=0.5)))
            Marker(cyl_face.at(uv.set(u=0.5, v=0.5)))
            Marker(cyl_face.at(uv.set(u=1.0, v=0.5)))
            Marker(cyl_face.at(uv.set(u=0.5, v=1.0)))
            Marker(cyl_face.at(uv.set(u=1.0, v=0.0)))
            Marker(cyl_face.at(uv.set(u=0.0, v=1.0)))
            Marker(cyl_face.at(uv.set(u=1.0, v=1.0)))
            with Locations(cyl_face.surface(uv)):
                with GridLocations(1, 1, 2, 2):
                    Marker(size=0.5)

        self.assertPart(
            result.part, 
            "8c15c55a5c00ecd8b7b4ebaad585fdadb223a1db2a3a7cc56eaf14944914040f", 
            "test_half_cylinder_surface_mapping"
        )

    def test_curve_conversion(self):
        """Verify that Edge and Wire can be converted to Curves and manipulated."""
        with BuildPart() as result:
            # Cylinder with top edge beveled
            Cylinder(2, 4)
            top_cyl_edge = edges().top()[0]
            add(top_cyl_edge.curve().bevel())

            # Box with top edge beveled and side wire beveled
            with Locations(Pos(X=4)):
                with BuildPart():
                    Box(2, 2, 2)
                    top_box_edge = edges().top()[0]
                    side_wire = wires().max_x()[0]
                    add(top_box_edge.curve().bevel())
                    add(side_wire.curve().bevel(resolution=0))

        self.assertPart(
            result.part,
            "f0b5f4a1c715b69f728fae6e68201319af15e8ea30f89ed80893f68bf74fd85f", 
            "test_curve_conversion"
        )

    def test_part_extraction(self):
        """Verify that Face can be extracted into new Parts."""
        with BuildPart() as result:
            # Box
            with BuildPart():
                Box(2, 2, 2)
                f = faces()
                add(f.top().part(), offset=Pos(Z = 1), mode=Mode.JOIN)
                add(f.side().part(), offset=Pos(Z = -3), mode=Mode.JOIN)
            # Cylinder
            with BuildPart():
                with Locations(Pos(X=3)):
                    Cylinder(1, 2)
                f = faces()
                add(f.top().part(), offset=Pos(Z = 1), mode=Mode.JOIN)
                add(f.side().part(), offset=Pos(Z = -3), mode=Mode.JOIN)

        self.assertPart(
            result.part,
            "58b2ef140153fd16b9b5f0692f007eb172373ace962f8c58b1982be2ae85041f", 
            "test_part_extraction"
        )

    def test_topology_reconstruction_complex(self):
        """
        Verify topology reconstruction with:
        1. Vertex merging after high-radius bevel (avoiding duplicate edges).
        2. Handling of non-manifold extrusions (T-junctions) and flipped normals.
        """
        with BuildPart() as result:
            # Create base geometry
            Box(100, 80, 5)
            # 1. Bevel that potentially creates overlapping geometry at the corners
            bevel(faces().max_x().edges().side(), radius=0.5)
            
            # 2. Extrude an edge to create a T-junction (non-manifold)
            # This face might have an inverted normal relative to the box top face
            extrude(edges().top(), op=Pos(Z=30))
            
            # Select the beveled surface (cylinder) and scale it to stress the wire assembly
            cyl_faces = faces().filter_by(GeomType.CYLINDER)
            cyl_face = cyl_faces[0]
            cyl_face.mat = mat.red
            transform(cyl_face.edges().top(), op=Scale(X=1.5, Y=1.5))

        self.assertPart(
            result.part,
            "8f032e26b6e40bb2a5e5273a6a03e08582111b165bb7e829cc8077b398756a8f", 
            "test_topology_reconstruction_complex"
        )

class TestGeometryCheckpoints(BaseCADTest):
    def test_stacked_boxes_no_overlap(self):
        """
        Verify that adding a separate object (stacked) correctly identifies 
        all faces of the new object as 'new'.
        """
        with BuildPart() as result:
            Box(3, 3, 3)
            with Locations(Pos(Z=3)):
                Box(2, 2, 2)
            # Both objects are joined; only the second box faces should be red
            faces().filter_by(lambda f: f.is_new()).mat = mat.red

        self.assertPart(result.part, "d95dcbf4d15bb2355593186206896d919036a6e49173973780eedbebaf809406", "test_stacked_boxes_no_overlap", use_materials=True)

    def test_overlapping_boxes_union(self):
        """
        Verify that in a boolean union, only the external new surfaces are 
        marked as 'new', while modified original faces are ignored.
        """
        with BuildPart() as result:
            Box(3, 3, 3)
            with Locations(Pos(Z=1)):
                Box(2, 2, 2)
            faces().filter_by(lambda f: f.is_new()).mat = mat.red

        self.assertPart(result.part, "1b48e9edb1f6297b0d09c1f2a2cf18a76b7a590abc46941dfef6994e8490ef25", "test_overlapping_boxes_union", use_materials=True)

    def test_boolean_subtraction_cutout(self):
        """
        Verify that internal faces generated by a subtraction are 'new', 
        while the original exterior faces (even if holes are cut in them) are 'old'.
        """
        with BuildPart() as result:
            Box(3, 3, 3)
            Cylinder(radius=1, height=4, mode=Mode.SUBTRACT)
            faces().filter_by(lambda f: f.is_new()).mat = mat.red

        self.assertPart(result.part, "353c20c4255eb3c95b9d70b1a6c25dad48c358f0f42d0f346fbcdcaa09eea7e5", "test_boolean_subtraction_cutout", use_materials=True)

    def test_plane_subtraction(self):
        """
        Check tracking when a solid is subtracted from a 2D surface (Plane).
        """
        with BuildPart() as result:
            Plane(2)
            Cylinder(radius=1, height=1, mode=Mode.SUBTRACT)
            faces().filter_by(lambda f: f.is_new()).mat = mat.red

        self.assertPart(result.part, "07886bc191e2685b2001ddc0a9f0ac97eb0e2022312c1817b6ce82a386900e06", "test_plane_subtraction", use_materials=True)

    def test_separate_planes(self):
        """
        Verify spatial tracking for disjoint 2D entities.
        """
        with BuildPart() as result:
            Plane(2)
            with Locations(Pos(X=3)):
                Plane(2)
            faces().filter_by(lambda f: f.is_new()).mat = mat.red

        self.assertPart(result.part, "2509615fd58e0c709a82d6488e0d612233aca4183de66fca945c143c3a0b61a5", "test_separate_planes", use_materials=True)

    def test_extrude_wire_selection(self):
        """
        Verify that extruding a wire results in new side faces and a new top face.
        """
        with BuildPart() as result:
            Box(2, 2, 2)
            extrude(wires().top(), op=Pos(Z=1))
            # split() is used to ensure we evaluate single faces
            faces().split().filter_by(lambda f: f.is_new()).mat = mat.red

        self.assertPart(result.part, "208d1e9ba5078288f8a4926068771856569c0f1aa2fdedd78677e9e3f0ae6414", "test_extrude_wire_selection", use_materials=True)

    def test_manual_checkpoint_persistence(self):
        """
        Verify that passing a manual checkpoint allows tracking changes across 
        multiple subsequent operations.
        """
        with BuildPart() as result:
            Box(2, 2, 2)
            cp = make_checkpoint()
            extrude(edges().top(), op=Pos(Z=1))
            extrude(edges().top(), op=Pos(Z=1))
            # Should highlight everything created since the checkpoint 'cp'
            faces().split().filter_by(lambda f: f.is_new(cp)).mat = mat.red

        self.assertPart(result.part, "6f00fd2a2069b4e0133c05bf7e522305a69a3e35c6b7cb9ec85ad9b7e1a12b4d", "test_manual_checkpoint_persistence", use_materials=True)

    def test_subdivision_topology_change(self):
        """
        Verify that subdividing existing faces results in all resulting faces 
        being 'old' (as their centers/normals remain on the original surface).
        """
        with BuildPart() as result:
            Box(2, 2, 2)
            subdivide(cuts=2)
            # In a strict spatial check, these might not be 'new' because they 
            # lie on the original planes.
            faces().split().filter_by(lambda f: f.is_new()).mat = mat.red

        self.assertPart(result.part, "2d0d6650aa9eec3ea1bb73d38155b21def230e20e1b82a78dd6a608bf1281952", "test_subdivision_topology_change", use_materials=True)

    def test_bevel_selection(self):
        """
        Verify that beveling (filleting) creates new transitional geometry 
        that is correctly identified as 'new'.
        """
        with BuildPart() as result:
            Box(2, 2, 2)
            bevel(faces().top(), radius=0.1)
            faces().split().filter_by(lambda f: f.is_new()).mat = mat.red

        self.assertPart(result.part, "edc1e93c754f53e4abcfac4c4b9cc5698f43e2b7e7456f6cdcf61c1ffa8bb0dd", "test_bevel_selection", use_materials=True)