from blender_cad import *
from tests.test_base import BaseCADTest

class TestGeometricModifiers(BaseCADTest):
    """
    Category: Topology Modification
    Tests for edge selection and geometric transformations like beveling/filleting.
    """

    def test_vertex_transform(self):
        """Verify vertex transformation on a face using Scale, Pos, and Rot."""
        with BuildPart() as result:
            Box(5, 5, 5)
            
            # Select the top faces (Z-max)
            top_faces = faces().top()
            
            # Apply a complex transformation: scale down, move, and rotate
            # This modifies the underlying vertices of the selected face
            transform(top_faces, op=Scale(0.5) * Pos(X=3) * Rot(Y=20))

        self.assertPart(result.part, "f49050e93fef5dfe65dfc2bb9f457e95c5f63537e7ba2ced913514ed6adc2ff1", "test_vertex_transform")

    def test_global_vertex_transform(self):
        """Verify complex transformation applied to all vertices in the BuildPart context."""
        with BuildPart() as result:
            # 1. Create base geometry
            Box(5, 5, 5)
            
            # 2. Add another object with its own local offset
            with Locations(Pos(Z=5)):
                Sphere(4)
            
            # 3. Apply a global transformation (Position + Rotation + Scale)
            # This should affect both the Box and the Sphere
            transform(op=Pos(X=3, Y=4, Z=5) * Rot(X=30, Y=40, Z=50) * Scale(0.1))

        self.assertPart(result.part, "e81ce80004306273739a1c596c6b881798eaca95c907e124f725799eedef718c", "test_global_vertex_transform")

    def test_vertex_transform_excluding_top_face(self):
        """Validate vertex-level transformation by excluding top face vertices to form a tapered (trapezoidal) solid."""

        with BuildPart() as result:
            # 1. Create base cube
            Box(1, 1, 1)

            # 2. Select all vertices except those belonging to the top face
            # This effectively isolates the bottom vertices
            bottom_vertices = vertices() - faces().top().vertices()

            # 3. Apply scaling only to bottom vertices
            # This expands the base while keeping the top unchanged,
            # resulting in a trapezoidal (tapered) shape
            transform(bottom_vertices, op=Scale(2))

        self.assertPart(
            result.part,
            "40d16d14e79d20978903dd367ae2426e5b3f4656ea173a679357221e4776ba33",
            "test_vertex_transform_excluding_top_face"
        )

    def test_extrude_all_geometries(self):
        """Verify extrude operation for Face, Wire, Edge, and Vertex types."""
        with BuildPart() as result:
            # Create base geometry
            Box(2, 2, 2)
            
            # 1. Face Extrude: Complex transform (Move, Rotate, Scale)
            # This creates a tapered, tilted extension on top
            extrude(faces().top(), op=Pos(X=1, Z=1) * Rot(Z=20) * Scale(0.8))
            
            # 2. Face Extrude: Scale up
            # Extruding the new top face further with expansion
            extrude(faces().top(), op=Pos(X=-1, Z=1) * Scale(1.5))
            
            # 3. Wire Extrude: Vertical offset
            # Extruding the top wire (boundary edges) to create a thin "wall" or lip
            extrude(wires().top(), op=Pos(Z=0.2))
            
            # 4. Edge Extrude: Single edge
            # Selecting the edge with maximum Y on the current top to create a "fin"
            extrude(edges().top().max_y(), op=Pos(Z=1))
            
            # 5. Vertex Extrude: All top vertices
            # This will create individual vertical edges (spikes) from each top vertex
            extrude(vertices().top(), op=Pos(Z=1))

        # The hash will represent the cumulative result of all these operations
        self.assertPart(
            result.part, 
            "c2971940abc96e6f27d511b835b4e0f7d884ab98ef4871a4d968e0c441a1d795", 
            "test_extrude_all_geometries"
        )

    def test_proportional_linear_transform_and_extrude(self):
        """Validate linear (axis-based) proportional editing for tapering shapes."""
        with BuildPart() as result:
            # 1. Create base cube
            Box(1, 1, 1)
            
            # 2. Scale XY proportionally along Z axis (creates a tapered base)
            transform(op=Scale(XY=0.5), prop_edit=LinearPropEdit(Axis.Z))
            
            # 3. Extrude top face with complex transformation (Translation + Scale) 
            # applied proportionally along the Z axis of the generated geometry
            extrude(faces().top(), op=Pos(Z=1) * Scale(XY=2), prop_edit=LinearPropEdit(Axis.Z))

        self.assertPart(
            result.part,
            "b24ce558ae5cfcb767a6f8163fbb372388d24e521ec3743ef45302813401bcbd",
            "test_proportional_linear_transform_and_extrude"
        )

    def test_proportional_math_operations_on_extrude(self):
        """Validate compound mathematical operations and clamping on proportional editing during extrusion."""
        with BuildPart() as result:
            # 1. Create base cube
            Box(1, 1, 1)
            
            # 2. Extrude top face using a complex combination of linear curves, clamping, subtraction, and division
            math_prop = (LinearPropEdit(Axis.X) + LinearPropEdit(Axis.Y).clamp(0.5, 0.8) - 0.1) / 2
            extrude(faces().top(), op=Pos(Z=1), prop_edit=math_prop)

        self.assertPart(
            result.part,
            "8a889d50d7062a2000715ce495430e907295780be2377078959f4c7a5665f2a9",
            "test_proportional_math_operations_on_extrude"
        )

    def test_proportional_box_sides_independent_masking(self):
        """Validate independent 4-side weight boundaries and multi-dependency graph resolution."""
        with BuildPart() as result:
            # 1. Create base solid
            Box(1, 1, 1)
            
            # 2. Build compound math mask
            prop_mask = make_box_sides_edit(neg_x=1.0, pos_x=0.5, neg_y=0.5, pos_y=1.0)
            
            # 3. Perform extrusion using the combined math nodes
            extrude(faces().top(), op=Pos(Z=1), prop_edit=prop_mask)

        self.assertPart(
            result.part,
            "6dad68ca29e3b7bf36f7fba74d59148d5e1e1911cf3157314cb1f0eab4f04df7",
            "test_proportional_box_sides_independent_masking"
        )

    def test_solidify_faces(self):
        """Verify solidify_faces on curved and planar face selections."""
        
        with BuildPart() as result:
            Cylinder(1, 2)
            
            # Initial position and rotation
            result.transform = Pos(2, 3, 4) * Rot(X=30, Y=40, Z=50)
            all_faces = faces()

            # Solidify cylindrical side surface
            add(
                solidify_faces(
                    all_faces.cylinders_only(),
                    height=0.5,
                    offset=0.5,
                )
            )

            # Solidify top + bottom caps
            add(
                solidify_faces(
                    all_faces.top() + all_faces.bottom(),
                    height=0.5,
                    offset=0.5,
                )
            )

        self.assertPart(
            result.part,
            "c10c12281d015f61df022ad074b5d40d5786041cfa26b79e17a34dae20fbf072",
            "test_solidify_faces"
        )

    def test_proportional_radial_transform(self):
        """Validate radial (distance-based) proportional editing using a bubble of influence."""
        with BuildPart() as result:
            # 1. Create base cube
            Box(1, 1, 1)
            
            # 2. Apply scaling to vertices near the corner (XY=1)
            # Vertices closer to the origin (1, 1, 0) will be scaled more significantly
            transform(op=Scale(XY=0.5), prop_edit=RadialPropEdit(Pos(XY=1), radius=2))

        self.assertPart(
            result.part,
            "f502fdb86e1f1e2fd6307ac48dc0252a2d2285b6494941b2f2ea43a3f86a82e8",
            "test_proportional_radial_transform"
        )

    def test_proportional_radial_plane_bending(self):
        """Validate radial bending of a subdivided plane to create a peaked surface."""
        with BuildPart() as result:
            # 1. Create a flat 2x2 plane
            Plane(2)
            
            # 2. Increase vertex density to allow for smooth deformation
            subdivide(cuts=6)
            
            # 3. Lift the center proportionally. 
            # Since no origin is provided to RadialPropEdit, it defaults to (0,0,0).
            # Using Falloff.LINEAR creates a cone-like peak.
            transform(
                op=Pos(Z=1), 
                prop_edit=RadialPropEdit(radius=1.0, falloff=Falloff.LINEAR)
            )

        self.assertPart(
            result.part,
            "a60593a767a4985c89e64f9bcf9b1a95e1d5b33455e12a7452bbfb4d699cbc17",
            "test_proportional_radial_plane_bending"
        )

    def test_delete_all_geometries(self):
        """Verify delete operation for Face, Wire, Edge, and Vertex types."""
        with BuildPart() as result:
            # 1. Face Delete: Remove a specific face
            # This will open the box, removing the top-most face in the Y direction
            with BuildPart(offset=Pos(X=0)):
                Box(2, 2, 2)
                delete(faces().top().max_y())

            # 2. Wire Delete: Remove a wire (closed loop of edges)
            # Deletes the boundary loop of the top face
            with BuildPart(offset=Pos(X=3)):
                Box(2, 2, 2)
                delete(wires().top().max_y())

            # 3. Edge Delete: Remove a single edge
            # Deleting one edge will also remove the faces connected to it
            with BuildPart(offset=Pos(X=6)):
                Box(2, 2, 2)
                delete(edges().top().max_y())

            # 4. Vertex Delete: Remove a single vertex
            # This removes the vertex and all edges/faces sharing it
            with BuildPart(offset=Pos(X=9)):
                Box(2, 2, 2)
                delete(vertices().top().max_y())

            # 5. Iterative Vertex Delete: Sequential removal
            # Removing one vertex, then selecting the next available top-Y vertex
            with BuildPart(offset=Pos(X=12)):
                Box(2, 2, 2)
                delete(vertices().top().max_y())
                delete(vertices().top().max_y())

        self.assertPart(
            result.part, 
            "dda30021dc1272acdb71424dc18227ec5b0bb06fc1dda4a2a86cb71c21c8d7a5", 
            "test_delete_all_geometries"
        )

    def test_edge_selection_and_bevel(self):
        """Verify that bevel applies correctly to all edges of a box."""
        with BuildPart() as result:
            Box(5, 5, 5)
            # Select all edges and apply a 0.1 radius bevel
            bevel(radius=0.1)
            
        self.assertPart(result.part, "6165eae68ec63c13703f2ee3a97f30de72fb1916482b1e1fc77cc8434504e174", "test_edge_selection_and_bevel")

    def test_bend_operation(self):
        """Verify the bend operation along the X axis with pre-subdivision."""
        with BuildPart() as result:
            # Create a thin plate
            Box(2, 1, 0.1)

            sel_faces=faces().bottom() + faces().top()

            # Apply a position, rotation, and scale
            result.transform = Pos(X=1, Y=2, Z=3) * Rot(X=30, Y=40, Z=50) * Scale(X=1.5)
            
            # Subdivide specific faces to provide enough geometry for a smooth bend
            # We select top and bottom faces to ensure the vertical edges get cuts
            subdivide(faces=sel_faces, cuts=4)
            
            # Apply a 30-degree bend along the X axis
            bend(angle=30, axis=Axis.Y)
            
        self.assertPart(
            result.part, 
            "9961604b9f10b6e5a04fa19ca72170997cef7b7a4b385ece1fd7e01ddbdd243c", 
            "test_bend_operation"
        )

    def test_wrap_operation(self):
        """Verify the wrap operation of a private cylinder onto a plane with material assignment."""
        with BuildPart() as result:
            # 1. Create a private cylinder to be used as the source for wrapping
            with BuildPart(mode=Mode.PRIVATE) as cyl:
                Cylinder(radius=1, height=2, segments=64)
            
            # 2. Define the target plane
            Plane(1)
            
            # 3. Assign material properties to the result
            # Note: Assuming 'mat' is an available global or imported color/material registry
            result.mat = mat.red
            
            # 4. Perform the wrap operation
            # This projects the cylindrical surface onto the current workplane
            # using the face location and specified segments/offset
            wrap(
                cyl, 
                loc=cyl.faces().cylinders_only()[0].at(uv), 
                segments=4, 
                offset=0.01
            )
            
            # 5. Explicitly add the cylinder back into the main part if needed, 
            # or combine it with the wrapped geometry
            add(cyl, mode=Mode.JOIN)

        self.assertPart(
            result.part, 
            "f077e2cd23b5038fa327276438b108a36b82be50929a9ce2cbed1052fcf3875a",
            "test_wrap_operation"
        )
        