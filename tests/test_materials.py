from blender_cad import *
from tests.test_base import BaseCADTest

class TestMaterialsAndVisuals(BaseCADTest):
    """
    Category: Material Properties & Visual Overrides
    Tests for material assignment, inheritance, and face-specific styling.
    """

    def test_simple_pbr(self):
        """Test basic PBR material without any layers."""
        gold = mat.PBR(base_color=(1.0, 0.75, 0.1, 1.0), metallic=1.0, roughness=0.5)
        self.assertMaterial(gold, "724b405cf27749911630d1c5d55a87ecf17c85644c435779665f6f6c81ee54fe", "test_simple_pbr")

    def test_layer_addition(self):
        """Test simple additive blending of a layer."""
        gold = mat.PBR(base_color=(1.0, 0.75, 0.1, 1.0))
        dirt = mat.DirtImg()
        my_mat = gold + dirt * 0.5
        self.assertMaterial(my_mat, "7138e2c8266040dbfbf0c74c1a18de294c617e5544e7b5d8fc72785170780302", "test_layer_addition")

    def test_factor_scaling(self):
        """Test layer blending with a constant float factor."""
        gold = mat.PBR(base_color=(1.0, 0.75, 0.1, 1.0))
        glow = mat.Glow(emission_color=(1.0, 0.0, 0.0))
        my_mat = gold + glow * 0.7
        self.assertMaterial(my_mat, "3918e19baf95b6caf000b50bd4f38ecc67003de0c99cf575340ac7a7617722d0", "test_factor_scaling")

    def test_variable_external(self):
        """Test material with an external variable (UI-tweakable factor)."""
        gold = mat.PBR(base_color=(1.0, 0.75, 0.1, 1.0))
        glow = mat.Glow()
        # Factor is controlled by a named variable
        my_mat = gold + glow * mat.Var("intensity", 0.5) * 0.2
        self.assertMaterial(my_mat, "85c68d35ee9f53e0e7d89b88e65f56bf7ab797bdac3139f879140e2fa0a0c575", "test_variable_external")

    def test_variable_internal(self):
        """Test custom MatLayer with internal variables."""
        class MyGlow(mat.Layer):
            def build(self, ctx):
                ctx.channels.emission_color = ctx.blend(
                    ctx.channels.emission_color, (1.0, 0.0, 0.0), factor=mat.Var("f", 1)
                )
                ctx.channels.emission_strength = ctx.blend(
                    ctx.channels.emission_strength, 10.0, factor=mat.Var("f", 1)
                )

        gold = mat.PBR(base_color=(1.0, 0.75, 0.1, 1.0))
        my_mat = gold + MyGlow()
        self.assertMaterial(my_mat, "2f0c96329812c688f38b8346fcb792e5025781d11865ec11fdf09f5d22dcfbbc", "test_variable_internal")

    def test_base_color_blending(self):
        """
        Test blending of two distinct base colors using a factor.
        Checks if RGBA tuples are correctly handled by the blend system.
        """
        # Blue base material
        blue_mat = mat.PBR(base_color=(0.0, 0.0, 1.0, 1.0))
        
        class RedOverlay(mat.Layer):
            def build(self, ctx):
                # Blend current color with Red
                ctx.channels.base_color = ctx.blend(
                    ctx.channels.base_color, 
                    (1.0, 0.0, 0.0, 1.0), 
                    factor=0.5
                )

        # Result should be a purple-ish material (0.5, 0.0, 0.5)
        my_mat = blue_mat + RedOverlay()
        self.assertMaterial(my_mat, "ce4c059f474a58e9b5acaba3cfdc261957badc2f2efc24c5fe965a8902a32177", "test_base_color_blending")

    def test_meta_naming(self):
        """Test if Meta layer correctly sets the material name."""
        gold = mat.PBR(base_color=(1.0, 0.75, 0.1, 1.0))
        mat_name = "RoyalGold"
        my_mat = gold + mat.Meta(name=mat_name)
        
        self.assertMaterial(my_mat, "5e195a89a97e7d747024c07d48b1f878d8b916cffe217ea661bd0b034dc019d3", "test_meta_naming")
        self.assertEqual(build_material(my_mat).name, mat_name)

    def test_nested_composition(self):
        """Test complex nested layering with group factors."""
        gold = mat.PBR(base_color=(1.0, 0.75, 0.1, 1.0), metallic=1.0)
        dirt = mat.DirtImg()
        rust = mat.RustImg()
        # (Dirt + Rust) combined and then scaled by 0.5
        my_mat = gold + (dirt + rust) * 0.5
        self.assertMaterial(my_mat, "4a96c147e5fb9b419a4b2fc7b0940656dd9f8c477b18f0547595fe16036b1915", "test_nested_composition")

    def test_mixed_types_in_channel(self):
        """
        Test switching from a constant value to a texture expression
        within the same PBR channel (Metallic).
        """
        class MixedMetallicMat(mat.Layer):
            def build(self, ctx) -> None:
                # 1. Set initial constant value
                ctx.channels.metallic = 0.0
                
                # 2. Blend with a constant
                ctx.channels.metallic = ctx.blend(ctx.channels.metallic, 1.0, factor=0.5)
                
                # 3. Blend with a texture (implicit build)
                # This tests the transition from float/socket to Texture-backed socket
                dirt_tex = mat.Tex(image_path="./assets/test/dirt.png")
                ctx.channels.metallic = ctx.blend(ctx.channels.metallic, dirt_tex, factor=0.3)

        base = mat.PBR()
        my_mat = base + MixedMetallicMat()
        
        self.assertMaterial(my_mat, "3d57b33ff689fa29db320b79e9a15b0bba75326fab24b507a4fdbd706f85d05a", "test_mixed_types_in_channel")

    def test_pbrmat_subtraction(self):
        """
        Test mathematical subtraction of two PBRMat layers using BlendMode.SUBTRACT.
        Calculates: 1.0 (Gold) - (0.7 * 1.0 factor) = 0.3
        """
        # Base material with full metallic
        base = mat.PBR(
            metallic=1.0, 
        )
        
        sub_layer = mat.PBR(
            metallic=0.7, 
            mode=BlendMode.SUBTRACT
        )
        
        # Composition: Resulting metallic should be 0.3
        my_mat = base + sub_layer
        
        self.assertMaterial(my_mat, "9e7efbe9ad3d3a52e02f07a54168ff0dcbf6acb99e71def15271d2d2992602b4", "test_pbrmat_subtraction")

    def test_pbrmat_multiply_scalar_by_rgba(self):
        """
        Test multiplying a scalar FLOAT channel by an RGBA color.
        Calculates: 1.0 (Base Metallic) * (0.5, 0.6, 0.7, 0.8) => 0.5
        """
        # Base material with float metallic
        base = mat.PBR(
            metallic=1.0,
        )
        
        # Multiplication layer with RGBA metallic
        # Note: Even though Metallic is usually a scalar, our system allows 
        # blending with RGBA, promoting the channel to a vector.
        mult_layer = mat.PBR(
            metallic=(0.5, 0.6, 0.7, 0.8),
            mode=BlendMode.MULTIPLY
        )
        
        my_mat = base + mult_layer

        # Resulting metallic should be 0.5 (float), remaining components should be truncated
        self.assertMaterial(my_mat, "a051a3cbabd3823360b5a2cb71490dd07a94beaacd21cfe93d8ebb54ec85629c", "test_pbrmat_multiply_scalar_by_rgba")

    def test_material_assignment_by_z_groups(self):
        """
        Verify that materials can be assigned at the BuildPart level (default slot)
        and overridden per-face using Z-axis grouping.
        """
        # Initialize with a default Blue material (Slot 0)
        with BuildPart(mat=mat.blue) as result:
            # Create a base geometry
            Box(10, 10, 1)
            
            # Group all faces by their position along the Z-axis
            # Expected: 3 groups (Bottom [0], Sides [1], Top [2])
            groups = faces().group_by(Axis.Z)
            
            # Override material for the bottom face (Group 0) to Red
            groups[0].mat = mat.red
            
            # Override material for the side faces (Group 1) to Green
            groups[1].mat = mat.green
            
            # The top face (Group 2) is left untouched, 
            # so it should inherit the default Blue material from BuildPart context.

        self.assertPart(result.part, "b70451b78ca6b075146ee35fd232754accfc26081c0f57605c856384b9bbcd6d", "test_material_assignment_by_z_groups", use_materials=True)

    def test_nested_material_logic_and_overrides(self):
        """
        Verify material inheritance in nested BuildPart contexts,
        explicit face overrides, and runtime default material changes.
        """
        # Start with a default Blue material
        with BuildPart(mat=mat.blue) as result:
            Box(10, 10, 1) # This box inherits Blue
            
            # Nested context with JOIN mode
            with BuildPart(mode=Mode.JOIN):
                with Locations(Pos(Z=1)):
                    Box(5, 5, 1)
                
                # Override the top-most face of the current context to Red
                faces().sort_by(Axis.Z)[-1].mat = mat.red
                
                # Change the default material for any subsequent objects in this context
                set_default_mat(mat.green)
                
        self.assertPart(result.part, "74afac8a0cb7071e993849bd7593f965ffe9b4ccf5e576603285aa1d6d666173", "test_nested_material_logic_and_overrides", use_materials=True)

    def test_procedural_dirt(self):
        """Test procedural dirt layer with custom scale and intensity."""
        base = mat.PBR(base_color=(0.5, 0.5, 0.5, 1.0), roughness=0.3)
        # Dirt with high scale (small spots) and 80% visibility
        dirt_layer = mat.Dirt(scale=25.0) * 0.8
        my_mat = base + dirt_layer
        
        self.assertMaterial(my_mat, "fb53f345ed86a5fc840c48f13ede7dbde25a0e6ced213158250a79325bacc327", "test_procedural_dirt")

    def test_heavy_rust_on_metal(self):
        """Test rust layer overlapping a metallic surface."""
        steel = mat.PBR(base_color=(0.8, 0.8, 0.8, 1.0), metallic=1.0, roughness=0.2)
        rust = mat.Rust()
        # Rust usually completely overrides metallic properties where it appears
        my_mat = steel + rust * 0.6
        
        self.assertMaterial(my_mat, "d60916d48f16120dae365c08670992ffb289259228deda37c6763d5aa39e5861", "test_heavy_rust_on_metal")

    def test_stacked_procedurals(self):
        """Test stacking multiple procedural effects."""
        base = mat.PBR(base_color=(0.2, 0.2, 0.2, 1.0))
        # Mix of rust and then some dirt on top
        my_mat = base + mat.Rust() * 0.4 + mat.Dirt(scale=5.0) * 0.3
        
        self.assertMaterial(my_mat, "a8ed63932bf870c7673456354719e10b3ae481716f01616ed5ffa7b4c386e82f", "test_stacked_procedurals")

    def test_clear_glass(self):
        """Test pure glass layer with standard IOR."""
        # Standard glass: IOR 1.5, full transmission, low roughness
        glass = mat.Glass(roughness=0.02)
        
        # In Blender 4.x this should connect to 'Transmission Weight' and 'IOR'
        self.assertMaterial(glass, "1abacd61f396abc5ff9b91c4ec01c65eb482faf38162696522d1a108181f81d8", "test_clear_glass")

    def test_frosted_glass_with_dust(self):
        """Test frosted glass combined with procedural dust residue."""
        # Frosted tinted glass
        base_glass = mat.Glass(
            color=(0.7, 0.9, 1.0, 1.0), 
            roughness=0.25, 
            ior=1.45
        )
        # Subtle dust layer with fine scale
        dust = mat.Dust(density=0.4, scale=40.0) * 0.5
        
        my_mat = base_glass + dust
        
        # This checks if DustProc correctly attenuates the Transmission channel
        self.assertMaterial(my_mat, "37a62a06fe5dea86765b9da7997867b885f52924857688fc41aa44e1fb5d61cc", "test_frosted_glass_with_dust")

    def test_cracked_glass(self):
        """Test composition of Glass base with procedural cracks."""
        # Setup: Glass base with a specific thickness and distortion for cracks
        glass_base = mat.Glass(roughness=0.05)
        cracks = mat.Cracks(
            scale=5.0, 
            thickness=0.01, 
            distortion=0.8, 
            dirt_color=(0.05, 0.05, 0.05, 1.0)
        )
        
        composite = glass_base + cracks * 0.6
        
        self.assertMaterial(
            composite, 
            "567aad4b565d2e4fcafa15a8780c973209dbe55814971dc459d57d1ce40ef3b6", 
            "test_cracked_glass"
        )

    def test_laser_grid_cube(self):
        """Test 3D Laser Grid application on a cube-like structure."""
        # Setup: High-intensity red laser grid
        # scale=2.0 means larger cells, thickness=0.05 for visible lines
        laser_material = mat.LaserGrid(
            color=(1.0, 0.0, 0.0, 1.0), 
            scale=2.0, 
            thickness=0.05, 
            glow_intensity=30.0
        )
        
        # In this framework, the layer itself acts as the material 
        # if not added to a base (it replaces the default PBR channels)
        self.assertMaterial(
            laser_material, 
            "580477849fe431cc3316cca784c90cc19c456034adb100ee319d601eb6319f5d",
            "test_laser_grid_cube"
        )

    def test_material_shoot_projection(self):
        """Test shooting a texture from a child part and applying it to a surface."""
        with BuildPart() as result:
            # 1. Create the target surface (The 'Monitor')
            Plane(2)
            
            # 2. Create the source object in a private context
            with BuildPart(mode=Mode.PRIVATE) as child:
                Cylinder(radius=1, height=2)
                # Assign a red glow to ensure high contrast in the shot
                child.mat = mat.Glow(emission_color=(1, 0, 0, 1))
                
                # Position camera at a distance looking back at the cylinder
                camera_loc = (child.loc * Pos(XYZ=4)).look_at(child.loc, flip_z=True)
                
                # Capture the visual representation of the cylinder
                cam_tex = child.shoot(camera_loc)
            
            # 3. Compose the final material: Green background + the captured texture
            # Using Mix mode with 0.8 factor as per your logic
            material = mat.green + mat.PBR(base_color=cam_tex, mode=BlendMode.MIX) * 0.8
            
            # 4. Apply to the top face of the Plane
            faces().top().mat = material

        # 5. Assert against the stable hash. 
        # Note: If the pixels of the cylinder change, this hash will now correctly fail.
        self.assertMaterial(
            material, 
            "122e7374dc6b3da1e41e9355e884cfd240234250ef8a4c2c3dbaf21f8e547a8c", 
            "test_material_shoot_projection",
            hash_image_pixels=True
        )


class TestPredefinedMaterials(BaseCADTest):
    # --- METALS ---

    def test_iron(self):
        """Test Iron procedural material."""
        self.assertMaterial(
            mat.iron, 
            "6369376a69301a2b6be8e94bbb732fe626e73b775b3612eb5bdb9950921fbf05", 
            "test_iron"
        )

    def test_gold(self):
        """Test Gold procedural material."""
        self.assertMaterial(
            mat.gold, 
            "060307f8a397945b680a2af2183ae6a0b508ef818a67c3103ae5752dd7be4b7a", 
            "test_gold"
        )

    def test_copper(self):
        """Test Copper procedural material."""
        self.assertMaterial(
            mat.copper, 
            "79f64a9402dbe60dea769f511ae391db9052729c891197ebba2272c6ca9dfd8b", 
            "test_copper"
        )

    # --- WOOD ---

    def test_wood_oak(self):
        """Test Oak Wood procedural material."""
        self.assertMaterial(
            mat.wood_oak, 
            "fdc683996845ccf47ff3043f142e8c691a4cd707fb97f871a63d046301ebf671", 
            "test_wood_oak"
        )

    def test_wood_pine(self):
        """Test Pine Wood procedural material."""
        self.assertMaterial(
            mat.wood_pine, 
            "e9ebed63ffac7d6753c4f92e47590add17a67741fe4ddf0faf75b3cf99b03b03", 
            "test_wood_pine"
        )

    # --- CONSTRUCTION ---

    def test_concrete(self):
        """Test Concrete procedural material."""
        self.assertMaterial(
            mat.concrete, 
            "91f50afaac852ed0c7cd7fdeeee9abbccd10051497adb2444763800578eaba3f", 
            "test_concrete"
        )

    def test_brick_red(self):
        """Test Red Brick procedural material."""
        self.assertMaterial(
            mat.brick_red, 
            "00e3017b4baa3c367c8b56ee73c7a9d264c844e514c0a12a098e40df694e0a70", 
            "test_brick_red"
        )

    # --- NATURE ---

    def test_sand_desert(self):
        """Test Desert Sand procedural material."""
        self.assertMaterial(
            mat.sand_desert, 
            "c3cb29efd4dd1109e7c8a0aef558ec9fa814a03db5c48bb132c32a209e5241ef", 
            "test_sand_desert"
        )

    # --- FABRIC & LEATHER ---

    def test_denim(self):
        """Test Denim Fabric procedural material."""
        self.assertMaterial(
            mat.denim, 
            "764a09100938c8a7804e7b83957303e55846f60a2f8fbece83178ce3ffc40ca5", 
            "test_denim"
        )

    def test_leather_brown(self):
        """Test Brown Leather procedural material."""
        self.assertMaterial(
            mat.leather_brown, 
            "6cd17a6e2fcbccb0304aaac4533d148075bf827ca2d12f77d18b15a7f0dfd2be", 
            "test_leather_brown"
        )

    # --- PLASTIC ---

    def test_plastic_glossy_red(self):
        """Test Glossy Red Plastic procedural material."""
        self.assertMaterial(
            mat.plastic_glossy_red, 
            "dc2cc4b8cc32310597fb4942cecf50c6541d2b40cd0b1962b800d27c6f0c58ea", 
            "test_plastic_glossy_red"
        )

    def test_plastic_matte_grey(self):
        """Test Matte Grey Plastic procedural material."""
        self.assertMaterial(
            mat.plastic_matte_grey, 
            "2d9c0eb170cda6eb9e965abbde37c7a558f9e41731d3223ae77e4d356e13a4bd", 
            "test_plastic_matte_grey"
        )