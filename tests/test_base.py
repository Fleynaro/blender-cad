import unittest
import os
import inspect

from blender_cad import *

class BaseCADTest(unittest.TestCase):
    """
    Base class for CAD unit tests.
    """
    @property
    def UPDATE_HASHES(self):
        return os.environ.get("UPDATE_HASHES") == "True"
    
    @property
    def SHOW_GEOMETRY_IF_FAILED(self):
        return os.environ.get("SHOW_GEOMETRY_IF_FAILED") == "True"

    def _update_source_hash(self, old_hash: str, new_hash: str):
        """
        Locates the caller's source file and replaces the hash string, 
        handling multi-line function calls.
        """
        # Get the caller's frame (the test method calling assertPart)
        frame_info = inspect.stack()[2] 
        file_path = frame_info.filename
        # Start searching from the line where the function was called
        start_line_idx = frame_info.lineno - 1

        if not os.path.exists(file_path):
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Check the call line and the next few lines for multi-line support
        search_range = 10 
        updated = False
        
        for i in range(start_line_idx, min(start_line_idx + search_range, len(lines))):
            if old_hash in lines[i]:
                lines[i] = lines[i].replace(old_hash, new_hash)
                updated = True
                break 

        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"\n[AUTO-UPDATE] Hash updated in {os.path.basename(file_path)} around line {start_line_idx + 1}")
        else:
            print(f"\n[AUTO-UPDATE] Warning: Could not find hash pattern near line {start_line_idx + 1}")

    def assertPart(self, result_part: Part, expected_hash: str, test_name: str, use_materials=False):
        """
        Validates the generated part against a hash and optionally displays it.
        """
        # Check if we are in update mode via environment variable
        actual_hash = result_part.hash(use_materials=use_materials)
        try:
            self.assertEqual(
                actual_hash, 
                expected_hash, 
                f"Hash mismatch in {test_name}! Expected {expected_hash}, got {actual_hash}"
            )
        except AssertionError as e:
            # Display the part if the hash does not match
            if self.SHOW_GEOMETRY_IF_FAILED:
                result_part.show(name=test_name, hide=True)
            if self.UPDATE_HASHES:
                self._update_source_hash(expected_hash, actual_hash)
            raise e
    
    def assertMaterial(self, layer: mat.Layer, expected_hash: str, test_name: str, hash_image_pixels = False):
        """
        Validates the generated bpy material against a stable hash.
        If the hash mismatch occurs, creates a 3D box with the material for visual inspection.
        """
        # 1. Build the actual Blender material from the layer expression
        material = build_material(layer)
        
        # 2. Calculate the hash of the resulting node tree
        actual_hash = bpy_material_hash(material, hash_image_pixels)
        
        try:
            self.assertEqual(
                actual_hash, 
                expected_hash, 
                f"Material hash mismatch in {test_name}! Expected {expected_hash}, got {actual_hash}"
            )
        except AssertionError as e:
            # If the test fails and visual debugging is enabled
            if self.SHOW_GEOMETRY_IF_FAILED:
                # Create a 1x1x1 box and assign the failed material to it
                with BuildPart(mat=layer) as result:
                    Box(1, 1, 1)
                
                # Show the part in the viewer/Blender scene
                result.part.show(name=test_name, hide=True)
            if self.UPDATE_HASHES:
                self._update_source_hash(expected_hash, actual_hash)
            raise e