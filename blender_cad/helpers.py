import bpy

def clear_scene(orphan_purge = False):
    """
    Completely clears the current scene of all objects and purges 
    all orphaned data blocks to free up memory.
    """
    # 1. Switch to Object Mode to avoid errors if the user is in Edit Mode
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # 2. Select all objects in the current scene
    bpy.ops.object.select_all(action='SELECT')
    
    # 3. Delete all selected objects
    # do_unlink=True ensures they are removed from all collections
    bpy.ops.object.delete(use_global=False, confirm=False)

    if not orphan_purge:
        return

    # 4. Remove other data blocks that are not objects (meshes, materials, lamps, etc.)
    # We repeat the purge multiple times because some orphans might be 
    # held by other orphans (e.g., a material held by an unused mesh)
    orphan_types = [
        'meshes', 'materials', 'textures', 'images', 
        'actions', 'armatures', 'cameras', 'lights'
    ]
    
    # Run purge multiple times to clear nested dependencies
    # (e.g., Mesh -> Material -> Texture)
    for _ in range(3):
        for data_type in orphan_types:
            data_blocks = getattr(bpy.data, data_type)
            for block in data_blocks:
                if block.users == 0:
                    data_blocks.remove(block)

    # 5. Alternative: Use the built-in operator for a deep purge
    # This is equivalent to clicking "Purge" in the Outliner
    bpy.ops.outliner.orphans_purge(
        do_local_ids=True, 
        do_linked_ids=True, 
        do_recursive=True
    )