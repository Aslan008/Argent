---
name: blender-mcp
description: Guide and best practices for controlling Blender 3D through the Blender MCP server in Argent. Trigger when creating 3D models, procedural meshes, shader materials, scene lighting, animations, exporting FBX/glTF, or interacting with a running Blender instance.
---

# Blender MCP Playbook for Argent

When the `blender` MCP server is connected, Argent can inspect and drive a live Blender session in real-time.

## Key Tools Overview

Call via `call_mcp_tool("blender", "<tool_name>", {<arguments>})`:

1. **Inspection & Context:**
   - `get_addon_status()`: verify connection, protocol version, and Blender version.
   - `get_scene_info(user_prompt)`: list objects, types, hierarchy, active camera, selection. Always run first!
   - `get_object_info(object_name, user_prompt)`: get detailed transform, mesh data, modifier stack, material slots.
   - `bpy_api_lookup(query)`: search official Blender Python API documentation.

2. **Code Execution:**
   - `execute_blender_code(code, user_prompt)`: run Python code inside Blender's context (`import bpy` is available).
   - `describe_node_type(bl_idname)`: inspect node sockets and property schema before wiring shader/geometry nodes.

3. **Verification:**
   - `get_viewport_screenshot(user_prompt)`: capture current 3D viewport to inspect visual appearance and layout.

4. **Assets & Textures:**
   - `search_polyhaven_assets(...)`, `download_polyhaven_asset(...)`, `set_texture(...)`
   - `search_polypizza_models(...)`, `download_polypizza_model(...)`
   - `export_scene(filepath, format)`: export scene/selection to FBX, OBJ, glTF/GLB.

---

## Hard Rules for `execute_blender_code`

### 1. Nodes by Type, Never by Name
Blender UI names can be localized. `nodes["Principled BSDF"]` will fail on non-English locales or custom node names.
```python
# GOOD:
node = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")

# BAD:
node = mat.node_tree.nodes["Principled BSDF"]
```

### 2. Materials require `use_nodes = True`
Setting `material.diffuse_color` only affects viewport display in Solid mode; rendered output and game engines use the shader node tree.
```python
mat = bpy.data.materials.new(name="M_Prop")
mat.use_nodes = True
bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
bsdf.inputs["Base Color"].default_value = (0.8, 0.2, 0.1, 1.0)
bsdf.inputs["Roughness"].default_value = 0.4
obj.data.materials.append(mat)
```

### 3. Dynamic RNA Enums
Never guess or hardcode enum identifiers that may vary across Blender LTS versions (e.g., 4.x vs 5.x). Query valid items dynamically:
```python
# Inspect valid items:
valid_formats = [i.identifier for i in bpy.context.scene.render.image_settings.bl_rna.properties["file_format"].enum_items]
```

### 4. Verify Every Multi-Step Change
Always call `get_scene_info()` or `get_viewport_screenshot()` after manipulating geometry, materials, or lighting to confirm the desired outcome was achieved without silent errors.
