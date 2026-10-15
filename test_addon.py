import bpy
# Проверяем, доступен ли аддон
bpy.context.preferences.addons["blender_mcp"]
print("ADDON_AVAILABLE")
