from skill_manager import skill_manager
from hook_manager import hook_manager
import os

print(f"--- Argent System Verification ---")

# 1. Check Skill
skills = skill_manager.list_skills()
expert_skill = next((s for s in skills if s['name'] == 'ExpertArchitect'), None)
if expert_skill:
    print(f"[OK] Skill 'ExpertArchitect' found: {expert_skill['description']}")
else:
    print(f"[FAIL] Skill 'ExpertArchitect' NOT found.")

# 2. Check Plugins (Hooks)
# We need to reload hooks to see the new ones if they were added while Argent was running (though here we just check the disk and the logic)
# Since Argent reloads hooks automatically on write_file in the hooks dir, we should be fine.
hook_manager.reload_plugins()
plugins = [p.__name__ for p in hook_manager._plugins]
print(f"Loaded plugins: {plugins}")

if 'skill_activator' in plugins:
    print(f"[OK] Plugin 'skill_activator' loaded.")
else:
    print(f"[FAIL] Plugin 'skill_activator' NOT loaded.")

if 'audit_plugin' in plugins:
    print(f"[OK] Plugin 'audit_plugin' loaded.")
else:
    print(f"[FAIL] Plugin 'audit_plugin' NOT loaded.")

# 3. Check Commands
commands = hook_manager.get_custom_commands()
if 'audit' in commands:
    print(f"[OK] Command '/audit' registered.")
else:
    print(f"[FAIL] Command '/audit' NOT registered.")
