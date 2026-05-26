import questionary
from project_manager import ProjectManager
from ui import console
from logger import get_logger

log = get_logger("tools")

def add_project_task(description: str) -> str:
    """Add a task to the current project plan."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project. Use /project to start one."
    task_id = pm.add_task(description)
    return f"Task {task_id} added: '{description}'. If you have no more tasks to add, stop calling tools and reply 'DONE'."

def complete_project_task(task_id: int, summary: str) -> str:
    """Mark a project task as completed. You MUST provide a summary of what you did."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project."
    status = pm.data.get("status", "")
    if status != "executing":
        return f"Error: Cannot complete tasks during '{status}' phase. You should only call add_project_task now."
    pm.complete_task(task_id, summary)
    return f"Task {task_id} completed. Summary saved."

def plan_work_changes(strategy: str, files_to_edit: str, files_to_create: str) -> str:
    """Submit the investigation plan for an existing project."""
    pm = ProjectManager()
    if not pm.active or pm.data.get("mode") != "work":
        return "Error: No active /work session."
        
    edit_list = [f.strip() for f in files_to_edit.split(',') if f.strip()] if files_to_edit else []
    create_list = [f.strip() for f in files_to_create.split(',') if f.strip()] if files_to_create else []
    
    if create_list and not pm.data.get("work_auto_mode", False):
        print(f"\n[bold yellow]Agent requesting to create NEW files for /work:[/bold yellow] {', '.join(create_list)}")
        approved = questionary.confirm("Do you want to allow these files to be created?").ask()
        if not approved:
            return f"Error: User denied creation of {', '.join(create_list)}. Revise your plan to ONLY modify existing files, without creating these new ones. Call plan_work_changes again."
        
    pm.data["work_strategy"] = strategy
    pm.data["files_to_edit"] = edit_list
    pm.data["files_to_create"] = create_list
    pm.set_status("work_planning")
    return "Plan accepted. Moving to task generation phase."

def add_work_task(description: str) -> str:
    """Add a micro-task for the current /work session."""
    pm = ProjectManager()
    if not pm.active or pm.data.get("mode") != "work":
        return "Error: No active /work session."
    task_id = pm.add_task(description)
    return f"Work task {task_id} added: '{description}'. If you have no more tasks to add, stop calling tools and reply 'DONE'."

def list_project_tasks() -> str:
    """View the current project status with all tasks and their summaries."""
    pm = ProjectManager()
    if not pm.active:
        return "No active project."
    data = pm.data
    result = f"Project: {data['objective']}\nProgress: {pm.get_progress_display()}\n\n"
    for t in data["tasks"]:
        marker = "[DONE]" if t["status"] == "completed" else "[ ]"
        result += f"{t['id']}. {marker} {t['description']}\n"
        if t.get("result_summary"):
            result += f"   Result: {t['result_summary']}\n"
    return result.strip()

def write_project_spec(spec: str) -> str:
    """Write the detailed technical specification for the current project."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project."
    pm.set_spec(spec)
    return "Project specification saved successfully."

def write_project_architecture(architecture: str, files: str) -> str:
    """Write the high-level architecture map for the project."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project."
    file_list = [f.strip() for f in files.split(',') if f.strip()]
    pm.set_architecture(architecture, file_list)
    return f"Architecture saved. Files to detail: {', '.join(file_list) if file_list else 'NONE - provide the files parameter!'}"

def write_file_spec(filename: str, spec: str) -> str:
    """Write a detailed specification for a single file."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project."
    pm.set_file_spec(filename, spec)
    pending = pm.get_pending_spec_files()
    if pending:
        return f"Spec for '{filename}' saved. Remaining files without spec: {', '.join(pending)}"
    else:
        return f"Spec for '{filename}' saved. All files now have detailed specs!"
