import questionary
from project_manager import ProjectManager
from ui import print_system, print_error
from config import get_obsidian_vault
from prompts import (
    build_spec_prompt, build_work_investigation_prompt,
    build_work_planning_prompt, build_planning_prompt, build_architecture_prompt
)

class ProjectOrchestrator:
    def __init__(self, agent):
        self.agent = agent
        self.pm = ProjectManager()
        self.last_task_id = None
        self.task_retries = 0
        self.MAX_TASK_RETRIES = 3

    def start_project(self, proj_prompt: str) -> tuple[bool, str]:
        """Start a new project. Prompts user for parameters."""
        run_research = questionary.confirm("Run Deep Research (Phase 0) to gather up-to-date context before planning?").ask()
        tdd_mode = questionary.confirm("Enable TDD Mode? (AI will write tests BEFORE code)").ask()
        
        use_obsidian = False
        if get_obsidian_vault():
            use_obsidian = questionary.confirm("Enable Obsidian integration for this project? (Create notes instead of normal files)").ask()
        
        self.pm.destroy()
        self.last_task_id = None
        self.task_retries = 0
        
        if run_research:
            self.pm.create(proj_prompt, status="researching", tdd_mode=tdd_mode, use_obsidian=use_obsidian)
            user_input = (
                f"You are the Phase 0 Research Agent for the new project: '{proj_prompt}'.\n\n"
                f"Your task is to gather the MAXIMUM amount of up-to-date information, best practices, and API references required to build this project.\n"
                f"You MUST call `run_deep_research(objective='...')` right now to search the web.\n"
                f"When the research is complete, output a detailed markdown report of everything you found.\n"
                f"DO NOT write code or architecture yet. Just gather information."
            )
            print_system(f"[Brain] Phase 0: Starting Deep Research...")
        else:
            self.pm.create(proj_prompt, status="specifying_architecture", tdd_mode=tdd_mode, use_obsidian=use_obsidian)
            user_input = build_architecture_prompt(proj_prompt)
            print_system(f"[Brain] Phase 1a: Designing architecture...")
            
        return True, user_input

    def start_work(self, work_prompt: str, auto_mode: bool = False) -> tuple[bool, str]:
        """Start codebase modification work."""
        run_research = questionary.confirm("Run Deep Research (Phase 0) to gather up-to-date context before investigation?").ask()
        tdd_mode = questionary.confirm("Enable TDD Mode? (AI will write tests BEFORE code)").ask()
        
        use_obsidian = False
        if get_obsidian_vault():
            use_obsidian = questionary.confirm("Enable Obsidian integration for this work task?").ask()
            
        self.pm.destroy()
        self.last_task_id = None
        self.task_retries = 0
        
        if run_research:
            self.pm.create(work_prompt, status="work_researching", mode="work", auto_mode=auto_mode, tdd_mode=tdd_mode, use_obsidian=use_obsidian)
            user_input = (
                f"You are the Phase 0 Research Agent for the codebase modification task: '{work_prompt}'.\n\n"
                f"Your task is to gather the MAXIMUM amount of up-to-date information, best practices, and API references required for this task.\n"
                f"You MUST call `run_deep_research(objective='...')` right now to search the web.\n"
                f"When the research is complete, output a detailed markdown report of everything you found.\n"
                f"DO NOT write code or investigate files yet. Just gather information."
            )
            print_system(f"[Brain] Phase 0: Starting Deep Research...")
        else:
            self.pm.create(work_prompt, status="work_investigating", mode="work", auto_mode=auto_mode, tdd_mode=tdd_mode, use_obsidian=use_obsidian)
            user_input = build_work_investigation_prompt(work_prompt, "")
            print_system(f"[Brain] Phase 1: Investigating codebase...")
            
        return True, user_input

    def step(self) -> tuple[bool, str | None]:
        """
        Advances the project/work state machine based on the current state.
        Returns (is_project_mode, auto_continue_input).
        """
        # Refresh PM status from state file
        self.pm = ProjectManager()
        
        if not self.pm.active:
            print_system("[Brain] No active project found.")
            return False, None
            
        status = self.pm.data.get("status", "")
        auto_continue_input = None
        is_project_mode = True
        
        # Phase 0 -> 1a: Research done, save data and generate architecture
        if status in ["researching", "work_researching"]:
            if not self.pm.data.get("research_data"):
                # Save the last AI response as the research synthesis
                self.pm.save_research_data(self.agent.messages[-1]["content"])
            
            if self.pm.data.get("mode") == "work":
                self.pm.set_status("work_investigating")
                self.agent.inject_context()
                auto_continue_input = build_work_investigation_prompt(self.pm.data["objective"], self.pm.data.get("research_data", ""))
                print_system(f"[Brain] Phase 1: Investigating codebase based on research...")
            else:
                self.pm.set_status("specifying_architecture")
                self.agent.inject_context()
                proj_prompt = self.pm.data["objective"]
                research_text = self.pm.data.get("research_data", "")
                auto_continue_input = build_architecture_prompt(proj_prompt, research_text)
                print_system(f"[Brain] Phase 1a: Designing architecture based on research...")

        # --- WORK MODE ---
        elif status == "work_investigating":
            self.agent.inject_context()
            auto_continue_input = "Investigation phase. Continue reading files, searching, or when ready, call `plan_work_changes`."
            print_system("[Brain] Phase 1: Investigation in progress...")

        elif status == "work_planning" and not self.pm.has_pending():
            self.agent.inject_context()
            print_system("[Brain] Phase 2: Generating micro-tasks from investigation plan...")
            auto_continue_input = build_work_planning_prompt(
                self.pm.data["objective"],
                self.pm.data.get("work_strategy", ""),
                self.pm.data.get("files_to_edit", []),
                self.pm.data.get("files_to_create", [])
            )
        
        elif status == "work_planning" and self.pm.has_pending():
            self.pm.set_status("work_executing")
            next_task = self.pm.get_next_pending()
            self.last_task_id = next_task["id"]
            self.task_retries = 0
            self.agent.inject_context()
            auto_continue_input = self.pm.build_execution_context(next_task)
            progress = self.pm.get_progress_display()
            print_system(f"[Brain] Phase 3: {progress} >> Work Task {next_task['id']}: {next_task['description']}")
        
        elif status == "work_executing" and self.pm.has_pending():
            next_task = self.pm.get_next_pending()
            if next_task["id"] == self.last_task_id:
                self.task_retries += 1
                if self.task_retries >= self.MAX_TASK_RETRIES:
                    print_system(f"[Brain] Task {next_task['id']} failed {self.MAX_TASK_RETRIES} times.")
                    action = questionary.select(
                        "What should we do with this stalled task?",
                        choices=["1. Continue trying (reset counter)", "2. Provide a hint to AI", "3. Skip task"]
                    ).ask()
                    
                    if action and action.startswith("1"):
                        self.task_retries = 0
                        print_system("Retrying task...")
                    elif action and action.startswith("2"):
                        hint = questionary.text("Enter your hint for the AI:").ask()
                        if hint:
                            next_task['description'] += f"\n\n[USER HINT AFTER FAILURE]: {hint}"
                            self.pm._save()
                        self.task_retries = 0
                        print_system("Hint added. Retrying task...")
                    else:
                        self.pm.complete_task(next_task["id"], "SKIPPED: failed to complete after multiple attempts")
                        self.last_task_id = None
                        self.task_retries = 0
                        return True, None
            else:
                self.last_task_id = next_task["id"]
                self.task_retries = 0
            
            self.agent.inject_context()
            auto_continue_input = self.pm.build_execution_context(next_task)
            progress = self.pm.get_progress_display()
            print_system(f"[Brain] {progress} >> Work Task {next_task['id']}: {next_task['description']}")
        
        elif status == "work_executing" and not self.pm.has_pending():
            is_project_mode = False
            self.pm.set_status("completed")
            print_system(f"[Brain] {self.pm.get_progress_display()} Work complete! Returning to chat.")

        # --- PROJECT MODE ---
        elif (status == "specifying" or status == "specifying_architecture") and self.pm.has_architecture():
            pending_files = self.pm.get_pending_spec_files()
            if pending_files:
                next_file = pending_files[0]
                self.pm.set_status("specifying_details")
                self.agent.inject_context()
                print_system(f"[Brain] Phase 1b: Detailing spec for {next_file} ({len(pending_files)} files remaining)...")
                auto_continue_input = build_spec_prompt(
                    self.pm.data['objective'], self.pm.get_architecture(), next_file
                )
            else:
                self.pm.set_status("planning")
                self.agent.inject_context()
                all_specs = self.pm.get_all_file_specs()
                specs_summary = "\n".join([f"  - {fname}" for fname in all_specs.keys()])
                print_system(f"[Brain] Phase 2: Creating tasks from {len(all_specs)} file specs...")
                auto_continue_input = build_planning_prompt(self.pm.data['objective'], specs_summary)

        elif status == "specifying_details":
            pending_files = self.pm.get_pending_spec_files()
            if pending_files:
                next_file = pending_files[0]
                self.agent.inject_context()
                print_system(f"[Brain] Phase 1b: Detailing spec for {next_file} ({len(pending_files)} files remaining)...")
                auto_continue_input = build_spec_prompt(
                    self.pm.data['objective'], self.pm.get_architecture(), next_file
                )
            else:
                self.pm.set_status("planning")
                self.agent.inject_context()
                all_specs = self.pm.get_all_file_specs()
                specs_summary = "\n".join([f"  - {fname}" for fname in all_specs.keys()])
                print_system(f"[Brain] Phase 2: Creating tasks from {len(all_specs)} file specs...")
                auto_continue_input = build_planning_prompt(self.pm.data['objective'], specs_summary)

        elif status == "planning" and self.pm.has_pending():
            self.pm.set_status("executing")
            next_task = self.pm.get_next_pending()
            self.last_task_id = next_task["id"]
            self.task_retries = 0
            self.agent.inject_context()
            auto_continue_input = self.pm.build_execution_context(next_task)
            progress = self.pm.get_progress_display()
            print_system(f"[Brain] Phase 3: {progress} >> Task {next_task['id']}: {next_task['description']}")

        elif status == "executing" and self.pm.has_pending():
            next_task = self.pm.get_next_pending()
            if next_task["id"] == self.last_task_id:
                self.task_retries += 1
                if self.task_retries >= self.MAX_TASK_RETRIES:
                    print_system(f"[Brain] Task {next_task['id']} failed {self.MAX_TASK_RETRIES} times.")
                    action = questionary.select(
                        "What should we do with this stalled task?",
                        choices=["1. Continue trying (reset counter)", "2. Provide a hint to AI", "3. Skip task"]
                    ).ask()
                    
                    if action and action.startswith("1"):
                        self.task_retries = 0
                        print_system("Retrying task...")
                    elif action and action.startswith("2"):
                        hint = questionary.text("Enter your hint for the AI:").ask()
                        if hint:
                            next_task['description'] += f"\n\n[USER HINT AFTER FAILURE]: {hint}"
                            self.pm._save()
                        self.task_retries = 0
                        print_system("Hint added. Retrying task...")
                    else:
                        self.pm.complete_task(next_task["id"], "SKIPPED: failed to complete after multiple attempts")
                        self.last_task_id = None
                        self.task_retries = 0
                        return True, None
            else:
                self.last_task_id = next_task["id"]
                self.task_retries = 0
            
            self.agent.inject_context()
            auto_continue_input = self.pm.build_execution_context(next_task)
            progress = self.pm.get_progress_display()
            print_system(f"[Brain] {progress} >> Task {next_task['id']}: {next_task['description']}")

        elif status == "executing" and not self.pm.has_pending():
            is_project_mode = False
            self.pm.set_status("completed")
            print_system(f"[Brain] {self.pm.get_progress_display()} Project complete! Returning to chat.")
            
        else:
            is_project_mode = False
            print_system("[Brain] Could not advance project. Check the model output.")
            
        return is_project_mode, auto_continue_input
