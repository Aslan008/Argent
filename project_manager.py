"""
Project Brain — internal memory and context manager for multi-step projects.
Allows small AI models to work autonomously by managing external persistent state.

Philosophy: Full context clearing = fresh head. All knowledge lives in detailed
per-file specifications on disk. The model reads a small piece of spec → executes
it in isolation → result is written to disk → context cleared → next piece.
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_FILE = Path(".argent_project.json")


class ProjectManager:
    """Manages project state, tasks, per-file specs, and context building."""

    def __init__(self):
        self.data = self._load()

    def _load(self) -> dict | None:
        if PROJECT_FILE.exists():
            try:
                return json.loads(PROJECT_FILE.read_text(encoding='utf-8'))
            except Exception:
                pass
        return None

    def _save(self):
        PROJECT_FILE.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

    @property
    def active(self) -> bool:
        return self.data is not None

    # ─── Project Lifecycle ────────────────────────────────────────────

    def create(self, objective: str, status: str = "specifying", mode: str = "project", auto_mode: bool = False, tdd_mode: bool = False):
        """Create a new project or work session from scratch."""
        self.data = {
            "mode": mode,
            "objective": objective,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "research_data": "",
            "architecture": "",
            "file_specs": {},
            "spec": "",
            "tasks": [],
            "files_created": [],
            "files_to_edit": [],
            "files_to_create": [],
            "work_strategy": "",
            "work_auto_mode": auto_mode,
            "tdd_mode": tdd_mode
        }
        self._save()

    def destroy(self):
        """Delete the project file and reset state."""
        if PROJECT_FILE.exists():
            PROJECT_FILE.unlink()
        self.data = None

    # ─── Research (Phase 0) ──────────────────────────────────────────

    def save_research_data(self, content: str):
        """Save the deep research synthesis report to context."""
        self.data["research_data"] = content
        self._save()

    # ─── Architecture Map (Phase 1a) ─────────────────────────────────

    def set_architecture(self, architecture: str, files: list = None):
        """Save the high-level architectural map and the explicit list of project files."""
        self.data["architecture"] = architecture
        if files:
            self.data["architecture_files"] = files
        self._save()

    def get_architecture(self) -> str:
        return self.data.get("architecture", "")

    def has_architecture(self) -> bool:
        return bool(self.data.get("architecture", "").strip())

    # ─── Per-File Specs (Phase 1b) ──────────────────────────────────

    def set_file_spec(self, filename: str, spec: str):
        """Save the detailed specification for a single file."""
        if "file_specs" not in self.data:
            self.data["file_specs"] = {}
        self.data["file_specs"][filename] = spec
        self._save()

    def get_file_spec(self, filename: str) -> str:
        return self.data.get("file_specs", {}).get(filename, "")

    def get_all_file_specs(self) -> dict:
        return self.data.get("file_specs", {})

    def get_pending_spec_files(self) -> list:
        """Return list of files from architecture that don't have a detailed spec yet."""
        # Use explicit file list if available (robust)
        file_list = self.data.get("architecture_files", [])
        if not file_list:
            return []

        specified = set(self.data.get("file_specs", {}).keys())
        return [f for f in file_list if f not in specified]

    def has_pending_specs(self) -> bool:
        return len(self.get_pending_spec_files()) > 0

    def all_specs_done(self) -> bool:
        """Check if all files from architecture have detailed specs."""
        pending = self.get_pending_spec_files()
        return len(pending) == 0 and len(self.get_all_file_specs()) > 0

    # ─── Legacy spec support ─────────────────────────────────────────

    def set_spec(self, spec: str):
        """Save the project specification (legacy, now also stores as architecture)."""
        self.data["spec"] = spec
        self._save()

    def get_spec(self) -> str:
        return self.data.get("spec", "")

    def has_spec(self) -> bool:
        return bool(self.data.get("spec", "").strip())

    # ─── Tasks ───────────────────────────────────────────────────────

    def add_task(self, description: str) -> int:
        """Add a task to the project. Returns the task ID."""
        tasks = self.data["tasks"]
        task_id = len(tasks) + 1
        tasks.append({
            "id": task_id,
            "description": description,
            "status": "pending",
            "result_summary": "",
            "files_affected": []
        })
        self._save()
        return task_id

    def complete_task(self, task_id: int, summary: str, files: list = None):
        """Mark a task as completed with a mandatory summary."""
        for t in self.data["tasks"]:
            if t["id"] == task_id:
                t["status"] = "completed"
                t["result_summary"] = summary
                if files:
                    t["files_affected"] = files
                    for f in files:
                        if f not in self.data["files_created"]:
                            self.data["files_created"].append(f)
                break
        self._save()

    def get_next_pending(self) -> dict | None:
        """Return the first pending task, or None."""
        for t in self.data["tasks"]:
            if t["status"] == "pending":
                return t
        return None

    def has_pending(self) -> bool:
        return self.get_next_pending() is not None

    def is_complete(self) -> bool:
        return len(self.data["tasks"]) > 0 and not self.has_pending()

    def set_status(self, status: str):
        self.data["status"] = status
        self._save()

    # ─── Context Building ────────────────────────────────────────────

    def _extract_interfaces_from_spec(self, spec: str) -> str:
        """Extract just the interface part (classes, functions, signatures)
        from a file spec, stripping implementation details."""
        # We keep lines that look like interface definitions:
        # - lines with class/function/method definitions
        # - lines with field/variable declarations
        # - import lines
        # For simplicity, return a condensed version of the spec
        lines = spec.strip().split('\n')
        interface_lines = []
        for line in lines:
            stripped = line.strip().lower()
            # Keep structural/interface lines, skip detailed logic descriptions
            if any(keyword in stripped for keyword in [
                'import', 'from', 'class', 'method', 'function', 'def ',
                'field', 'param', 'return', 'type:', ':', 'путь', 'path',
                'использует', 'uses', 'зависим', 'depend', 'экспорт', 'export',
                'интерфейс', 'interface', '(', '→', '->'
            ]):
                interface_lines.append(line)
            elif line.strip().startswith('-') or line.strip().startswith('*'):
                interface_lines.append(line)

        if not interface_lines:
            # Fallback: return first 15 lines as a summary
            return '\n'.join(lines[:15])

        return '\n'.join(interface_lines)

    def _get_task_filename(self, task: dict) -> str | None:
        """Try to extract a filename from the task description."""
        import re
        desc = task.get("description", "")
        # Look for filenames in the description
        match = re.search(r'[\w/\\]+\.\w+', desc)
        return match.group(0).replace('\\', '/') if match else None

    def _get_dependency_files(self, filename: str) -> list:
        """Find which files the given file depends on, based on the architecture."""
        architecture = self.get_architecture()
        if not architecture:
            return []

        # Simple heuristic: look for dependency mentions near the filename
        import re
        deps = []
        all_files = list(self.get_all_file_specs().keys())

        # Check the file's own spec for imports/dependencies
        spec = self.get_file_spec(filename)
        for other_file in all_files:
            if other_file == filename:
                continue
            # Check if this file is mentioned in the spec or architecture near our file
            base_name = Path(other_file).stem  # e.g., "converter" from "converter.py"
            if base_name in spec or other_file in spec:
                deps.append(other_file)

        return deps

    def build_execution_context(self, task: dict) -> str:
        """Build an isolated, self-sufficient context for executing a specific task.

        The model gets:
        1. Spec of THIS file only (full detail)
        2. Interfaces of dependency files (just signatures, not logic)
        3. Clear instructions
        """
        objective = self.data["objective"]
        filename = self._get_task_filename(task)

        ctx = f"=== ПРОЕКТ: {objective} ===\n\n"
        
        if self.data.get("tdd_mode"):
            ctx += "=== TDD MODE ACTIVE ===\n"
            ctx += (
                "You MUST follow the Test-Driven Development workflow:\n"
                "1. FIRST, write a failing unit test for the intended functionality in a separate test file (e.g., tests/test_filename.py).\n"
                "2. SECOND, run the test using `run_command` and verify it fails.\n"
                "3. THIRD, implement the minimum code required in the target file to make the test pass.\n"
                "4. FOURTH, run the tests again and ensure they are GREEN (Pass).\n"
                "Only after tests pass should you call `complete_project_task`.\n\n"
            )

        # If we have per-file specs and can identify the file
        if filename and self.get_file_spec(filename):
            file_spec = self.get_file_spec(filename)
            ctx += f"=== СПЕЦИФИКАЦИЯ ФАЙЛА: {filename} ===\n"
            ctx += file_spec + "\n\n"

            # Add interfaces of dependency files
            deps = self._get_dependency_files(filename)
            if deps:
                ctx += "=== ИНТЕРФЕЙСЫ ЗАВИСИМЫХ ФАЙЛОВ (НЕ РЕАЛИЗУЙ ИХ, ТОЛЬКО ИСПОЛЬЗУЙ) ===\n"
                for dep in deps:
                    dep_spec = self.get_file_spec(dep)
                    if dep_spec:
                        interfaces = self._extract_interfaces_from_spec(dep_spec)
                        ctx += f"\n--- {dep} ---\n"
                        ctx += interfaces + "\n"
                ctx += "\n"

        # Fallback: use legacy spec if no per-file specs available
        elif self.data.get("spec"):
            ctx += "=== СПЕЦИФИКАЦИЯ ПРОЕКТА ===\n"
            ctx += self.data["spec"] + "\n\n"

        # Short summaries of completed tasks
        completed_parts = []
        for t in self.data["tasks"]:
            if t["status"] == "completed" and t["result_summary"]:
                completed_parts.append(f"  [DONE] Task {t['id']}: {t['result_summary']}")

        if completed_parts:
            ctx += "Уже завершённые задачи:\n"
            ctx += "\n".join(completed_parts) + "\n\n"

        ctx += f"=== ТВОЯ ТЕКУЩАЯ ЗАДАЧА ===\n"
        ctx += f"Task ID {task['id']}: {task['description']}\n\n"
        ctx += (
            f"ПРАВИЛА (ОБЯЗАТЕЛЬНО СОБЛЮДАЙ):\n"
            f"1. Сначала используй `read_file`, чтобы прочитать ТЕКУЩЕЕ содержимое файла (скелет/другие методы).\n"
            f"2. ДОБАВЬ или ИЗМЕНИ нужный код для ТОЛЬКО ЭТОЙ ЗАДАЧИ.\n"
            f"3. Перезапиши ВЕСЬ файл через `write_file`. НЕ УДАЛЯЙ УЖЕ СУЩЕСТВУЮЩИЕ методы!\n"
            f"4. ПОСЛЕ записи вызови `complete_project_task(task_id={task['id']}, summary='...')`.\n"
            f"5. В параметре summary укажи, какой именно метод(ы) ты реализовал.\n"
            f"6. НЕ пиши complete_project_task как текст! Вызови его КАК ИНСТРУМЕНТ через JSON!\n"
        )

        return ctx



    # ─── Display ─────────────────────────────────────────────────────

    def get_progress_display(self) -> str:
        """Return a short progress string."""
        tasks = self.data.get("tasks", [])
        total = len(tasks)
        done = sum(1 for t in tasks if t["status"] == "completed")
        if total == 0:
            return "[No tasks]"
        bar_len = 20
        filled = int(bar_len * done / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"[{bar}] {done}/{total}"
