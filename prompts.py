def build_spec_prompt(objective: str, architecture: str, filename: str) -> str:
    return (
        f"You are writing a DETAILED specification for ONE file in the project: '{objective}'.\n\n"
        f"=== PROJECT ARCHITECTURE ===\n"
        f"{architecture}\n\n"
        f"Your task: write a DETAILED specification for the file: {filename}\n\n"
        f"You MUST call `write_file_spec(filename='{filename}', spec=...)` with a TEXT DESCRIPTION (NOT code!).\n\n"
        f"CRITICAL: Write a TEXT DESCRIPTION, NOT Python/C#/JS code! The spec must describe WHAT to implement, "
        f"not be the implementation itself.\n\n"
        f"Your spec MUST include:\n"
        f"1. File path\n"
        f"2. ALL imports/dependencies (exact module names)\n"
        f"3. ALL classes: name, inheritance\n"
        f"4. ALL methods/functions: name, ALL parameters with types, return type, and WHAT THE METHOD DOES (1-2 sentences of logic)\n"
        f"5. ALL fields/variables: name, type, default value\n"
        f"6. Which other project files this file imports and what names it uses from them\n\n"
        f"EXAMPLE OF A GOOD SPEC (this is what you should write):\n"
        f"  File: converter.py\n"
        f"  Imports: json (standard library)\n"
        f"  Dependencies: uses get_rate() from api_client.py\n"
        f"  Class: CurrencyConverter\n"
        f"    Fields:\n"
        f"      - rates_cache: dict, default empty dict\n"
        f"    Methods:\n"
        f"      - __init__(self): initializes empty rates_cache\n"
        f"      - convert(self, amount: float, from_cur: str, to_cur: str) -> float: calls get_rate(), multiplies amount by rate, returns result\n"
        f"      - supported_currencies(self) -> list[str]: returns hardcoded list of supported currency codes\n\n"
        f"EXAMPLE OF A BAD SPEC (DO NOT write code like this):\n"
        f"  def convert(self, amount, from_cur, to_cur):\n"
        f"      rate = get_rate(from_cur, to_cur)\n"
        f"      return amount * rate\n\n"
        f"BE EXTREMELY SPECIFIC with names and types. The AI that implements this will have NO OTHER CONTEXT.\n\n"
        f"Call `write_file_spec` NOW!"
    )

def build_work_investigation_prompt(objective: str, research_context: str = "") -> str:
    res = (
        f"You are the Lead Investigator for an existing codebase. Your objective is: '{objective}'.\n\n"
    )
    if research_context:
        res += f"=== LATEST RESEARCH CONTEXT ===\n{research_context}\n===============================\n\n"
    res += (
        f"Your task is to INVESTIGATE the current codebase and devise a plan to achieve the objective.\n"
        f"RULES:\n"
        f"1. You MUST use tools like `list_directory`, `grep_search`, and CRITICALLY `read_file` to examine the ACTUAL CODE.\n"
        f"2. You MUST NOT guess or hallucinate class names, method names, or file paths. READ THE FILES FIRST.\n"
        f"3. When you have a complete plan, call `plan_work_changes` with:\n"
        f"   - strategy: Detailed explanation of how you will solve the objective.\n"
        f"   - files_to_edit: Comma-separated list of EXISTING files to modify.\n"
        f"   - files_to_create: Comma-separated list of NEW files to create (leave empty if none).\n"
        f"4. ONLY call `plan_work_changes` when you are absolutely sure about the exact files to touch.\n\n"
        f"Start by listing the directory or searching for relevant files now!"
    )
    return res

def build_work_planning_prompt(objective: str, strategy: str, files_to_edit: list, files_to_create: list) -> str:
    return (
        f"The objective is: '{objective}'.\n"
        f"The agreed strategy is:\n{strategy}\n\n"
        f"Files to modify: {', '.join(files_to_edit) if files_to_edit else 'None'}\n"
        f"Files to create: {', '.join(files_to_create) if files_to_create else 'None'}\n\n"
        f"Now create implementation MICRO-TASKS using `add_work_task`. RULES:\n"
        f"1. Break the work down into a sequence of extremely small tasks.\n"
        f"2. For EXISTING files, create a task for EACH specific method or logic block you need to change.\n"
        f"3. For NEW files, Task 1 must be creating the skeleton (imports, empty classes/methods), followed by tasks for each method.\n"
        f"4. ONLY call `add_work_task`. Do NOT write code or complete tasks yet!\n"
        f"5. IMPORTANT: When you have added ALL tasks, you MUST stop calling tools and reply exactly with 'DONE'.\n\n"
        f"Call `add_work_task` for EVERY step NOW, then stop."
    )

def build_planning_prompt(objective: str, specs_summary: str) -> str:
    return (
        f"The project '{objective}' has detailed specifications for these files:\n{specs_summary}\n\n"
        f"Now create implementation MICRO-TASKS. RULES:\n"
        f"1. Break EACH FILE down into a sequence of small tasks.\n"
        f"2. Task 1 for a file MUST be creating the 'skeleton' (imports, empty classes, methods with 'pass').\n"
        f"3. Task 2, Task 3, etc. for that file MUST implement EXACTLY ONE method each.\n"
        f"4. CRITICAL: ENTRY POINT FILES (like main.py) MUST have implementation tasks! Do NOT just create a skeleton for main.py. You MUST create a task to implement the main logic/function.\n"
        f"5. Create tasks in DEPENDENCY ORDER: standalone files first, then files that depend on them.\n"
        f"6. ONLY call `add_project_task`. Do NOT call `complete_project_task` or `write_file`!\n\n"
        f"EXAMPLE of Micro-Tasks for 'database.py':\n"
        f"  add_project_task('Create skeleton for database.py (imports and empty DB class)')\n"
        f"  add_project_task('Implement DB.__init__ method in database.py')\n"
        f"  add_project_task('Implement DB.save method in database.py')\n\n"
        f"Call `add_project_task` for EVERY file's micro-tasks NOW. When finished adding all tasks, stop calling tools and reply exactly with 'DONE'."
    )

def build_architecture_prompt(proj_prompt: str, research_text: str = "") -> str:
    prompt = f"You are the architect for this project: '{proj_prompt}'.\n\n"
    if research_text:
        prompt += f"=== LATEST RESEARCH CONTEXT ===\n{research_text}\n\n"
    prompt += (
        f"Your task is to design the HIGH-LEVEL ARCHITECTURE.\n"
        f"You MUST call `write_project_architecture` with TWO parameters:\n"
        f"  1. architecture = text description of all files and their dependencies\n"
        f"  2. files = comma-separated list of ALL file paths to create\n\n"
        f"EXAMPLE call:\n"
        f"  write_project_architecture(\n"
        f"    architecture='Files:\\n1. MyApp/main.py — Entry point. Uses: calculator.py\\n2. MyApp/calculator.py — Math logic. Standalone.',\n"
        f"    files='MyApp/main.py, MyApp/calculator.py'\n"
        f"  )\n\n"
        f"RULES:\n"
        f"- ALL file paths MUST include the project folder (e.g. MyProject/main.py, NOT just main.py)\n"
        f"- The 'files' parameter must list ONLY the files to CREATE, not referenced libraries\n"
        f"- Do NOT describe implementation details (no method names, no types)\n"
        f"- Do NOT create any files yet\n"
        f"- Do NOT call any other tools\n"
        f"- Call `write_project_architecture` NOW!"
    )
    return prompt
