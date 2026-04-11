"""
Pipeline Execution for Argent.
Breaks down complex tasks into steps and executes them automatically.
"""

import json
from typing import List, Dict, Any

from logger import get_logger
from providers import create_provider

log = get_logger("pipeline")

PLAN_PROMPT = """You are a task decomposition engine. Break down the following task into 3-6 concrete, sequential steps.
Each step should be a specific action that can be executed using file operations, command execution, or code analysis.

TASK: {task}

Return ONLY a JSON array of steps:
[
  {{"step": 1, "action": "description of what to do", "tool_hint": "read_file/run_command/write_file/search_files/etc"}}
]

Rules:
- Steps must be ordered sequentially
- Each step should produce output useful for the next step
- Be specific about file paths, commands, and expected outcomes
- Keep actions concrete and executable"""


class Pipeline:
    """Plans and executes a multi-step task automatically."""

    def __init__(self, agent):
        self.agent = agent
        self.steps = []
        self.results = []
        self.current_step = 0

    def plan(self, description: str) -> List[Dict[str, Any]]:
        """Use LLM to decompose task into steps."""
        provider = create_provider()
        prompt = PLAN_PROMPT.format(task=description)
        
        response = provider.sync_chat(
            model=self.agent.model_name,
            messages=[{"role": "user", "content": prompt}],
            json_format=True,
        )
        
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            self.steps = json.loads(cleaned)
            if not isinstance(self.steps, list):
                self.steps = [self.steps]
            log.info("Pipeline planned %d steps for: %s", len(self.steps), description[:60])
        except json.JSONDecodeError:
            log.warning("Pipeline plan parsing failed, using single-step fallback")
            self.steps = [{"step": 1, "action": description, "tool_hint": "general"}]
        
        return self.steps

    def execute(self) -> str:
        """Execute all planned steps sequentially."""
        from ui import console
        
        if not self.steps:
            return "No steps to execute. Run plan() first."

        results = []
        for i, step in enumerate(self.steps):
            self.current_step = i + 1
            action = step.get("action", step.get("task", ""))
            tool_hint = step.get("tool_hint", "")
            
            console.print(f"\n  [bold cyan]Step {i+1}/{len(self.steps)}[/bold cyan]: {action}")
            console.print(f"  [dim]Tool hint: {tool_hint}[/dim]")
            
            step_prompt = (
                f"Execute this specific step and return ONLY the result.\n"
                f"Step: {action}\n"
                f"Recommended tool: {tool_hint}\n"
                f"Previous results context: {json.dumps(results[-2:], ensure_ascii=False) if results else 'None'}\n"
                f"Be concise. Execute the action and report the outcome."
            )
            
            full_result = ""
            try:
                for chunk in self.agent.process_user_input(step_prompt):
                    if chunk["type"] == "content_stream":
                        full_result += chunk["content"]
                    elif chunk["type"] == "tool_end":
                        pass
                    elif chunk["type"] == "error":
                        full_result += f"\n[ERROR]: {chunk['content']}"
            except Exception as e:
                full_result = f"Step failed: {e}"
                log.error("Pipeline step %d failed: %s", i + 1, e)
            
            result_entry = {"step": i + 1, "action": action, "result": full_result[:500]}
            results.append(result_entry)
            console.print(f"  [bold green]✓ Step {i+1} complete[/bold green]")
        
        self.results = results
        log.info("Pipeline executed: %d/%d steps", len(results), len(self.steps))
        
        return self._summarize()

    def _summarize(self) -> str:
        """Generate a final summary of all pipeline results."""
        summary_parts = []
        for r in self.results:
            step_num = r["step"]
            action = r["action"]
            result_preview = r["result"][:200]
            summary_parts.append(f"Step {step_num} ({action}): {result_preview}")
        
        return "\n".join(summary_parts)
