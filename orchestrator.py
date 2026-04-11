import json
from typing import List, Dict, Any
from agent import ArgentSubAgent

class Orchestrator:
    """
    Handles task decomposition and sub-agent management.
    Ensures that complex user requests are broken down into manageable pieces.
    """
    
    def __init__(self, supervisor_agent):
        self.supervisor = supervisor_agent

    def decompose_task(self, big_task: str) -> List[Dict[str, Any]]:
        """Ask the model to break down a complex request into sub-agent tasks."""
        prompt = f"""
        Break down the following complex technical task into 2-4 discrete sub-tasks for specialized sub-agents.
        Roles available: Coder, Researcher, Reviewer, DocWriter.
        
        TASK: {big_task}
        
        Return a JSON LIST of tasks:
        [
          {{"role": "RoleName", "task": "detailed instructions", "tools": ["tool1", "tool2"]}}
        ]
        """
        # Simple implementation for now - this would ideally be done by an LLM call
        # For demonstration, we'll return a hardcoded structure or simple split
        return [
            {"role": "Researcher", "task": f"Research implementation details for: {big_task}", "tools": ["search_web", "read_webpage"]},
            {"role": "Coder", "task": f"Implement the core logic for: {big_task}", "tools": ["write_file", "replace_in_file", "find_definition"]}
        ]

    def run_hive_mind(self, big_task: str) -> str:
        """Execute a complex task using multiple specialized agents."""
        tasks = self.decompose_task(big_task)
        reports = []
        
        for t in tasks:
            sub = ArgentSubAgent(role=t["role"], task=t["task"], tools_override=t.get("tools"))
            report = sub.execute()
            reports.append(f"### Report from {t['role']}:\n{report}\n")
            
        # Synthesize final result
        final_synthesis = "\n".join(reports)
        return f"Hive Mind execution complete.\n\n{final_synthesis}"

# Static helper
def spawn_subagent(role: str, task: str, tools: List[str] = None) -> str:
    """Convenience tool to run a one-off subagent."""
    agent = ArgentSubAgent(role, task, tools)
    return agent.execute()
