import json
from logger import get_logger
from ui import console

log = get_logger("tools")

def run_swarm_workers(tasks_json: str) -> str:
    """
    Takes a JSON list of micro-tasks and executes them sequentially using freshly initialized, 
    isolated sub-agents. This allows complex tasks to be broken down so small local models 
    don't get overwhelmed by context.
    """
    try:
        tasks = json.loads(tasks_json)
        if not isinstance(tasks, list):
            return "Error: tasks_json must be a JSON array of strings."
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for tasks_json. Must be a JSON array of strings."
        
    from agent import ArgentSubAgent
    
    results = []
    console.print(f"\n[bold magenta][SWARM ACTIVATED]: Delegating {len(tasks)} micro-tasks to workers...[/bold magenta]")
    
    for i, task in enumerate(tasks):
        console.print(f"\n[bold yellow]Worker {i+1}/{len(tasks)} starting task:[/bold yellow] {task}")
        try:
            # We use 'Coder' role by default since it's the most common use-case, 
            # but the task description itself should dictate the action.
            agent = ArgentSubAgent(role="Worker", task=task)
            
            # Execute the subagent
            worker_result = agent.execute()
            
            summary = f"--- Task {i+1} Result ---\nTask: {task}\nResult: {worker_result}\n"
            results.append(summary)
            log.info(f"Worker {i+1} completed task.")
        except Exception as e:
            err_msg = f"--- Task {i+1} Failed ---\nTask: {task}\nError: {e}\n"
            results.append(err_msg)
            log.error(f"Worker {i+1} failed: {e}")
            
    console.print("\n[bold green][SWARM COMPLETED] ALL TASKS.[/bold green]")
    
    final_report = "SWARM EXECUTION REPORT\n======================\n\n" + "\n".join(results)
    return final_report
