import os
import jedi
from pathlib import Path
from typing import List, Dict, Any

class CodeIntelligence:
    """Specialized module for semantic code analysis using Jedi."""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).expanduser().resolve()
        
    def find_definitions(self, file_path: str, line: int, column: int) -> List[Dict[str, Any]]:
        """Find definitions of a symbol at a specific location."""
        try:
            full_path = Path(file_path).resolve()
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            script = jedi.Script(source, path=full_path)
            defs = script.goto(line=line, column=column)
            
            results = []
            for d in defs:
                if d.module_path:
                    results.append({
                        "name": d.name,
                        "file_path": str(d.module_path),
                        "line": d.line,
                        "column": d.column,
                        "type": d.type,
                        "description": d.description
                    })
            return results
        except Exception as e:
            return [{"error": str(e)}]

    def find_references(self, file_path: str, line: int, column: int) -> List[Dict[str, Any]]:
        """Find references to a symbol at a specific location."""
        try:
            full_path = Path(file_path).resolve()
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            script = jedi.Script(source, path=full_path)
            refs = script.get_references(line=line, column=column)
            
            results = []
            for r in refs:
                if r.module_path:
                    results.append({
                        "name": r.name,
                        "file_path": str(r.module_path),
                        "line": r.line,
                        "column": r.column,
                        "description": r.description
                    })
            return results
        except Exception as e:
            return [{"error": str(e)}]

    def get_symbol_at(self, file_path: str, line: int, column: int) -> str:
        """Helper to get the symbol name at a location."""
        try:
            full_path = Path(file_path).resolve()
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if 0 < line <= len(lines):
                target_line = lines[line-1]
                # Simple word extraction around column
                import re
                words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', target_line)
                for w in words:
                    start = target_line.find(w)
                    if start <= column <= start + len(w):
                        return w
            return ""
        except:
            return ""

# Global instance
intel = CodeIntelligence()
