import json
import requests
from typing import Dict, Any, List

class MCPClient:
    """
    Standardized client for interacting with Model Context Protocol (MCP) servers.
    This allows Argent to leverage external standardized tool ecosystems.
    """
    
    def __init__(self):
        self.servers: Dict[str, str] = {} # server_name -> url
        
    def register_server(self, name: str, url: str):
        """Register an MCP server endpoint."""
        self.servers[name] = url
        
    def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List tools available on a specific MCP server."""
        if server_name not in self.servers:
            return [{"error": f"Server '{server_name}' not registered."}]
            
        try:
            # MCP conventional endpoint for listing tools
            response = requests.get(f"{self.servers[server_name]}/tools", timeout=5)
            response.raise_for_status()
            return response.json().get("tools", [])
        except Exception as e:
            return [{"error": str(e)}]

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific tool on an MCP server."""
        if server_name not in self.servers:
            return f"Error: Server '{server_name}' not registered."
            
        try:
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            response = requests.post(f"{self.servers[server_name]}/call", json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if result.get("isError"):
                return f"MCP Tool Error: {result.get('content')}"
            return str(result.get("content"))
        except Exception as e:
            return f"Error calling MCP tool '{tool_name}' on '{server_name}': {e}"

# Global instance
mcp_client = MCPClient()
