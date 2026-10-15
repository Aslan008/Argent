# Unity MCP Discovery

## Root cause (verified 2026-09-26)
The `unity` CLI (1.0.0-beta.3, package com.unity.pipeline 0.8.0-exp.1) auto-discovers
the Editor **from the process CWD only** — it reads `<CWD>/Library/Pipeline/.unity-pipeline-port`
literally. No walk-up the directory tree, no Hub-registry scan.

| CWD | `unity mcp` tools/list |
|---|---|
| Unity project root | 150+ tools |
| Any subfolder (e.g. `Assets/`) or Argent's launch dir | `[]` (empty) |
| Anywhere + `--project-path <root>` | 150+ tools |

## Fix applied (Argent patch)
`C:\Users\user\Desktop\Argent\mcp_client.py`:
- `StdioTransport.__init__(..., cwd=None)` → passed to `subprocess.Popen(cwd=...)`.
- `MCPServer.start()` passes `cwd=self.config.get("cwd") or project_root_or_cwd()`.
- So every stdio MCP server is anchored to the project root (found via
  `.argent` / `.argent_project.json` / `.git` markers). A per-server `"cwd"`
  config entry overrides this.

## Diagnosing "unity MCP has no tools"
1. `list_mcp_tools()` → `unity (stdio): unreachable` or "no tools".
2. Check the descriptor: `Library/Pipeline/.unity-pipeline-port` (has `port`,
   `evalToken`, `pid`, heartbeat). Fresh heartbeat = editor server alive.
3. Test discovery from a shell: `unity pipeline list` — empty table means CWD
   is not a project root; `Set-Location <project root>` first, then retry.
4. Manual handshake test (background): run `unity mcp`, send
   `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}`,
   then `{"jsonrpc":"2.0","method":"notifications/initialized"}`, then
   `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`. Empty `tools:[]` = discovery
   failed; full list = OK.
5. Editor server port range: 7800–7849. Bearer token = `evalToken` from the
   descriptor; `GET /api/status` with `Authorization: Bearer <token>` must
   return 200 (also makes the server rewrite the descriptor clean, no BOM).

## Gotchas
- `unity command list` exits 1 ("Command 'list' failed") even when it works —
  the server prints its 150+ commands to stdout; there is no command named `list`.
- The editor launched without `-automated` sets an `info` field in the
  descriptor (modal-dialog warning) — informational, not a discovery filter.
- `unity pipeline list` from a subfolder shows an empty table — same CWD rule.