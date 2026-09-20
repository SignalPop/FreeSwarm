"""Example MCP connector -- a template for writing your own.

Run standalone it speaks MCP over stdio, which is what `mcp_servers.json` launches:

    {"name": "freetoken", "transport": "stdio",
     "command": "python",
     "args": ["<abs path>/ui/backend/connectors/freetoken_tools.py"],
     "enabled": true}

Every `@mcp.tool()` function becomes a tool the model can call. The docstring becomes the
tool description the model reads to decide *whether* to call it, and the type hints become
the JSON schema it must satisfy -- so both are load-bearing, not documentation.

The tools here are deliberately read-only. Anything that writes, deletes, or shells out is
an action a model can be talked into taking, so give those their own connector and think
about the blast radius before enabling it.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

# mcp 2.x renamed FastMCP -> MCPServer. Pin `mcp<2` if you are porting a v1 connector.
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("freetoken-tools")

# Filesystem tools are confined to this root. Set FREESWARM_MCP_ROOT to move it. Without a
# confinement check, `read_file("C:/Users/you/.ssh/id_rsa")` is a valid tool call.
SANDBOX = Path(os.getenv("FREESWARM_MCP_ROOT", Path(__file__).resolve().parents[3])).resolve()

MAX_READ_BYTES = 200_000


def _resolve_inside(relative: str) -> Path:
    """Resolve `relative` under SANDBOX, refusing anything that escapes it.

    resolve() first, then check containment: that collapses `..` and follows symlinks, so a
    path that *looks* contained but points outside is still caught.
    """
    candidate = (SANDBOX / relative).resolve()
    if candidate != SANDBOX and not candidate.is_relative_to(SANDBOX):
        raise ValueError(f"path escapes the sandbox ({SANDBOX}): {relative}")
    return candidate


@mcp.tool()
def gpu_status() -> str:
    """Report each NVIDIA GPU's name, memory use, utilisation and temperature.

    Use this to decide whether there is room to load another model.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return "nvidia-smi is not on PATH"
    out = subprocess.run(
        [exe, "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    return out.stdout.strip() or "no GPUs reported"


@mcp.tool()
def host_info() -> str:
    """Report the host OS, CPU count and total RAM. Useful for sizing decisions."""
    try:
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MemStatus()
        st.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        ram = f"{st.ullTotalPhys / 2**30:.1f} GiB total, {st.ullAvailPhys / 2**30:.1f} GiB free"
    except Exception:  # noqa: BLE001 - non-Windows or a blocked call
        ram = "unknown"
    return (
        f"platform={platform.platform()}\n"
        f"python={platform.python_version()}\n"
        f"cpus={os.cpu_count()}\n"
        f"ram={ram}\n"
        f"sandbox={SANDBOX}"
    )


@mcp.tool()
def list_files(subdir: str = ".") -> str:
    """List files and directories under `subdir`, relative to the sandbox root.

    Returns one entry per line, directories suffixed with a slash.
    """
    target = _resolve_inside(subdir)
    if not target.is_dir():
        return f"not a directory: {subdir}"
    rows = []
    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if child.is_dir():
            rows.append(f"{child.name}/")
        else:
            try:
                rows.append(f"{child.name}  ({child.stat().st_size} bytes)")
            except OSError:
                rows.append(child.name)
    return "\n".join(rows) or "(empty)"


@mcp.tool()
def read_file(path: str, max_bytes: int = MAX_READ_BYTES) -> str:
    """Read a UTF-8 text file relative to the sandbox root.

    Truncates at `max_bytes` so a model cannot blow its own context window on a large file.
    """
    target = _resolve_inside(path)
    if not target.is_file():
        return f"not a file: {path}"
    cap = max(1, min(int(max_bytes), MAX_READ_BYTES))
    data = target.read_bytes()[:cap]
    text = data.decode("utf-8", errors="replace")
    if target.stat().st_size > cap:
        text += f"\n\n[truncated at {cap} bytes of {target.stat().st_size}]"
    return text


@mcp.tool()
def search_files(pattern: str, subdir: str = ".", limit: int = 100) -> str:
    """Find files whose name matches a glob `pattern` (e.g. '*.py') under `subdir`."""
    root = _resolve_inside(subdir)
    if not root.is_dir():
        return f"not a directory: {subdir}"
    hits: list[str] = []
    for p in root.rglob(pattern):
        # rglob follows into directories that may be symlinked outside; re-check each hit.
        try:
            resolved = p.resolve()
            if not resolved.is_relative_to(SANDBOX):
                continue
            hits.append(str(resolved.relative_to(SANDBOX)))
        except (OSError, ValueError):
            continue
        if len(hits) >= max(1, min(limit, 1000)):
            break
    return "\n".join(hits) or "(no matches)"


if __name__ == "__main__":
    mcp.run()
