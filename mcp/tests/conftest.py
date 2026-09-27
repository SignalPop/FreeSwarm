"""mcp/tests: the task servers and the optional taskkit helpers. Run from the repository root or
from mcp/:   python -m pytest mcp/tests -q   (the GEX tests skip when mcp/gex is absent)."""

from __future__ import annotations

import sys
from pathlib import Path

MCP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MCP))
