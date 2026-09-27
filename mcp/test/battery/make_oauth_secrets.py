"""Create this task server's OAuth secrets and register it with the FreeSwarm control plane.

    python make_oauth_secrets.py              # first time
    python make_oauth_secrets.py --rotate     # new secrets (connected clients must reconnect)

Writes .oauth/server.json (the client, the approval passphrase's hash, the token signing key --
owner-only), the client secret for the control plane (ui/backend/auth/mcp_clients/battery-demo.secret,
owner-only) and the 'battery-demo' entry in ui/backend/mcp_servers.json (http + oauth). Then start the
server with --http and connect from the console: Connectors -> battery-demo -> Connect, approving with
the passphrase this prints (also saved to .oauth/approval_passphrase.txt -- delete it once stored).
Options: --host, --port, --control-plane, --passphrase, --no-register, --quiet (see --help).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))            # mcp/, for taskkit

from taskkit.oauth import generate  # noqa: E402

if __name__ == "__main__":
    generate(HERE, "battery-demo", 8201)
