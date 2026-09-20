"""The FreeSwarm control plane.

FreeSwarm is the platform -- the console, the swarm, projects, the forecast lab, the
message board. FreeToken is the inference engine underneath it: it loads a model into
GPU memory and serves it, and keeps its own name.

The platform's settings were renamed to match, FREETOKEN_* -> FREESWARM_*. Every
FREETOKEN_* name a machine already exports still works: `adopt_legacy_env` copies it
onto the new name at import, so nothing reads two spellings and no existing setup
breaks. The engine's own variables below are not platform settings and are left alone.
"""

from __future__ import annotations

import os

# Settings that configure the ENGINE (loading a model into VRAM and running it), not the
# platform. These keep the FreeToken name, so they are never copied onto a FREESWARM_ one.
_ENGINE_VARS = frozenset({
    "FREETOKEN_VISIBLE_DEVICES",     # which GPUs an engine may use
    "FREETOKEN_ENGINE_PORT",         # the port an engine serves on
    "FREETOKEN_HOST_PIN_GB",         # host memory pinned for weight offload
    "FREETOKEN_HOST_PIN_FRACTION",
    "FREETOKEN_CUDA_HOME",           # toolchain used to build its kernels
    "FREETOKEN_VCVARS",
})


def adopt_legacy_env() -> None:
    """Let a FREETOKEN_* platform setting stand in for its FREESWARM_* name.

    `setdefault`, so an explicit FREESWARM_ value always wins over the old spelling.
    """
    for key, value in list(os.environ.items()):
        if key.startswith("FREETOKEN_") and key not in _ENGINE_VARS:
            os.environ.setdefault("FREESWARM_" + key[len("FREETOKEN_"):], value)


adopt_legacy_env()
