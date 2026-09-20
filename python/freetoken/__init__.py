"""FreeToken inference runtime."""

from freetoken import _win_compat as _win_compat  # noqa: F401  applies Windows runtime shims

from freetoken.version import __version__

__all__ = ["__version__"]
