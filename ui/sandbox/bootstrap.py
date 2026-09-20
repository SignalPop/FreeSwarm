"""Runs the user's script with the headless-environment sharp edges filed off.

The sandbox has no display, so `plt.show()` -- which is what a model writes nine times out
of ten, and what every tutorial shows -- silently does nothing: no window, no file, no
output. The user sees "No output and no files produced" and has no idea their chart was
discarded rather than never drawn.

So `show()` is redirected to a file, and any figure still open when the script ends is
written out too (a script that only calls `plt.plot(...)` and stops is just as common).

Care is taken NOT to duplicate work the script did itself: if it called `savefig`, the
exit-time sweep stays out of the way, because the figures it saved are the ones it wanted.

This runs as `python -I bootstrap.py script.py`, so the user's script still sees
`__name__ == "__main__"` and its own filename in tracebacks.
"""

from __future__ import annotations

import os
import sys
import traceback

_PREFIX = "plot"
_DPI = 110


def _install_pyplot_capture() -> None:
    """Redirect plt.show() to a PNG, and flush any figure left open at exit."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except Exception:  # noqa: BLE001 -- matplotlib absent is fine; nothing to capture
        return

    state = {"n": 0, "user_saved": False}

    def _next_name() -> str:
        # Never clobber a file the script wrote itself.
        while True:
            state["n"] += 1
            name = f"{_PREFIX}.png" if state["n"] == 1 else f"{_PREFIX}-{state['n']}.png"
            if not os.path.exists(name):
                return name

    def _save_open_figures() -> list[str]:
        saved: list[str] = []
        for num in plt.get_fignums():
            fig = plt.figure(num)
            # An axes-less figure is a blank canvas nobody asked to see.
            if not fig.get_axes():
                continue
            name = _next_name()
            try:
                fig.savefig(name, dpi=_DPI, bbox_inches="tight")
            except Exception as exc:  # noqa: BLE001 -- a failed save must not kill the script
                print(f"[could not save figure: {exc}]", file=sys.stderr)
                continue
            saved.append(name)
        return saved

    real_show = plt.show

    def show(*_args, **_kwargs):
        """plt.show() -> write the figures out and say where they went."""
        names = _save_animations() + _save_open_figures()
        for name in names:
            print(f"[saved {name}]")
        if names:
            plt.close("all")
        return None

    show.__doc__ = (real_show.__doc__ or "") + "\n\n(FreeToken sandbox: saves to a file.)"
    plt.show = show

    # Track explicit saves so the exit sweep does not double-write the same chart.
    real_fig_savefig = Figure.savefig

    def savefig(self, *args, **kwargs):
        state["user_saved"] = True
        return real_fig_savefig(self, *args, **kwargs)

    Figure.savefig = savefig

    # Animations. `FuncAnimation(...); plt.show()` is how a model writes "make an animated
    # plot" -- and without a display, show() would save one still frame of it as a PNG,
    # silently throwing the animation away. Track every Animation as it is constructed so
    # show() and the exit sweep can write it out as a GIF instead.
    animations: list = []
    try:
        from matplotlib import animation as _anim

        real_anim_init = _anim.Animation.__init__

        def anim_init(self, *args, **kwargs):
            real_anim_init(self, *args, **kwargs)
            animations.append(self)

        _anim.Animation.__init__ = anim_init

        real_anim_save = _anim.Animation.save

        def anim_save(self, *args, **kwargs):
            # An explicit save means the script chose its own output; do not duplicate it.
            state["user_saved"] = True
            if self in animations:
                animations.remove(self)
            return real_anim_save(self, *args, **kwargs)

        _anim.Animation.save = anim_save
    except Exception:  # noqa: BLE001 -- no animation support; stills still work
        _anim = None

    def _save_animations() -> list[str]:
        saved: list[str] = []
        while animations:
            anim = animations.pop(0)
            name = "animation.gif" if not saved and not os.path.exists("animation.gif") else None
            if name is None:
                n = len(saved) + 2
                while os.path.exists(f"animation-{n}.gif"):
                    n += 1
                name = f"animation-{n}.gif"
            try:
                # The ORIGINAL save, not the patched one: the patched save marks the run as
                # "the script saved its own output", which would then suppress writing out
                # any other still figures at exit.
                real_anim_save(anim, name, writer=_anim.PillowWriter(fps=20))
            except Exception as exc:  # noqa: BLE001 -- fall back to the still frame
                print(f"[could not save animation: {exc}]", file=sys.stderr)
                continue
            saved.append(name)
            # Its figure is now represented by the GIF; do not also write a still of it.
            try:
                plt.close(anim._fig)
            except Exception:  # noqa: BLE001
                pass
        return saved

    import atexit

    @atexit.register
    def _flush_remaining() -> None:
        # A script that drew but never called show() or savefig() still meant to produce
        # something. If it saved its own files, leave whatever is open alone.
        # Animations first: they are never "saved" by a plain savefig, so they would be lost.
        for name in _save_animations():
            print(f"[saved {name}]")
        if state["user_saved"]:
            return
        for name in _save_open_figures():
            print(f"[saved {name}]")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bootstrap.py <script.py>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    sys.argv = sys.argv[1:]  # the script sees its own argv, not the bootstrap's

    _install_pyplot_capture()

    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return 2

    code = compile(source, path, "exec")
    globals_ns = {"__name__": "__main__", "__file__": path, "__builtins__": __builtins__}
    try:
        exec(code, globals_ns)  # noqa: S102 -- executing the user's script is the job
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:  # noqa: BLE001 -- report the script's traceback, not ours
        exc_type, exc_value, tb = sys.exc_info()
        # Drop this file's frame so the traceback starts at the user's own code.
        traceback.print_exception(exc_type, exc_value, tb.tb_next if tb else None)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
