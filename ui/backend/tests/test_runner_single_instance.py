"""Only one swarm runner may run at a time (10-01 20:32: a restart left two, doubling every agent)."""

import subprocess
import sys
import textwrap
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


def test_a_second_runner_is_refused_while_the_first_holds_the_lock(tmp_path):
    lock = tmp_path / "runner.lock"
    holder = textwrap.dedent(f"""
        import importlib.util, sys, time
        spec = importlib.util.spec_from_file_location("r", r"{RUNNER}")
        r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
        print(r._single_instance(r"{lock}"), flush=True)
        time.sleep(30)
    """)
    first = subprocess.Popen([sys.executable, "-c", holder], stdout=subprocess.PIPE, text=True)
    try:
        assert first.stdout.readline().strip() == "True"
        second = subprocess.run([sys.executable, "-c", holder.replace("time.sleep(30)", "")],
                                capture_output=True, text=True, timeout=120)
        assert second.stdout.strip() == "False"
    finally:
        first.kill()
        first.wait()
    third = subprocess.run([sys.executable, "-c", holder.replace("time.sleep(30)", "")],
                           capture_output=True, text=True, timeout=120)
    assert third.stdout.strip() == "True"                       # a killed holder never blocks the next runner
