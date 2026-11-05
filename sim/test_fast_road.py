"""Bit-exactness of fast_road: the same scenarios are run in two fresh processes (original
highway-env and fast_road) and the full state of every vehicle is compared after every frame.
Run from the repo root:  python -m pytest -q sim/test_fast_road.py
Full scenario set (~10 min):  FAST_ROAD_FULL=1 python -m pytest -q sim/test_fast_road.py"""
import json
import os
import pathlib
import subprocess
import sys

RUNNER = pathlib.Path(__file__).resolve().parent / "equiv_runner.py"


def _run(mode, tmp_path):
    out = tmp_path / f"{mode}.json"
    quick = [] if os.environ.get("FAST_ROAD_FULL") else ["--quick"]
    subprocess.run([sys.executable, str(RUNNER), mode, str(out), *quick], check=True)
    return json.loads(out.read_text())


def test_fast_road_is_bit_exact(tmp_path):
    a, b = _run("orig", tmp_path), _run("fast", tmp_path)
    assert len(a["runs"]) == len(b["runs"]) > 0
    for ra, rb in zip(a["runs"], b["runs"]):
        tag = f'{ra["kind"]} {ra["p"]} ep{ra["ep"]}'
        assert ra["frames"] == rb["frames"] > 0, tag
        first = next((i for i, (x, y) in enumerate(zip(ra["frame_list"], rb["frame_list"])) if x != y), None)
        assert first is None, f"{tag}: state differs from frame {first}"
        assert ra["steps_digest"] == rb["steps_digest"], f"{tag}: observations/rewards/shield differ"
