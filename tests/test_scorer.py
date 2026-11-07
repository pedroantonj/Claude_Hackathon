"""Official scorer parity + harness sanity (sample phase, fast)."""
import json
import os
import subprocess
import sys

from _util import DATA, ROOT, golden_as_submission_bytes, sample_golden, sample_submission

import bankrec_score as S


def _evaluate_score(path, *extra):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "evaluate.py"), path, "--phase", "sample",
                          "--json", path + ".json", *extra], check=True, capture_output=True, text=True).stdout
    return json.load(open(path + ".json")), out


def test_golden_as_submission_scores_100():
    res = S.score_submission(sample_golden(), golden_as_submission_bytes(), "golden.csv")
    assert res["ok"], res["errors"]
    assert res["score"] == 100.0
    assert res["metrics"]["txn_accuracy"] == 1.0 and res["metrics"]["exception_macro_f1"] == 1.0


def test_harness_matches_official_score_train():
    sub = sample_submission()
    res, _ = _evaluate_score(sub)
    off = subprocess.run([sys.executable, os.path.join(DATA, "starter", "score_train.py"), sub, "--data", DATA, "--sample"],
                         check=True, capture_output=True, text=True).stdout
    off = json.loads(off[:off.rindex("}") + 1])
    assert off["ok"] and res["ok"]
    assert res["score"] == off["score"]
    assert res["metrics"] == off["metrics"]


def test_harness_diagnostic_macro_equals_official():
    res, out = _evaluate_score(sample_submission())
    assert "WARNING: diagnostic macro" not in out
    k = [v for v in res["per_exception"].values() if v["counts_in_macro"]]
    macro = sum(v["f1"] for v in k) / len(k)
    assert abs(macro - res["metrics"]["exception_macro_f1"]) < 1e-3


def test_window_filter_drops_outside_rows():
    """Sample is June 2025: scoring it on a May window must drop every row and leave an empty universe."""
    sub = sample_submission()
    p = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "evaluate.py"), sub, "--phase", "sample",
                        "--from", "2025-06-01", "--to", "2025-06-15"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "submission rows outside window dropped" in p.stdout
    dropped = int(p.stdout.split(" submission rows outside")[0].rsplit(" ", 1)[1])
    assert dropped > 0


def test_pipeline_sample_floor():
    """Regression floor: sample score must not fall below the Phase-C baseline minus slack."""
    res, _ = _evaluate_score(sample_submission())
    assert res["ok"]
    assert res["score"] >= float(os.environ.get("BANKREC_SAMPLE_FLOOR", "75"))
