import csv

import pytest

from vera import crim_eval as ce


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ce.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ce.FIELDS})


def _row(i, split, mech, label, text):
    return {"id": f"x{i}", "split": split, "mechanism": mech, "technique": "t", "label": label, "text": text}


def test_repo_item_file_is_valid():
    assert ce.validate(ce.load()) == []


def test_validation_catches_bad_rows(tmp_path):
    p = tmp_path / "items.csv"
    _write(p, [_row(0, "dev", "control", "block", "a"), _row(0, "train", "other", "maybe", "")])
    errs = ce.validate(ce.load(p))
    assert any("control items" in e for e in errs)
    assert any("duplicate id" in e for e in errs)
    assert any("split" in e for e in errs) and any("mechanism" in e for e in errs)


def test_freeze_detects_test_edits(tmp_path):
    p, m = tmp_path / "items.csv", tmp_path / "manifest.json"
    rows = [_row(0, "test", "control", "allow", "Plan a trip to Kyoto."),
            _row(1, "dev", "control", "allow", "Suggest a pasta recipe.")]
    _write(p, rows)
    ce.freeze(ce.load(p), m)
    ce.check_frozen(ce.load(p), m)
    rows[1]["text"] = "Dev items may change."
    _write(p, rows)
    ce.check_frozen(ce.load(p), m)  # dev edits are allowed
    rows[0]["text"] = "Edited test item."
    _write(p, rows)
    with pytest.raises(ce.FrozenSetChanged):
        ce.check_frozen(ce.load(p), m)


def test_score_breakdown():
    rows = [_row(0, "dev", "neutralization", "block", "a"), _row(1, "dev", "neutralization", "block", "b"),
            _row(2, "dev", "control", "allow", "c")]
    s = ce.score(rows, {"x0": True, "x1": False, "x2": True})
    assert s["by_mechanism"]["neutralization"] == 0.5
    assert s["missed"] == ["x1"] and s["over_blocked"] == ["x2"]
