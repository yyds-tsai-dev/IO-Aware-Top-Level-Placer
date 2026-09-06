import json

import numpy as np
import pytest

from scripts.stage2_verify_identity import compare


def _snapshot(path, *, reverse=False, mutate=None):
    data = dict(node_names=np.array(["u0", "port"]),
                node_kind=np.array(["inst", "bterm"]),
                node_orientation=np.array(["R0", "NONE"]),
                node_x=np.array([1., 5.]), node_y=np.array([2., 7.]),
                node_w=np.array([2., 1.]), node_h=np.array([3., 1.]),
                net_names=np.array(["n0", "n1"]),
                pin_names=np.array(['["ITerm","u0","A"]', '["BTerm","port"]']),
                pin_net=np.array([0, 1]))
    if reverse:
        for key in list(data):
            if key.startswith("node_"):
                data[key] = data[key][::-1].copy()
        data["net_names"] = data["net_names"][::-1].copy()
        data["pin_net"] = 1 - data["pin_net"]
        data["node_x"][1] += 0.5
    if mutate is not None:
        mutate(data)
    np.savez(path, **data)
    meta = dict(lef_sha256={"cells.lef": "abc"},
                excluded_special_net_names=["VDD"], def_sha256=str(path))
    path.with_suffix(".npz.json").write_text(json.dumps(meta))


def test_identity_accepts_reordering_and_reports_geometry_change(tmp_path):
    before, after = tmp_path / "before.npz", tmp_path / "after.npz"
    _snapshot(before)
    _snapshot(after, reverse=True)
    result = compare(before, after)
    assert result["overall_pass"]
    assert result["changed_node_geometry"] == 1
    assert result["n_endpoints"] == 2


@pytest.mark.parametrize("mutation", [
    lambda d: d["pin_net"].__setitem__(0, 1),
    lambda d: d.__setitem__("pin_names", np.array([d["pin_names"][0]] * 2)),
    lambda d: d.__setitem__("net_names", np.array(["n0", "different"])),
    lambda d: d.__setitem__("node_names", np.array(["renamed", "port"])),
])
def test_identity_rejects_net_endpoint_or_instance_changes(tmp_path, mutation):
    before, after = tmp_path / "before.npz", tmp_path / "after.npz"
    _snapshot(before)
    _snapshot(after, mutate=mutation)
    assert not compare(before, after)["overall_pass"]


@pytest.mark.parametrize("field,value", [
    ("lef_sha256", {"cells.lef": "changed"}),
    ("excluded_special_net_names", ["VDD", "VSS"]),
])
def test_identity_rejects_library_and_special_set_changes(tmp_path, field, value):
    before, after = tmp_path / "before.npz", tmp_path / "after.npz"
    _snapshot(before)
    _snapshot(after)
    path = after.with_suffix(".npz.json")
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    assert not compare(before, after)["overall_pass"]


def test_identity_rejects_duplicate_net_names(tmp_path):
    before, after = tmp_path / "before.npz", tmp_path / "after.npz"
    _snapshot(before)
    _snapshot(after, mutate=lambda d: d.__setitem__("net_names", np.array(["n0", "n0"])))
    with pytest.raises(ValueError, match="duplicate net"):
        compare(before, after)
