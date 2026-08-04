import numpy as np
import pytest
torch = pytest.importorskip("torch")
from ioplace.reweight import update_net_weights

def test_update_net_weights_formula_and_inplace():
    w = torch.ones(5)
    wid = id(w)
    update_net_weights(w, np.array([0, 1, 2, 20, 5]), alpha=0.5, cap=10.0)
    assert id(w) == wid
    assert torch.allclose(w, torch.tensor([1.0, 1.5, 2.0, 6.0, 3.5]))

# NOTE on the benchmark choice below (coordinator override; brief Step 2
# originally specified install/test/simple.json):
#
# `simple` (8 movable cells) has a non-deterministic filler-sizing bug (see
# tests/test_fence_inject.py's note on the same benchmark) and is too small
# for a meaningful reweighting signal, so this integration test targets
# adaptec1 instead -- a full flat GP+LG run on adaptec1 takes ~1.5-3 minutes.
# `every=100` (vs. the brief's `every=20`, sized for `simple`'s handful of
# iterations) is scaled to adaptec1's iteration count instead. Assertions are
# otherwise unchanged from the brief.
@pytest.mark.slow
def test_reweight_adaptec1_runs_and_calls_back(tmp_path):
    import os
    from ioplace.drivers.run_placement_reweight import run_reweight
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    res = run_reweight(os.path.join(root, "install/test/ispd2005/adaptec1.json"),
                       4, "grid", 0, str(tmp_path / "rw.json"), every=100, alpha=0.5)
    assert res["mode"] == "reweight"
    assert res["num_reweights"] >= 1
