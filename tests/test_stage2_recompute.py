import numpy as np
import pytest

from scripts.stage2_recompute_evaluator import align_netlist
from tests.test_netlist import make_tiny_netlist


def test_net_name_alignment_preserves_pin_identity_and_multiplicity():
    nl = make_tiny_netlist()
    aligned = align_netlist(nl, ["a", "b"], ["b", "a"])
    assert aligned.net_degrees.tolist() == [3, 2]
    assert aligned.pin2net.tolist() == [1, 1, 0, 0, 0]
    for new, old in enumerate((1, 0)):
        pins = aligned.flat_net2pin[aligned.flat_net2pin_start[new]:aligned.flat_net2pin_start[new + 1]]
        original = nl.flat_net2pin[nl.flat_net2pin_start[old]:nl.flat_net2pin_start[old + 1]]
        np.testing.assert_array_equal(pins, original)


def test_net_set_mismatch_rejected():
    with pytest.raises(ValueError, match="different net sets"):
        align_netlist(make_tiny_netlist(), ["a", "b"], ["a", "other"])
