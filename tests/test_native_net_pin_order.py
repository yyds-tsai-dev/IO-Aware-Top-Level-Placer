import numpy as np
from ioplace.bench.native_pin_order import native_net_pin_order

def test_native_output_rotation_is_not_stable_partition():
    # One five-pin net; outputs are b,d,e (indices 1,3,4).
    got = native_net_pin_order([5], [False, True, False, True, True])
    assert got.tolist() == [4, 0, 2, 1, 3]

def test_mixed_nets_zero_degree_and_first_pin_output():
    deg = [0, 3, 2, 1]
    flags = [True, False, True, False, False, True]
    got = native_net_pin_order(deg, flags)
    assert got.tolist() == [2, 1, 0, 3, 4, 5]

def test_invalid_degree_and_flag_lengths_rejected():
    try:
        native_net_pin_order([2, 2], [False])
    except ValueError:
        pass
    else:
        raise AssertionError("expected mismatched flags to fail")
