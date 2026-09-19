"""Independent invariants for the experimental per-layer route comparison."""
import numpy as np
import pytest
from ioplace.route_eval import common_grt as module


def baseline():
    return dict(x_edges_dbu=np.array([0, 10, 20, 30]),
                y_edges_dbu=np.array([0, 10, 20]),
                layer_names=np.array(['metal2', 'metal3']),
                layer_directions=np.array(['H', 'V']),
                capacity=np.ones((2, 2, 3), dtype=np.int64),
                fixed_usage=np.zeros((2, 2, 3), dtype=np.int64))


def run(tmp_path, content, cap=None):
    path = tmp_path / 'segments.txt'
    path.write_text(content)
    return module.evaluate(path, baseline() if cap is None else cap)


def test_union_within_net_and_sum_between_nets(tmp_path):
    result = run(tmp_path, 'a\n(\n5 5 metal2 25 5 metal2\n25 5 metal2 5 5 metal2\n)\nb\n(\n15 5 metal2 25 5 metal2\n)\n')
    assert result['common_wire_edge_overflow'] == 1
    assert result['wire_edge_demand'] == 3
    assert result['wirelength_dbu'] == 30
    assert result['routed_net_count'] == 2


def test_layer_separation_fixed_usage_and_vias(tmp_path):
    cap = baseline()
    cap['fixed_usage'][0, 0, 0] = 1
    result = run(tmp_path, 'a\n(\n5 5 metal2 15 5 metal2\n5 5 metal2 5 5 metal3\n5 5 metal3 5 15 metal3\n)\n', cap)
    assert result['common_wire_edge_overflow'] == 1
    assert result['wire_edge_demand'] == 2
    assert result['via_records'] == 1
    assert result['wirelength_dbu'] == 20


def test_nonpreferred_wires_are_not_silently_ignored(tmp_path):
    result = run(tmp_path, 'a\n(\n5 5 metal3 15 5 metal3\n)\n')
    assert result['nonpreferred_wire_records'] == 1
    assert result['common_wire_edge_overflow'] == 1


@pytest.mark.parametrize('content', [
    'a\n(\n-1 5 metal2 15 5 metal2\n)\n',
    'a\n(\n5 5 metal2 15 15 metal2\n)\n',
    'a\n(\n5 5 metal2 15 5 metal3\n)\n',
    'a\n(\n5 5 metal2 15 5 metal2\n',
    'a\n(\n)\na\n(\n)\n',
    'a\n(\n5 5 missing 15 5 missing\n)\n',
])
def test_invalid_geometry_or_syntax_rejected(tmp_path, content):
    with pytest.raises(ValueError):
        run(tmp_path, content)


def test_int64_demand_exceeds_native_uint8(tmp_path):
    content = ''.join(f'n{i}\n(\n5 5 metal2 15 5 metal2\n)\n' for i in range(300))
    assert run(tmp_path, content)['common_wire_edge_overflow'] == 299


def test_boundary_and_zero_length_records(tmp_path):
    result = run(tmp_path, 'a\n(\n0 5 metal2 30 5 metal2\n5 5 metal2 5 5 metal2\n)\n')
    assert result['wire_edge_demand'] == 2
    assert result['wirelength_dbu'] == 30


def test_baseline_shape_rejected(tmp_path):
    cap = baseline()
    cap['capacity'] = np.ones((2, 3))
    with pytest.raises(ValueError):
        run(tmp_path, '', cap)


def test_generic_layer_name_cannot_bypass_via_adjacency(tmp_path):
    cap = baseline()
    cap['layer_names'] = np.array(['M2', 'M4'])
    with pytest.raises(ValueError, match='nonadjacent via'):
        run(tmp_path, 'a\n(\n5 5 M2 5 5 M4\n)\n', cap)
