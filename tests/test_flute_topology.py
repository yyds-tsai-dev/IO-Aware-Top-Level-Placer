import numpy as np
import pytest
from ioplace.route_eval.topology import flute_edges, segment_union, union_metrics
from ioplace.route_eval.budgeted import BudgetedRouter, route_net
from tests.test_budgeted_routing import dense_regions


def test_actual_flute_three_pin_steiner_beats_mst():
    edges,meta=flute_edges([[5,5],[95,5],[50,95]])
    assert np.abs(np.diff(edges,axis=1)).sum()==180
    assert meta['topology']=='flute' and meta['source_sha256']
    r=route_net(BudgetedRouter(dense_regions()),[5,95,50],[5,5,95],topology='flute')
    assert r['union']['wirelength']<=180*1.05


def test_shared_partial_reversed_segments_count_once():
    lines=np.array([[[5,50],[70,50]],[[95,50],[40,50]],[[20,50],[50,50]]])
    union=segment_union(lines)
    np.testing.assert_array_equal(union,[[[5,50],[95,50]]])
    metric=union_metrics(dense_regions(),lines)
    assert metric['wirelength']==90 and metric['crossings']==11


def test_duplicate_pins_and_quantization_reconnect_exact_terminals():
    pins=np.array([[5.00011,5.00027],[5.00011,5.00027],[95.00015,5.00029],[50.00017,95.00021]])
    edges,_=flute_edges(pins)
    for p in pins:
        assert np.any(np.all(edges.reshape(-1,2)==p,axis=1))
    assert flute_edges([[2,2],[2,2]])[0].shape==(0,2,2)
    with pytest.raises(ValueError):flute_edges([[0,0],[1e20,0]])


def test_flute_random_tree_union_budget_and_pin_connectivity():
    rng=np.random.default_rng(21);router=BudgetedRouter(dense_regions())
    for degree in (4,9,12,32):
        pins=rng.integers(1,99,(degree,2)).astype(float)
        route=route_net(router,*pins.T,topology='flute')
        segments=route['union']['segments']
        # Every terminal lies on the geometric union, including shared branches.
        for pin in pins:
            assert any(np.all(pin>=np.minimum(a,b)) and np.all(pin<=np.maximum(a,b)) for a,b in segments)
        assert route['union']['wirelength']<=route['baseline_union']['wirelength']*1.05
        assert route['union']['crossings']<=route['baseline_union']['crossings']
