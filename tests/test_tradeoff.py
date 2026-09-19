import pytest
from ioplace.bench.tradeoff import assess, select


def metrics(**kw):
    return dict(dict(workload_status="completed", legalization_status="success",
                     num_unplaced_cells=0, stop_overflow_reached=True,
                     hpwl=100., io_count=100, ft_count=10, config="same.json",
                     k=16, rtype="grid", seed=0, dp_seed=1000, det=1,
                     benchmark_kind="real"), **kw)


def test_budget_controls_actual_selected_placement():
    baseline = metrics()
    candidates = [metrics(io_count=90, hpwl=101), metrics(io_count=70, hpwl=104),
                  metrics(io_count=50, hpwl=108)]
    assert select(baseline, candidates, hpwl_budget=.02)["selected_index"] == 0
    assert select(baseline, candidates, hpwl_budget=.05)["selected_index"] == 1
    assert select(baseline, candidates, hpwl_budget=.10)["selected_index"] == 2
    assert select(baseline, candidates, hpwl_budget=0)["selected_index"] is None


def test_ft_reduction_cannot_hide_increased_io():
    assert not assess(metrics(), metrics(io_count=101, ft_count=0))["accepted"]


@pytest.mark.parametrize("kw", [dict(legalization_status="failed"),
    dict(num_unplaced_cells=1), dict(stop_overflow_reached=False), dict(dp_seed=2),
    dict(hpwl=float("nan")), dict(io_count=-1), dict(ft_count=float("inf"))])
def test_invalid_or_unmatched_candidate_rejected(kw):
    assert not assess(metrics(), metrics(**{"io_count": 80, **kw}))["accepted"]


def test_routing_is_an_independent_required_measurement_when_requested():
    b, c = metrics(routed_wirelength=150.), metrics(io_count=80, routed_wirelength=170.)
    assert assess(b, c)["accepted"]
    assert not assess(b, c)["routing_verified"]
    assert not assess(b, c, routed_budget=.05)["accepted"]
    assert assess(b, c, routed_budget=.15)["accepted"]
    assert not assess(metrics(), c, routed_budget=.15)["accepted"]


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_bad_budget_rejected(budget):
    with pytest.raises(ValueError):
        assess(metrics(), metrics(), hpwl_budget=budget)
