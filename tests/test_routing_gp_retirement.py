import pytest

torch = pytest.importorskip("torch")

from ioplace.norm import TermNormalizer
from ioplace.ops.routing_gp_controller import RoutingGPController


def test_controller_is_retired_unless_the_escape_hatch_is_set(monkeypatch):
    monkeypatch.delenv("IOPLACE_ENABLE_GR_IN_LOOP", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        RoutingGPController(object(), object())
    assert "IOPLACE_ENABLE_GR_IN_LOOP" in str(excinfo.value)


def test_controller_constructs_behind_the_escape_hatch(monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    controller = RoutingGPController(object(), object(), route_strength=.25)
    assert controller.route_strength == .25
    assert controller.route_lambda == 0.


def test_controller_still_validates_its_arguments(monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    with pytest.raises(ValueError):
        RoutingGPController(object(), object(), mode="bogus")
    with pytest.raises(ValueError):
        RoutingGPController(object(), object(), start=0)
    with pytest.raises(ValueError):
        RoutingGPController(object(), object(), tau=0.)


def test_oneshot_lambda_matches_the_retired_expression():
    for strength, wa_l1, norm in ((.1, 1234.5, 6.78), (.25, 1e6, 3.),
                                  (1., 7.5, 2.5)):
        assert (TermNormalizer.oneshot_lambda(strength, wa_l1, norm)
                == strength * wa_l1 / norm)
    assert TermNormalizer.oneshot_lambda(.1, 1000., 1e-13) == 0.


def test_calibrate_uses_the_normalizer(monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")

    class _Term(object):
        def paper(self, pos, gamma):
            return (pos ** 2).sum()

        def components(self, pos, tau, gamma):
            value = (pos ** 3).sum()
            return {"io": value, "wirelength": value, "congestion": value,
                    "objective": value}

    class _Model(object):
        class _Ops(object):
            wirelength_op = staticmethod(lambda p: 4. * p.sum())
        op_collections = _Ops()

    class _Placer(object):
        model = _Model()

    controller = RoutingGPController(_Term(), _Placer(), mode="joint",
                                     route_strength=.1)
    controller.gamma = 1.
    pos = torch.tensor([1., 2., 3.], dtype=torch.float64)
    stats = controller._calibrate(pos)
    wa_l1 = 12.                                     # |4| three times
    route_l1 = 3. + 12. + 27.                       # |3 p^2|
    assert stats["wa_gradient_l1"] == pytest.approx(wa_l1, rel=1e-12)
    assert stats["route_lambda"] == .1 * wa_l1 / route_l1
    assert controller.route_lambda == stats["route_lambda"]
