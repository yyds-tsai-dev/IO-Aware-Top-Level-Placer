import torch
import pytest

from ioplace.ops.ft_callback import independent_gradient, publish_atomic
from ioplace.ops.ft_term import FtTerm
from ioplace.schedules import ScheduleState
from tests.test_ft_term import _make, _pos


def test_independent_gradient_ignores_live_grad_and_masks_fixed_fillers():
    pos = torch.arange(8., dtype=torch.float64, requires_grad=True)
    pos.grad = torch.full_like(pos, 123.)
    got = independent_gradient(lambda p: (p ** 2).sum(), pos, 2, 4)
    torch.testing.assert_close(got, torch.tensor([0., 2., 0., 0., 8., 10., 0., 0.], dtype=pos.dtype))
    assert torch.all(pos.grad == 123.)


@pytest.mark.parametrize("fraction", [0., .25])
def test_atomic_transaction_publishes_current_norms_once(fraction):
    nl, io, _, D = _make(k=4, chunk=1)
    ft = FtTerm(io, D)
    ft.set_home([0])
    state = ScheduleState(rho_max=.4, f_ft_max=fraction, c_lip=1e6)
    state.update_continuous(0, .8, 100., 1.)
    state.update_continuous(100, .1, 100., 1.)
    old = state.obj_version
    pos = _pos(nl)
    record = publish_atomic(state, io, ft if fraction else None, lambda p: (p**2).sum(),
                            pos, 100, .04, float(D.max()), 1.)
    assert state.obj_version == old + 1
    assert state.needs_refresh()
    assert record["lambda_io"] == state.lambda_io > 0
    assert record["grad_l1_ft"] > 0 if fraction else record["grad_l1_ft"] == 0
    assert state.kappa_ft > 0 if fraction else state.kappa_ft == 0
    assert record["ratio_inst"] == pytest.approx(record["grad_l1_wl"] / record["grad_l1_merged"])
    state.mark_refreshed()
    assert not state.needs_refresh()
