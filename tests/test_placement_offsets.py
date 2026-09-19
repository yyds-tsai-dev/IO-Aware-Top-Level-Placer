import numpy as np
import pytest
from tests.test_route_gp import fixture


def test_oriented_offsets_copy_movable_canonical_pins_and_preserve_fixed():
    from ioplace.ops.placement_offsets import oriented_netlist
    nl,_,_,_=fixture();nl.num_movable=1
    nl.node_size_x[:]=2.;nl.node_size_y[:]=3.
    nl.pin_offset_x[:]=.4;nl.pin_offset_y[:]=.7
    for orientation,expected in [('N',(.4,.7)),('MX',(.4,2.3)),('FN',(1.6,.7)),('R180',(1.6,2.3))]:
        changed=oriented_netlist(nl,[orientation])
        assert (changed.pin_offset_x[0],changed.pin_offset_y[0])==pytest.approx(expected)
        assert changed.pin_offset_x[1]==.4 and changed.pin_offset_y[1]==.7
        np.testing.assert_array_equal(nl.pin_offset_x,[.4,.4])
        np.testing.assert_array_equal(nl.pin_offset_y,[.7,.7])
        assert not np.shares_memory(changed.pin_offset_x,nl.pin_offset_x)


@pytest.mark.parametrize('orientations',[[],['R90'],['UNKNOWN'],['N','N']])
def test_unsupported_orientation_cannot_silently_calibrate(orientations):
    from ioplace.ops.placement_offsets import oriented_netlist
    nl,_,_,_=fixture();nl.num_movable=1
    with pytest.raises(ValueError):oriented_netlist(nl,orientations)


def test_measured_orientation_changes_calibration_without_changing_live_graph():
    import torch
    from ioplace.ops.route_gp import FrozenJointRouteCost
    nl,rg,grid,pos=fixture()
    nl.node_size_x[:]=2.;nl.pin_offset_x[:]=.4
    pos=pos.detach().clone();pos[0]=4.2
    op=FrozenJointRouteCost(nl,rg,grid,np.ones(grid.edge_count),num_nodes=3,hot_nets=0)
    op.rebuild(pos)
    before=op.components(pos,tau=1.,gamma=.1,hard=True)
    assert before['io_raw']==1
    obs=dict(grid=grid,capacity=np.ones(grid.edge_count),usage=np.ones(grid.edge_count),
        net_io={0:1},net_usage={0:dict(keys=[],counts=[])},actual_io=1,actual_wirelength=4.,
        source_sha256='fixture',placement_sha256='fixture',movable_orientations=['FN','N','N'])
    record=op.assimilate(obs,pos)
    assert record['measured_pin_model']=='oriented row pins'
    assert op.io_calibration_np[0]==pytest.approx(1.5) # actual1 / oriented prediction0, with +1 smoothing
    assert op.generation==1 and op.published_observation_version==0
    assert op.components(pos,tau=1.,gamma=.1,hard=True)['io_raw']==1
    torch.testing.assert_close(op.io_weights,torch.ones_like(op.io_weights))
    np.testing.assert_array_equal(nl.pin_offset_x,[.4,.4])
