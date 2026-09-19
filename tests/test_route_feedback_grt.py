import json
from scripts.verify_route_feedback_grt import parse_segments,select_router_checkpoint
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions


def test_independent_router_veto_and_rejected_candidates_are_not_selected():
    metric=lambda io,inside,wl:dict(common_net_io=io,in_die_common_io=inside,common_net_wirelength=wl)
    summary=dict(baseline=metric(414,358,100),round0_control=metric(422,365,100),
                 round0_accepted=metric(415,359,100),round0_candidate1=metric(1,1,1))
    assert select_router_checkpoint(summary)=='baseline'
    summary['round0_accepted']=metric(410,350,106)
    assert select_router_checkpoint(summary)=='baseline'
    summary['round0_accepted']=metric(410,350,104)
    assert select_router_checkpoint(summary)=='round0_accepted'


def test_global_route_segment_decode_keeps_layers_and_merges_shared_branches(tmp_path):
    path=tmp_path/'segments.txt'
    path.write_text('net_a\n(\n0 5 metal1 10 5 metal1\n5 5 metal1 10 5 metal1\n'
                    '0 5 metal2 10 5 metal2\n10 5 metal1 10 5 metal2\n)\n')
    rg=RegionGrid(make_grid_regions((0,0,10,10),2,1,lattice=10))
    result=parse_segments(path,dict(shift_factor=[0,0],scale_factor=1),rg)
    assert result['net_a']['wirelength']==20
    assert result['net_a']['io']==2
