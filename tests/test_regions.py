import numpy as np
import pytest
from ioplace.regions import RegionSet, RegionSpec, make_grid_regions, make_slicing_regions

DIE = (0., 0., 1024., 1024.)

def test_grid_regions_cover_and_count():
    rs = make_grid_regions(DIE, 4, 2, lattice=512)
    assert rs.k == 8
    rs.validate()
    total = sum(((r.rects[:, 2]-r.rects[:, 0])*(r.rects[:, 3]-r.rects[:, 1])).sum()
                for r in rs.regions)
    assert total == pytest.approx(1024.*1024.)

def test_slicing_regions_valid_and_seeded():
    rs = make_slicing_regions(DIE, 16, seed=7)
    assert rs.k == 16
    rs.validate()
    rs2 = make_slicing_regions(DIE, 16, seed=7)
    assert all(np.array_equal(a.rects, b.rects) for a, b in zip(rs.regions, rs2.regions))

def test_validate_rejects_overlap():
    bad = RegionSet(die=DIE, lattice=512, regions=[
        RegionSpec("P0", np.array([[0., 0., 600., 1024.]])),
        RegionSpec("P1", np.array([[500., 0., 1024., 1024.]]))])
    with pytest.raises(ValueError):
        bad.validate()

def test_json_roundtrip(tmp_path):
    rs = make_grid_regions(DIE, 2, 2)
    p = tmp_path / "r.json"
    rs.to_json(str(p))
    rs2 = RegionSet.from_json(str(p))
    assert rs2.k == 4 and rs2.lattice == rs.lattice
    assert np.array_equal(rs2.regions[0].rects, rs.regions[0].rects)
