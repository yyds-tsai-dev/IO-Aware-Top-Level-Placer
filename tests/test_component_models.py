import numpy as np
import pytest
from ioplace.diagnostics.component_models import fit_model, predict


def test_scaled_qr_covariance_and_grouped_loo_match_explicit_refits():
    rng=np.random.default_rng(42)
    x=np.column_stack((np.ones(40),rng.uniform(1e6,3e6,40),rng.uniform(5e7,9e7,40)))
    y=x@np.array([5.,2e-5,3e-7])+rng.normal(0,.01,40)
    model=fit_model(x,y,names=["intercept","N","P"],group_ids=list(range(40)))
    assert model["identifiable"]
    assert not model["extrapolation_allowed"]
    beta=np.array([entry["value"] for entry in model["coefficients"].values()])
    assert np.allclose(beta,np.linalg.lstsq(x,y,rcond=None)[0],rtol=1e-8)
    for i in (0,5,20):
        keep=np.arange(len(x))!=i
        expected=y[i]-x[i]@np.linalg.lstsq(x[keep],y[keep],rcond=None)[0]
        assert model["residuals"][i]["leave_one_recipe_out_residual"]==pytest.approx(expected,abs=1e-7)
    point=predict(model,x[0])
    assert point["pi95"][0] < point["point"] < point["pi95"][1]


def test_rank_deficient_and_zero_coefficient_do_not_pass():
    x=np.column_stack((np.ones(20),np.arange(20),np.arange(20)))
    model=fit_model(x,x[:,1],names=["intercept","a","b"],group_ids=list(range(20)))
    assert not model["identifiable"]
    assert predict(model,x[0])["pi95"] is None
    x=np.column_stack((np.ones(20),np.arange(20)))
    model=fit_model(x,np.arange(20),names=["intercept","a"],group_ids=list(range(20)))
    assert not model["identifiable"]
    assert "coefficient_relative_interval_gate_failed" in model["gate_reasons"]


def test_repeated_recipe_rows_rejected():
    with pytest.raises(ValueError,match="unique recipe"):
        fit_model([[1,2],[1,3]],[2,3],names=["intercept","a"],group_ids=["same","same"])
