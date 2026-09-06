"""Identifiability and prediction evidence for separate component experiments."""
import numpy as np
from scipy.stats import t as student_t


def fit_model(features, observations, *, names, group_ids, condition_limit=30., relative_ci_limit=.25):
    """Fit one row per independent recipe, with an intercept and scaled QR.

    Replicate aggregation belongs to the caller. Intervals are conditional on
    the specified linear model; they do not establish transport to real designs.
    """
    x, y = np.asarray(features,dtype=float), np.asarray(observations,dtype=float)
    if (x.ndim != 2 or y.shape != (len(x),) or x.shape[1] != len(names)
            or len(group_ids) != len(x) or len(set(group_ids)) != len(x)):
        raise ValueError("expected one finite row per unique recipe and named feature")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(x[:,0] == 1):
        raise ValueError("finite data with a leading intercept column required")
    n,p=x.shape
    mean,scale=x.mean(axis=0),x.std(axis=0)
    mean[0],scale[0]=0.,1.
    scale[scale==0]=1.
    z=(x-mean)/scale
    rank=int(np.linalg.matrix_rank(z))
    condition=float(np.linalg.cond(z)) if rank==p else None
    beta_z=np.linalg.lstsq(z,y,rcond=None)[0]
    transform=np.diag(1/scale)
    transform[0,1:]=-mean[1:]/scale[1:]
    beta=transform@beta_z
    residual=y-x@beta
    dof=n-rank
    variance=float(residual@residual/dof) if dof>0 else None
    covariance=None
    halfwidth=np.full(p,np.nan)
    loo=np.full(n,np.nan)
    if rank==p and dof>0:
        q,r=np.linalg.qr(z,mode="reduced")
        inverse=np.linalg.solve(r,np.eye(p))
        covariance=transform@(variance*inverse@inverse.T)@transform.T
        halfwidth=student_t.ppf(.975,dof)*np.sqrt(np.maximum(np.diag(covariance),0))
        leverage=np.sum(q*q,axis=1)
        np.divide(residual,1-leverage,out=loo,where=(1-leverage)>1e-10)
    tolerance=np.finfo(float).eps*max(1.,float(np.max(np.abs(beta))))*100
    relative=np.divide(halfwidth,np.abs(beta),out=np.full(p,np.nan),where=np.abs(beta)>tolerance)
    ci_ok=bool(np.all(np.isfinite(relative)) and np.all(relative<=relative_ci_limit))
    passed=bool(rank==p and dof>0 and condition<=condition_limit and ci_ok)
    finite=lambda value:float(value) if np.isfinite(value) else None
    return dict(n_recipes=n,n_regressors=p,rank=rank,dof=dof,
        standardized_condition=condition,condition_limit=condition_limit,
        relative_ci_limit=relative_ci_limit,identifiable=passed,
        coefficients={name:dict(value=float(beta[i]),ci95_halfwidth=finite(halfwidth[i]),
            relative_halfwidth=finite(relative[i])) for i,name in enumerate(names)},
        covariance=covariance.tolist() if covariance is not None else None,
        residual_variance=variance,
        residuals=[dict(recipe=group_ids[i],actual=float(y[i]),prediction=float((x@beta)[i]),
                        residual=float(residual[i]),leave_one_recipe_out_residual=finite(loo[i])) for i in range(n)],
        gate_reasons=[message for failed,message in (
            (rank<p,"rank_deficient"),(dof<=0,"no_residual_degrees_of_freedom"),
            (condition is None or condition>condition_limit,"condition_gate_failed"),
            (not ci_ok,"coefficient_relative_interval_gate_failed")) if failed],
        holdout_status="not_run", extrapolation_allowed=False,
        interval_scope="conditional OLS t interval across recipe means; real-design holdouts separate")


def predict(model, values):
    x=np.asarray(values,dtype=float)
    beta=np.asarray([entry["value"] for entry in model["coefficients"].values()])
    if x.shape != beta.shape or not np.all(np.isfinite(x)):
        raise ValueError("prediction feature vector mismatch")
    point=float(x@beta)
    interval=None
    if model["identifiable"]:
        covariance=np.asarray(model["covariance"])
        width=float(student_t.ppf(.975,model["dof"])*np.sqrt(model["residual_variance"]+x@covariance@x))
        interval=[point-width,point+width]
    return dict(point=point,pi95=interval,identified_model=bool(model["identifiable"]),
                requires_external_validation=True)
