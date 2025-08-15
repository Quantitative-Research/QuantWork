# convex_projection.py
"""
Projection L2 d'une slice de volatilités (sigma_K) sur l'ensemble des
fonctions convexes en K (discret) — deux méthodes :
 - 'slopes' : isotonic regression sur les pentes (rapide, sans solveur)
 - 'qp'     : résolution par QP (cvxpy), flexible (contraintes additionnelles possibles)

Usage:
    from convex_projection import project_convex
    sigma_hat = project_convex(K, sigma, method='slopes', weights=vega)
"""

from typing import Optional
import numpy as np

# méthode 'slopes' requiert sklearn
try:
    from sklearn.isotonic import IsotonicRegression
except Exception as e:
    IsotonicRegression = None

# méthode 'qp' optionnelle (cvxpy)
try:
    import cvxpy as cp
except Exception:
    cp = None


def _validate_inputs(K, sigma, weights: Optional[np.ndarray]):
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if K.ndim != 1 or sigma.ndim != 1:
        raise ValueError("K and sigma must be 1-D arrays.")
    if len(K) != len(sigma):
        raise ValueError("K and sigma must have same length.")
    if not np.all(np.diff(K) > 0):
        raise ValueError("K must be strictly increasing.")
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != sigma.shape:
            raise ValueError("weights must have same shape as sigma.")
    return K, sigma


def convex_proj_by_slopes(K, sigma, weights: Optional[np.ndarray] = None, anchor: str = "ls"):
    """
    Projection L2 onto convex sequences by isotonic regression on slopes.

    Args:
        K: strictly increasing strikes (n,)
        sigma: observed vols (n,)
        weights: optional node weights for L2 norm (n,)
        anchor: "ls" (least-squares optimal vertical shift) or "fix_first" (keep sigma_hat[0]=sigma[0])

    Returns:
        sigma_hat (n,)
    """
    if IsotonicRegression is None:
        raise ImportError("sklearn.isotonic.IsotonicRegression is required for 'slopes' method.")

    K, y = _validate_inputs(K, sigma, weights)
    n = len(y)
    if n < 3:
        return y.copy()

    dK = np.diff(K)            # n-1
    slopes = np.diff(y) / dK   # n-1

    # slope weights: natural choice dK, optionally combine node-weights
    slope_weights = dK.copy()
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        slope_weights = w[:-1] + w[1:]

    # Fit isotonic on slopes (increasing => convex s)
    idx = np.arange(len(slopes))
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    slopes_hat = ir.fit_transform(idx, slopes, sample_weight=slope_weights)

    # Reconstruct sigma_hat from slopes_hat
    sigma_hat = np.empty(n, dtype=float)
    sigma_hat[0] = y[0]  # anchor initial
    for i in range(n - 1):
        sigma_hat[i + 1] = sigma_hat[i] + slopes_hat[i] * dK[i]

    # Optionally apply vertical shift alpha to minimize weighted L2 residuals
    if anchor == "ls":
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            alpha = np.sum(w * (y - sigma_hat)) / np.sum(w)
        else:
            alpha = np.mean(y - sigma_hat)
        sigma_hat = sigma_hat + alpha
    elif anchor == "fix_first":
        # keep sigma_hat[0] equal to y[0] (already the case), do nothing
        pass
    else:
        raise ValueError("anchor must be 'ls' or 'fix_first'")

    return sigma_hat


def convex_proj_qp(K, sigma, weights: Optional[np.ndarray] = None, solver: str = "OSQP"):
    """
    Projection L2 onto convex sequences using quadratic programming (cvxpy).

    Minimize 0.5 * sum_i w_i * (s_i - sigma_i)^2 subject to discrete convexity:
        (s_{i+2}-s_{i+1})/dK[i+1] - (s_{i+1}-s_i)/dK[i] >= 0  for i=0..n-3

    Args:
        K, sigma: arrays of length n
        weights: optional weights per node (n,) -> corresponds to weighted L2
        solver: preferred cvxpy solver (e.g. "OSQP", "SCS", "ECOS")

    Returns:
        sigma_hat (n,)
    """
    if cp is None:
        raise ImportError("cvxpy is required for 'qp' method. Install cvxpy.")

    K, y = _validate_inputs(K, sigma, weights)
    n = len(y)
    if n < 3:
        return y.copy()

    dK = np.diff(K)  # length n-1
    m = n - 2
    # Build A matrix such that A @ s >= 0 en convexe (m x n)
    A = np.zeros((m, n), dtype=float)
    for i in range(m):
        # inequality: (s_{i+2}-s_{i+1})/dK[i+1] - (s_{i+1}-s_i)/dK[i] >= 0
        A[i, i] = -1.0 / dK[i]
        A[i, i + 1] = 1.0 / dK[i] + -1.0 / dK[i + 1]
        A[i, i + 2] = 1.0 / dK[i + 1]
        # rearranged to A s >= 0

    s = cp.Variable(n)

    if weights is None:
        objective = 0.5 * cp.sum_squares(s - y)
    else:
        w = np.asarray(weights, dtype=float)
        # objective: 0.5 * sum w_i * (s_i - y_i)^2
        objective = 0.5 * cp.sum(cp.multiply(w, cp.square(s - y)))

    prob = cp.Problem(cp.Minimize(objective), [A @ s >= 0])
    # try solve
    try:
        prob.solve(solver=getattr(cp, solver) if hasattr(cp, solver) else solver)
    except Exception:
        # fallback to default solver
        prob.solve()

    if s.value is None:
        raise RuntimeError("QP solver failed to return a solution.")
    return np.asarray(s.value, dtype=float)


def project_convex(K, sigma, method: str = "slopes", weights: Optional[np.ndarray] = None, **kwargs):
    """
    Convenience wrapper to project sigma on convex functions in K.

    Args:
        method: 'slopes' or 'qp'
        weights: optional weights per node (n,)
        kwargs: forwarded to the chosen method

    Returns:
        sigma_hat (n,)
    """
    method = method.lower()
    if method == "slopes":
        return convex_proj_by_slopes(K, sigma, weights=weights, **kwargs)
    elif method == "qp":
        return convex_proj_qp(K, sigma, weights=weights, **kwargs)
    else:
        raise ValueError("method must be 'slopes' or 'qp'.")

def convex_monotone_projection(K, y, monotone: str):
    """
    L2 projection of a price curve y(K) onto the cone of convex + monotone functions.
    monotone: 'nonincreasing' for calls, 'nondecreasing' for puts
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = y.size
    if n < 3:
        return y.copy()

    dK = np.diff(K)
    slopes = np.diff(y) / dK           # length n-1

    # convexity = nondecreasing slopes
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    slopes_hat = ir.fit_transform(np.arange(n-1), slopes, sample_weight=dK)

    # enforce monotonicity sign on slopes
    if monotone == "nonincreasing":
        slopes_hat = np.minimum(slopes_hat, 0.0)
    elif monotone == "nondecreasing":
        slopes_hat = np.maximum(slopes_hat, 0.0)
    else:
        raise ValueError("monotone must be 'nonincreasing' or 'nondecreasing'")

    # reconstruct y_hat from slopes_hat, with optimal vertical shift (least squares)
    y_hat = np.empty_like(y)
    y_hat[0] = y[0]
    for i in range(n-1):
        y_hat[i+1] = y_hat[i] + slopes_hat[i] * dK[i]

    alpha = (y - y_hat).mean()   # least-squares optimal vertical shift
    return y_hat + alpha

