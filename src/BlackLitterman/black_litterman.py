from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _normalize_weights(weights: pd.Series) -> pd.Series:
    """Normalize a weight series to sum to one while keeping zeros stable."""
    cleaned = pd.Series(np.nan_to_num(weights.to_numpy(dtype=float), nan=0.0), index=weights.index, dtype=float)
    total = float(cleaned.sum())
    if np.isclose(total, 0.0):
        cleaned = pd.Series(np.ones(len(cleaned), dtype=float) / len(cleaned), index=cleaned.index, dtype=float)
        return cleaned
    return cleaned / total


def _prepare_market_weights(market_weights: Sequence[float] | None, n_assets: int) -> pd.Series:
    if market_weights is None:
        values = np.ones(n_assets, dtype=float) / n_assets
    else:
        values = np.asarray(list(market_weights), dtype=float)
        if values.shape[0] != n_assets:
            raise ValueError(f"Expected {n_assets} market weights but received {values.shape[0]}.")
        if np.isclose(values.sum(), 0.0):
            raise ValueError("Market weights cannot sum to zero.")
        values = values / values.sum()
    return pd.Series(values, dtype=float)


def _prepare_views(views: Sequence[float] | None, n_assets: int) -> np.ndarray | None:
    if views is None:
        return None

    values = np.asarray(list(views), dtype=float)
    if values.ndim == 0:
        values = np.full(n_assets, float(values))
    if values.shape[0] != n_assets:
        raise ValueError(f"Expected {n_assets} views but received {values.shape[0]}.")

    if np.max(np.abs(values)) > 1:
        values = values / 100.0
    return values


def _prepare_confidence(confidence: Sequence[float] | float | None, n_assets: int) -> np.ndarray:
    if confidence is None:
        return np.ones(n_assets, dtype=float)

    if np.isscalar(confidence):
        return np.full(n_assets, float(confidence), dtype=float)

    values = np.asarray(list(confidence), dtype=float)
    if values.ndim == 0:
        values = np.full(n_assets, float(values), dtype=float)
    if values.shape[0] != n_assets:
        raise ValueError(f"Expected {n_assets} confidence values but received {values.shape[0]}.")
    return np.clip(values, 1e-6, None)


def black_litterman_weights(
    returns: pd.DataFrame,
    market_weights: Sequence[float] | None = None,
    views: Sequence[float] | None = None,
    view_confidence: Sequence[float] | float | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    allow_short: bool = False,
) -> pd.Series:
    """
    Black-Litterman portfolio weights from a historical return covariance matrix.

    The function follows the standard model:
      - Prior equilibrium returns: pi = lambda * Sigma * w_mkt
      - Posterior expected returns with investor views
      - Optimal portfolio: w = (lambda * Sigma)^-1 * E[R]
    """
    if returns is None or returns.empty:
        raise ValueError("A non-empty return matrix is required for Black-Litterman allocation.")

    returns = returns.dropna(axis=1)
    if returns.empty:
        raise ValueError("No valid columns remain after dropping missing data.")

    columns = list(returns.columns)
    if len(columns) == 1:
        return pd.Series([1.0], index=columns, dtype=float)

    market_weights_series = _prepare_market_weights(market_weights, len(columns))
    market_weights_series.index = columns

    covariance = returns.cov().fillna(0.0) * 252.0
    covariance = covariance.reindex(index=columns, columns=columns).fillna(0.0)

    if np.linalg.matrix_rank(covariance.to_numpy()) < covariance.shape[0]:
        covariance = covariance + np.eye(covariance.shape[0]) * 1e-6

    prior_returns = risk_aversion * (covariance @ market_weights_series.to_numpy())
    prior_returns = pd.Series(prior_returns, index=columns, dtype=float)

    if views is None:
        weights = np.linalg.pinv(risk_aversion * covariance.to_numpy()) @ prior_returns.to_numpy()
        weights = pd.Series(weights, index=columns, dtype=float)
        return _normalize_weights(_finalize_weights(weights, allow_short))

    q = _prepare_views(views, len(columns))
    if q is None:
        q = prior_returns.to_numpy()

    omega = _prepare_confidence(view_confidence, len(columns))
    omega_diag = np.diag((1.0 / omega) * tau)
    view_matrix = np.eye(len(columns), dtype=float)
    implied_inverse = np.linalg.pinv(tau * covariance.to_numpy())
    posterior_mean = np.linalg.pinv(
        implied_inverse + view_matrix.T @ np.linalg.pinv(omega_diag) @ view_matrix
    ) @ (
        implied_inverse @ prior_returns.to_numpy()
        + view_matrix.T @ np.linalg.pinv(omega_diag) @ q
    )

    weights = np.linalg.pinv(risk_aversion * covariance.to_numpy()) @ posterior_mean
    weights = pd.Series(weights, index=columns, dtype=float)
    return _normalize_weights(_finalize_weights(weights, allow_short))


def _finalize_weights(weights: pd.Series, allow_short: bool) -> pd.Series:
    cleaned = pd.Series(np.nan_to_num(weights.to_numpy(dtype=float), nan=0.0), index=weights.index, dtype=float)
    if not allow_short:
        cleaned = cleaned.clip(lower=0.0)
    total = float(cleaned.sum())
    if np.isclose(total, 0.0):
        cleaned = pd.Series(np.ones(len(cleaned), dtype=float) / len(cleaned), index=cleaned.index, dtype=float)
        return cleaned
    return cleaned / total
