# Black-Litterman Model

The Black-Litterman model combines a market equilibrium prior with investor views to produce a more robust asset allocation than a pure mean-variance optimization.

## 1. The intuition

A traditional Markowitz optimizer can be highly sensitive to estimated expected returns and covariance matrices. In practice, these inputs are noisy. Black-Litterman solves this by starting from a market-implied equilibrium set of returns and then blending those with the investor's views.

The result is a posterior estimate of expected returns that is less fragile than raw historical estimates and more aligned with market structure.

## 2. Market equilibrium prior

Let:

- $w_{mkt}$ be the market-capitalization portfolio weights
- $\Sigma$ be the covariance matrix of asset returns
- $\lambda$ be the risk-aversion coefficient

The implied equilibrium excess returns are:

$$
\Pi = \lambda \Sigma w_{mkt}
$$

This is the prior view of expected returns implied by the market.

## 3. Investor views

The investor contributes a set of views such as:

- Asset A will outperform the market by 4%
- Asset B will underperform by 2%
- Asset C will return 8%

In matrix form, this becomes:

$$
Q = P \times E[R]
$$

where:

- $Q$ is the vector of view returns
- $P$ is a matrix mapping the asset universe to the chosen views
- $E[R]$ is the posterior expected return vector

## 4. Posterior expected returns

The Black-Litterman posterior return estimate is:

$$
\mu_{BL} = \left[(\tau \Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1}
\left[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q\right]
$$

where:

- $\tau$ is a scaling parameter for the uncertainty of the prior
- $\Omega$ is the diagonal matrix of view variances (confidence levels)

This keeps the market prior anchored while allowing the investor's beliefs to tilt the portfolio.

## 5. Portfolio weights

Once the posterior expected returns are computed, the portfolio can be optimized using the usual inverse-covariance relation:

$$
w = (\lambda \Sigma)^{-1} \mu_{BL}
$$

After solving for the weights, they are normalized and clipped to remain long-only if desired.

## 6. Why it is useful

Compared with pure Markowitz optimization, Black-Litterman is typically more stable because it:

- starts from market-consistent equilibrium returns,
- incorporates subjective views without overreacting,
- reduces estimation error in expected returns,
- produces more realistic, diversified weights.

## 7. Practical implementation in this project

This project uses the Black-Litterman model to build an allocation dashboard from historical returns and optional investor views. The model supports:

- multi-asset return estimation from Yahoo Finance price history,
- a market prior based on equal or user-defined starting weights,
- optional investor return views and confidence levels,
- a long-only or short-enabled final allocation.

The implementation lives in the Black-Litterman dashboard package and mirrors the structure of the original Markowitz dashboard while keeping the Markowitz code untouched.
