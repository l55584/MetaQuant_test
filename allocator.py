# allocator.py - OPTIMIZED FOR COMPETITION METRICS
import numpy as np
import pandas as pd
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)


def get_target_weights(
    asset_list, price_data, alpha=0.95, risk_aversion=1.3
):  # Increased risk aversion
    """CVaR portfolio optimization optimized for Sortino/Sharpe/Calmar ratios"""

    if not asset_list:
        raise ValueError("asset_list cannot be empty")

    if len(asset_list) == 1:
        return {asset_list[0]: 1.0}

    # Filter to available assets
    available_assets = [asset for asset in asset_list if asset in price_data.columns]
    if len(available_assets) == 0:
        raise ValueError("No assets with price data available")

    prices = price_data[available_assets].dropna()

    if len(prices) < 50:
        logger.warning("Insufficient data points, using equal weights")
        return {asset: 1.0 / len(available_assets) for asset in available_assets}

    # Compute returns with focus on downside risk (for Sortino ratio)
    returns = prices.pct_change().dropna()
    n = len(available_assets)

    # Enhanced smoothing for stability
    mu = returns.ewm(span=30).mean().iloc[-1].values

    # Focus on downside covariance for Sortino optimization
    downside_returns = returns[returns < 0].fillna(0)
    covariance_matrix = (
        downside_returns.ewm(span=30)
        .cov()
        .iloc[-n * len(returns) :]
        .groupby(level=1)
        .mean()
    )

    # CVaR optimization with competition metrics focus
    w = cp.Variable(n)
    z = cp.Variable(len(returns))
    VaR = cp.Variable()

    portfolio_returns = returns.values @ w
    losses = -portfolio_returns

    # Conservative constraints for better Calmar ratio
    constraints = [
        cp.sum(w) == 1,
        w >= 0.05,  # Minimum 5% allocation
        w <= 0.25,  # Maximum 25% allocation (reduced concentration)
        z >= 0,
        z >= losses - VaR,
    ]

    CVaR = VaR + (1 / (1 - alpha)) * cp.sum(z) / len(returns)

    # Objective function weighted towards competition metrics
    # Higher risk aversion for better Sortino/Sharpe
    objective = cp.Maximize(mu @ w - risk_aversion * CVaR)

    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if w.value is None:
            logger.warning("CVaR optimization failed - using equal weights")
            weights = np.ones(n) / n
        else:
            weights = w.value

        # Enhanced normalization for stability
        weights = np.clip(weights, 0.04, 0.3)
        weights /= np.sum(weights)

        # Aggressive rounding to prevent unnecessary rebalancing
        weights = np.round(weights, 2)
        weights /= np.sum(weights)

        result = dict(zip(available_assets, weights))
        logger.info(f"✅ COMPETITION Portfolio weights: {len(result)} assets")
        logger.info(f"📊 Weight range: {min(weights):.3f} - {max(weights):.3f}")
        return result

    except Exception as e:
        logger.error(f"Optimization error: {e} - using equal weights")
        weights = np.ones(n) / n
        return dict(zip(available_assets, weights))