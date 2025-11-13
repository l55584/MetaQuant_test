# qubo_optimizer.py - COMPLETE ADAPTIVE VERSION (Mac Compatible)
import pandas as pd
import numpy as np
import logging
from pyqubo import Array
import dimod

logger = logging.getLogger(__name__)


class MarketHealthDetector:
    def __init__(self):
        self.health_history = []

    def assess_market_health(self, returns_data):
        """Comprehensive market health assessment"""
        if returns_data.empty:
            return "UNKNOWN", 14400

        try:
            # Health indicators
            avg_return = returns_data.mean().mean()
            volatility = returns_data.std().mean()
            negative_ratio = (returns_data < 0).mean().mean()

            # Health scoring
            return_score = 40 * (
                1 if avg_return > -0.001 else 0.5 if avg_return > -0.005 else 0
            )
            vol_score = 30 * (1 - min(volatility / 0.08, 1))
            trend_score = 20 * (1 if returns_data.tail(24).mean().mean() > 0 else 0.3)
            consistency_score = 10 * (1 - negative_ratio)

            health_score = return_score + vol_score + trend_score + consistency_score

            # Adaptive cycles
            if health_score >= 80:
                state, cycle = "VERY_HEALTHY", 7200  # 2 hours
            elif health_score >= 60:
                state, cycle = "HEALTHY", 10800  # 3 hours
            elif health_score >= 40:
                state, cycle = "WEAK", 18000  # 5 hours
            else:
                state, cycle = "UNHEALTHY", 28800  # 8 hours

            logger.info(
                f"🏥 Market Health: {state} (Score: {health_score:.1f}) → {cycle/3600}h"
            )
            return state, cycle

        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            return "UNKNOWN", 14400


class MarketRegimeDetector:
    def __init__(self):
        self.volatility_thresholds = {"low": 0.02, "high": 0.05}

    def get_optimal_n(self, returns_data):
        avg_volatility = returns_data.std().mean()

        if avg_volatility > self.volatility_thresholds["high"]:
            regime = "HIGH_VOLATILITY"
            n_assets = 12
        elif avg_volatility < self.volatility_thresholds["low"]:
            regime = "LOW_VOLATILITY"
            n_assets = 5
        else:
            regime = "NORMAL"
            n_assets = 8

        logger.info(f"Market Regime: {regime} → n={n_assets}")
        return n_assets, regime


class AdaptiveQUBOOptimizer:
    def __init__(self, base_lambda_risk=0.7):
        self.regime_detector = MarketRegimeDetector()
        self.health_detector = MarketHealthDetector()
        self.base_lambda_risk = base_lambda_risk
        self.lambda_history = []

    def get_adaptive_alpha_weights(self, market_state):
        """Dynamic alpha weights based on market health"""
        if market_state == "VERY_HEALTHY":
            return (0.50, 0.30, 0.20)
        elif market_state == "HEALTHY":
            return (0.40, 0.30, 0.30)
        elif market_state == "WEAK":
            return (0.25, 0.25, 0.50)
        else:  # UNHEALTHY
            return (0.15, 0.20, 0.65)

    def calculate_momentum_persistence(self, asset, price_data):
        """Stricter momentum persistence filter"""
        returns = price_data[asset].pct_change()
        positive_periods = (returns > 0).rolling(36).mean().iloc[-1]
        return positive_periods

    def calculate_alpha_score(self, asset, price_data, sentiment_scores, market_state):
        """Adaptive alpha calculation"""
        try:
            momentum_w, sentiment_w, meanrev_w = self.get_adaptive_alpha_weights(
                market_state
            )

            # Momentum with longer timeframes
            returns_36h = price_data[asset].pct_change(36).iloc[-1]
            returns_96h = price_data[asset].pct_change(96).iloc[-1]
            momentum_alpha = 0.6 * returns_36h + 0.4 * returns_96h

            # Sentiment
            sentiment_alpha = sentiment_scores.get(asset, 0.0)

            # Mean Reversion with longer MA
            current_price = price_data[asset].iloc[-1]
            ma_36h = price_data[asset].rolling(36).mean().iloc[-1]
            mean_reversion_alpha = (
                (ma_36h - current_price) / current_price if current_price > 0 else 0
            )

            composite_alpha = (
                momentum_w * momentum_alpha
                + sentiment_w * sentiment_alpha
                + meanrev_w * mean_reversion_alpha
            )

            # Stricter momentum persistence filter
            momentum_persistence = self.calculate_momentum_persistence(
                asset, price_data
            )
            if momentum_persistence < 0.65:
                composite_alpha *= 0.2

            return composite_alpha

        except Exception as e:
            logger.debug(f"Alpha calc failed for {asset}: {e}")
            return 0.0

    def calculate_dynamic_lambda(
        self, returns_data, correlation_matrix, regime, market_state
    ):
        """More conservative dynamic lambda"""
        regime_multipliers = {
            "HIGH_VOLATILITY": 2.0,
            "NORMAL": 1.3,
            "LOW_VOLATILITY": 1.0,
        }

        health_multipliers = {
            "VERY_HEALTHY": 0.9,
            "HEALTHY": 1.0,
            "WEAK": 1.6,
            "UNHEALTHY": 2.2,
        }

        base_multiplier = regime_multipliers.get(regime, 1.0)
        health_multiplier = health_multipliers.get(market_state, 1.0)

        dynamic_lambda = self.base_lambda_risk * base_multiplier * health_multiplier
        dynamic_lambda = max(0.2, min(3.5, dynamic_lambda))

        logger.info(f"🎚️ Dynamic Lambda: {dynamic_lambda:.3f}")
        return dynamic_lambda

    def build_qubo_hamiltonian(
        self, assets, alpha_scores, correlation_matrix, n_target, lambda_risk
    ):
        x = Array.create("x", shape=len(assets), vartype="BINARY")

        alpha_term = sum(-alpha_scores[i] * x[i] for i in range(len(assets)))

        risk_term = 0
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                risk_term += correlation_matrix.iloc[i, j] * x[i] * x[j]

        P = 2.0
        constraint_term = P * (sum(x) - n_target) ** 2

        H = alpha_term + lambda_risk * risk_term + constraint_term
        return H, x

    def select_optimal_portfolio(self, price_data, sentiment_scores):
        """Complete adaptive portfolio selection"""
        returns_data = price_data.pct_change().dropna()

        if returns_data.empty:
            return ["BTC-USD", "ETH-USD"], "UNKNOWN", 5, 0.5, 14400

        # Market Health Assessment
        market_state, optimal_cycle_seconds = self.health_detector.assess_market_health(
            returns_data
        )

        # Market Regime Detection
        n_assets, regime = self.regime_detector.get_optimal_n(returns_data)

        # Calculate alpha scores
        alpha_scores = {}
        for asset in price_data.columns:
            alpha_scores[asset] = self.calculate_alpha_score(
                asset, price_data, sentiment_scores, market_state
            )

        # Correlation matrix
        correlation_matrix = returns_data.corr()

        # Dynamic Lambda
        dynamic_lambda = self.calculate_dynamic_lambda(
            returns_data, correlation_matrix, regime, market_state
        )

        # QUBO Optimization
        assets = price_data.columns.tolist()
        alpha_values = [alpha_scores[asset] for asset in assets]

        H, x = self.build_qubo_hamiltonian(
            assets, alpha_values, correlation_matrix, n_assets, dynamic_lambda
        )

        try:
            model = H.compile()
            bqm = model.to_bqm()
            
            # Use dimod's simulated annealer
            sampler = dimod.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=100)
            
            solution = sampleset.first.sample
            selected_assets = [
                assets[i] for i in range(len(assets)) if solution.get(f"x[{i}]", 0) == 1
            ]

            logger.info(
                f"✅ ADAPTIVE: {len(selected_assets)} assets, λ={dynamic_lambda:.3f}"
            )
            return (
                selected_assets,
                regime,
                n_assets,
                dynamic_lambda,
                optimal_cycle_seconds,
            )

        except Exception as e:
            logger.error(f"QUBO failed: {e}")
            # Conservative fallback
            sorted_assets = sorted(
                alpha_scores.items(), key=lambda x: x[1], reverse=True
            )
            selected_assets = [asset for asset, score in sorted_assets[:n_assets]]
            return (
                selected_assets,
                regime,
                n_assets,
                dynamic_lambda,
                optimal_cycle_seconds,
            )


def get_target_assets(price_data, sentiment_scores, base_lambda_risk=0.7):
    optimizer = AdaptiveQUBOOptimizer(base_lambda_risk=base_lambda_risk)
    return optimizer.select_optimal_portfolio(price_data, sentiment_scores)