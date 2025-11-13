# performance_logger.py - COMPETITION OPTIMIZED
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


class PerformanceLogger:
    def __init__(self):
        self.portfolio_history = []
        self.trade_log = []
        self.regime_history = []
        self.competition_metrics = {
            "portfolio_values": [],
            "timestamps": [],
            "hourly_returns": [],
            "daily_returns": [],
        }

        # Competition-specific tracking
        self.competition_start_time = datetime.now()
        self.best_portfolio_value = 0
        self.max_drawdown = 0
        self.volatility = 0

    def log_rebalance(
        self, timestamp, selected_assets, weights, regime, n_assets, lambda_risk
    ):
        """Log rebalance decision with competition context"""
        log_entry = {
            "timestamp": timestamp,
            "regime": regime,
            "n_assets": n_assets,
            "lambda_risk": lambda_risk,
            "selected_assets": selected_assets,
            "weights": weights,
            "portfolio_size": len(selected_assets),
        }

        self.regime_history.append(log_entry)
        logger.info(f"📊 Rebalance logged: {regime}, n={n_assets}, λ={lambda_risk:.3f}")

    def log_trade(self, asset, action, quantity, price, success, error_msg=None):
        """Log individual trade with enhanced competition tracking"""
        trade_entry = {
            "timestamp": datetime.now(),
            "asset": asset,
            "action": action,
            "quantity": quantity,
            "price": price,
            "success": success,
            "error_msg": error_msg,
        }

        self.trade_log.append(trade_entry)

        status = "✅" if success else "❌"
        logger.info(f"{status} Trade: {action} {quantity:.6f} {asset} @ ${price:.2f}")

    def log_portfolio_value(self, portfolio_value):
        """Log portfolio value for competition metrics calculation"""
        current_time = datetime.now()
        self.competition_metrics["portfolio_values"].append(portfolio_value)
        self.competition_metrics["timestamps"].append(current_time)

        # Update best portfolio value
        if portfolio_value > self.best_portfolio_value:
            self.best_portfolio_value = portfolio_value

        # Calculate current drawdown
        if self.best_portfolio_value > 0:
            current_drawdown = (
                portfolio_value - self.best_portfolio_value
            ) / self.best_portfolio_value
            self.max_drawdown = min(self.max_drawdown, current_drawdown)

    def calculate_competition_metrics(self, risk_free_rate=0.02):
        """Calculate competition-specific metrics for awards"""
        if len(self.competition_metrics["portfolio_values"]) < 10:
            return self.get_empty_metrics()

        portfolio_values = np.array(self.competition_metrics["portfolio_values"])

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        hourly_returns = pd.Series(returns)

        if len(hourly_returns) < 2:
            return self.get_empty_metrics()

        # Annualization factors
        hours_per_year = 24 * 365
        days_per_year = 365

        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[
            0
        ]
        annualized_return = total_return * (
            days_per_year / max(1, (datetime.now() - self.competition_start_time).days)
        )

        # Volatility (annualized)
        hourly_volatility = hourly_returns.std()
        annualized_volatility = hourly_volatility * np.sqrt(hours_per_year)

        # Sharpe Ratio
        excess_returns = hourly_returns - risk_free_rate / hours_per_year
        sharpe_ratio = (
            excess_returns.mean() / hourly_returns.std() * np.sqrt(hours_per_year)
            if hourly_returns.std() > 0
            else 0
        )

        # Sortino Ratio (downside risk only)
        downside_returns = hourly_returns[hourly_returns < 0]
        downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (
            (hourly_returns.mean() * hours_per_year - risk_free_rate)
            / downside_volatility
            * np.sqrt(hours_per_year)
            if downside_volatility > 0
            else 0
        )

        # Calmar Ratio
        cumulative_returns = (1 + hourly_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win Rate and Profit Factor
        winning_trades = [
            t
            for t in self.trade_log
            if t.get("success", False) and t.get("action") == "SELL"
        ]
        total_trades = len([t for t in self.trade_log if t.get("success", False)])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # Competition Score (weighted as per criteria)
        competition_score = (
            0.4 * (sortino_ratio if not np.isnan(sortino_ratio) else 0)
            + 0.3 * (sharpe_ratio if not np.isnan(sharpe_ratio) else 0)
            + 0.3 * (calmar_ratio if not np.isnan(calmar_ratio) else 0)
        )

        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "max_drawdown": float(max_drawdown),
            "volatility": float(annualized_volatility),
            "win_rate": float(win_rate),
            "total_trades": total_trades,
            "successful_trades": len(winning_trades),
            "competition_score": float(competition_score),
            "current_portfolio_value": float(
                portfolio_values[-1] if len(portfolio_values) > 0 else 0
            ),
            "best_portfolio_value": float(self.best_portfolio_value),
            "hours_elapsed": (
                datetime.now() - self.competition_start_time
            ).total_seconds()
            / 3600,
        }

        return metrics

    def get_empty_metrics(self):
        """Return empty metrics when insufficient data"""
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_return": 0,
            "annualized_return": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0,
            "max_drawdown": 0,
            "volatility": 0,
            "win_rate": 0,
            "total_trades": 0,
            "successful_trades": 0,
            "competition_score": 0,
            "current_portfolio_value": 0,
            "best_portfolio_value": 0,
            "hours_elapsed": 0,
        }

    def generate_performance_report(self):
        """Generate comprehensive performance report for dashboard and competition"""
        # Calculate current metrics
        metrics = self.calculate_competition_metrics()

        # Get regime distribution
        regime_distribution = {}
        if self.regime_history:
            regime_df = pd.DataFrame(self.regime_history)
            regime_distribution = regime_df["regime"].value_counts().to_dict()

        # Get recent trades
        recent_trades = self.get_recent_trades(15)

        # Calculate additional statistics
        total_rebalances = len(self.regime_history)
        unique_assets = len(
            set(
                [
                    asset
                    for entry in self.regime_history
                    for asset in entry.get("selected_assets", [])
                ]
            )
        )

        # Trading frequency analysis
        if self.trade_log:
            trade_times = [t["timestamp"] for t in self.trade_log if "timestamp" in t]
            if len(trade_times) > 1:
                trade_intervals = [
                    (trade_times[i + 1] - trade_times[i]).total_seconds() / 3600
                    for i in range(len(trade_times) - 1)
                ]
                avg_trade_interval = np.mean(trade_intervals) if trade_intervals else 0
            else:
                avg_trade_interval = 0
        else:
            avg_trade_interval = 0

        report = {
            "summary_metrics": metrics,
            "regime_distribution": regime_distribution,
            "recent_trades": recent_trades,
            "strategy_metrics": {
                "total_rebalances": total_rebalances,
                "unique_assets_traded": unique_assets,
                "avg_trade_interval_hours": float(avg_trade_interval),
                "current_regime": (
                    self.regime_history[-1].get("regime", "UNKNOWN")
                    if self.regime_history
                    else "UNKNOWN"
                ),
                "current_lambda": (
                    self.regime_history[-1].get("lambda_risk", 0)
                    if self.regime_history
                    else 0
                ),
                "avg_portfolio_size": (
                    np.mean(
                        [
                            entry.get("portfolio_size", 0)
                            for entry in self.regime_history
                        ]
                    )
                    if self.regime_history
                    else 0
                ),
            },
            "performance_history": self.get_performance_history(),
            "competition_ranking": self.calculate_competition_ranking(metrics),
        }

        return report

    def calculate_competition_ranking(self, metrics):
        """Estimate competition ranking based on current metrics"""
        score = metrics["competition_score"]

        # Ranking thresholds (estimated)
        if score > 2.0:
            rank = "Top 3 - Award Contender"
        elif score > 1.0:
            rank = "Top 10 - Strong Performer"
        elif score > 0.5:
            rank = "Top 25 - Competitive"
        elif score > 0:
            rank = "Middle Tier"
        else:
            rank = "Needs Improvement"

        return {
            "estimated_rank": rank,
            "competition_score": score,
            "award_potential": self.assess_award_potential(metrics),
        }

    def assess_award_potential(self, metrics):
        """Assess potential for specific awards"""
        awards = []

        # 2nd Award – Risk-adjusted Return potential
        risk_adjusted_score = (
            0.4 * metrics["sortino_ratio"]
            + 0.3 * metrics["sharpe_ratio"]
            + 0.3 * metrics["calmar_ratio"]
        )

        if risk_adjusted_score > 1.5:
            awards.append("🥈 Strong Risk-Adjusted Return Contender")
        elif risk_adjusted_score > 0.8:
            awards.append("🥈 Potential Risk-Adjusted Return")

        # 3rd Award – Best Strategy/Technique potential
        strategy_score = (
            metrics["win_rate"] * 0.3
            + (1 - abs(metrics["max_drawdown"])) * 0.3
            + min(metrics["total_return"] * 10, 1) * 0.4
        )

        if strategy_score > 0.7:
            awards.append("🥉 Strong Strategy/Technique Contender")
        elif strategy_score > 0.5:
            awards.append("🥉 Potential Strategy/Technique")

        return awards if awards else ["Focus on consistency and risk management"]

    def get_recent_trades(self, n=10):
        """Get recent trades for reporting"""
        if not self.trade_log:
            return []

        try:
            recent_trades = (
                self.trade_log[-n:] if len(self.trade_log) > n else self.trade_log
            )
            formatted_trades = []

            for trade in recent_trades:
                formatted_trades.append(
                    {
                        "asset": trade.get("asset", "UNKNOWN"),
                        "action": trade.get("action", "UNKNOWN"),
                        "quantity": float(trade.get("quantity", 0)),
                        "price": float(trade.get("price", 0)),
                        "success": trade.get("success", False),
                        "timestamp": (
                            trade.get("timestamp", datetime.now()).strftime(
                                "%m/%d %H:%M"
                            )
                            if hasattr(trade.get("timestamp"), "strftime")
                            else str(trade.get("timestamp", ""))
                        ),
                    }
                )

            return formatted_trades
        except Exception as e:
            logger.error(f"Error formatting recent trades: {e}")
            return []

    def get_performance_history(self):
        """Get historical performance data for charts"""
        if not self.regime_history:
            return []

        return [
            {
                "timestamp": entry.get("timestamp", datetime.now()).strftime(
                    "%m/%d %H:%M"
                ),
                "portfolio_size": entry.get("portfolio_size", 0),
                "lambda_risk": float(entry.get("lambda_risk", 0)),
                "regime": entry.get("regime", "UNKNOWN"),
            }
            for entry in self.regime_history
        ]

    def save_logs(self):
        """Save comprehensive logs to files for competition submission"""
        try:
            # Save regime history
            if self.regime_history:
                regime_df = pd.DataFrame(self.regime_history)
                regime_df.to_csv("competition_regime_history.csv", index=False)

            # Save trade log
            if self.trade_log:
                trade_df = pd.DataFrame(self.trade_log)
                trade_df.to_csv("competition_trade_log.csv", index=False)

            # Save performance metrics
            metrics = self.calculate_competition_metrics()
            with open("competition_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            # Generate competition report
            report = self.generate_performance_report()
            with open("competition_final_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info("📁 Competition logs saved successfully")

        except Exception as e:
            logger.error(f"Error saving competition logs: {e}")

    def print_competition_update(self):
        """Print competition performance update"""
        metrics = self.calculate_competition_metrics()

        print("\n" + "=" * 70)
        print("🏆 COMPETITION PERFORMANCE UPDATE")
        print("=" * 70)
        print(f"📈 Portfolio Value: ${metrics['current_portfolio_value']:,.2f}")
        print(f"📊 Total Return: {metrics['total_return']:+.2%}")
        print(f"⏰ Hours Elapsed: {metrics['hours_elapsed']:.1f}")
        print("\n🎯 AWARD METRICS:")
        print(f"   • Sortino Ratio: {metrics['sortino_ratio']:.3f} (40% weight)")
        print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.3f} (30% weight)")
        print(f"   • Calmar Ratio: {metrics['calmar_ratio']:.3f} (30% weight)")
        print(f"   🏅 Competition Score: {metrics['competition_score']:.3f}")
        print(f"   📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   🎯 Win Rate: {metrics['win_rate']:.1%}")
        print(f"   📊 Volatility: {metrics['volatility']:.2%}")

        ranking = self.calculate_competition_ranking(metrics)
        print(f"\n🏅 ESTIMATED RANKING: {ranking['estimated_rank']}")
        print("🎖️ AWARD POTENTIAL:")
        for award in ranking["award_potential"]:
            print(f"   • {award}")
        print("=" * 70)

    def get_dashboard_data(self):
        """Get data formatted for dashboard"""
        report = self.generate_performance_report()

        return {
            "summary": {
                "totalRebalances": report["strategy_metrics"]["total_rebalances"],
                "currentRegime": report["strategy_metrics"]["current_regime"],
                "avgPortfolioSize": report["strategy_metrics"]["avg_portfolio_size"],
                "currentLambda": report["strategy_metrics"]["current_lambda"],
                "lastUpdate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "competitionScore": report["summary_metrics"]["competition_score"],
                "totalReturn": report["summary_metrics"]["total_return"],
            },
            "regimeDistribution": report["regime_distribution"],
            "recentTrades": report["recent_trades"],
            "performanceHistory": report["performance_history"],
            "competitionMetrics": report["summary_metrics"],
        }


# For backward compatibility
def generate_performance_report(self):
    return self.generate_performance_report()


def get_recent_trades(self, n=10):
    return self.get_recent_trades(n)


def get_empty_report(self):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rebalances": 0,
        "regime_distribution": {"NO_DATA": 1},
        "avg_portfolio_size": 0,
        "unique_assets_traded": 0,
        "latest_lambda": 0,
        "current_regime": "NO_DATA",
        "recent_trades": [],
    }