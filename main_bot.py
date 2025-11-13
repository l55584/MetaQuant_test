# main_bot.py - COMPETITION OPTIMIZED (2-WEEK TIMEFRAME)
import time
import logging
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Import your modules
import data_fetcher
import qubo_optimizer
import allocator
import bot_executor
import performance_logger
import dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CompetitionQUBOBot:
    def __init__(self, lambda_risk=0.5):
        """Initialize competition-optimized trading bot for 2-week timeframe"""
        self.lambda_risk = lambda_risk
        self.performance_logger = performance_logger.PerformanceLogger()
        self.iteration_count = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.last_successful_data = None
        self.cache_duration = 300
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.dashboard_thread = None
        self.current_optimal_cycle = 10800  # Start with 3 hours (more conservative)
        self.market_state = "UNKNOWN"
        self.portfolio_value_history = []
        self.circuit_breaker_triggered = False
        self.competition_start_time = datetime.now()
        self.competition_days_remaining = 14

        # UPDATED COMPETITION HYPERPARAMETERS - MATCH bot_executor.py
        self.hyperparameters = {
            "rebalance_interval": 6 * 3600,  # 6 hours base (more conservative)
            "trade_threshold": 0.12,  # 12% threshold (matches bot_executor)
            "min_hold_hours": 8,  # 8 hours between trades (matches wash controller)
            "min_profit_threshold": 0.008,  # 0.8% net profit (matches wash controller)
            "max_daily_trades": 3,  # Maximum 3 trades per day (matches wash controller)
            "emergency_stop_loss": -0.02,  # Stop trading if 2% loss in 2 cycles
            "final_week_conservative": False,
            "min_trade_value": 100.0,  # Minimum $100 trade size
        }

        logger.info("🏆 COMPETITION QUBO Bot Initialized (2-Week Timeline)")
        logger.info(f"⏰ Competition ends in: {self.competition_days_remaining} days")
        logger.info(
            f"🎯 Aggressive Mode: {self.hyperparameters['min_hold_hours']}h min hold"
        )

    def update_competition_parameters(self):
        """Dynamically adjust parameters based on competition progress"""
        days_elapsed = (datetime.now() - self.competition_start_time).days
        days_remaining = 14 - days_elapsed

        # Become more conservative in final 5 days to protect gains
        if days_remaining <= 5 and not self.hyperparameters["final_week_conservative"]:
            logger.info("🛡️ Entering final week conservative mode")
            self.hyperparameters.update(
                {
                    "trade_threshold": 0.15,  # Higher threshold
                    "min_hold_hours": 12,  # Longer holds
                    "min_profit_threshold": 0.012,  # Higher profit requirement
                    "max_daily_trades": 2,
                    "final_week_conservative": True,
                }
            )

        self.competition_days_remaining = days_remaining

    def emergency_circuit_breaker(self):
        """Enhanced circuit breaker with daily trade limits"""
        if self.circuit_breaker_triggered:
            return True

        if (
            not hasattr(self.performance_logger, "trade_log")
            or not self.performance_logger.trade_log
        ):
            return False

        # Check daily trade limits
        today_trades = [
            t
            for t in self.performance_logger.trade_log
            if t.get("timestamp", datetime.now()).date() == datetime.now().date()
            and t.get("success", False)
        ]

        if len(today_trades) >= self.hyperparameters["max_daily_trades"]:
            logger.error(f"🚨 DAILY LIMIT: {len(today_trades)} trades today")
            self.circuit_breaker_triggered = True
            return True

        # Check for rapid flipping on same assets
        recent_trades = self.performance_logger.trade_log[-10:]
        asset_activity = {}
        for trade in recent_trades:
            asset = trade["asset"]
            if asset not in asset_activity:
                asset_activity[asset] = []
            asset_activity[asset].append(
                {
                    "action": trade["action"],
                    "timestamp": trade.get("timestamp", datetime.now()),
                }
            )

        # Enhanced pattern detection
        for asset, trades in asset_activity.items():
            if len(trades) >= 3:
                buy_count = sum(1 for t in trades if t["action"] == "BUY")
                sell_count = sum(1 for t in trades if t["action"] == "SELL")

                # Detect wash patterns: multiple round-trips in short time
                if buy_count >= 1 and sell_count >= 1:
                    time_span = max(t["timestamp"] for t in trades) - min(
                        t["timestamp"] for t in trades
                    )
                    if time_span.total_seconds() < 28800:  # Within 8 hours
                        logger.error(
                            f"🚨 CIRCUIT BREAKER: Excessive flipping on {asset}"
                        )
                        self.circuit_breaker_triggered = True
                        return True
        return False

    def should_skip_rebalance(self, current_holdings, price_data):
        """Competition-optimized emergency checks"""
        if self.emergency_circuit_breaker():
            logger.error("🚨 CIRCUIT BREAKER ACTIVE - Skipping rebalance")
            return True

        if len(self.portfolio_value_history) < 3:
            return False

        # If we've lost 3% in last 2 cycles, stop trading temporarily
        recent_loss = (
            self.portfolio_value_history[-1] - self.portfolio_value_history[-3]
        ) / self.portfolio_value_history[-3]

        if recent_loss < self.hyperparameters["emergency_stop_loss"]:
            logger.warning(
                f"🚨 EMERGENCY: {recent_loss:.2%} loss - cooling down for 2 cycles"
            )
            return True

        return False

    def fetch_data_parallel(self):
        """Competition-optimized data fetching with faster timeouts"""
        try:
            # Use cached data if recent
            if (
                self.last_successful_data
                and (datetime.now() - self.last_successful_data["timestamp"]).seconds
                < self.cache_duration
            ):
                return (
                    self.last_successful_data["price_data"],
                    self.last_successful_data["sentiment_scores"],
                )

            # Faster parallel data fetching for competition
            with ThreadPoolExecutor(max_workers=2) as executor:
                market_future = executor.submit(
                    data_fetcher.get_all_market_data, "1h", 80  # Add the interval parameter
                )
                sentiment_future = executor.submit(
                    data_fetcher.get_sentiment_score
                )

                # Faster timeouts for competition responsiveness
                price_data, successful_assets = market_future.result(timeout=20)
                sentiment_scores = sentiment_future.result(timeout=10)

            # Validate data
            if price_data is not None and not price_data.empty and len(price_data) > 10:
                # Cache successful data
                self.last_successful_data = {
                    "price_data": price_data,
                    "sentiment_scores": sentiment_scores,
                    "timestamp": datetime.now(),
                }
                return price_data, sentiment_scores
            else:
                logger.warning("⚠️ Insufficient price data received")
                if self.last_successful_data:
                    return (
                        self.last_successful_data["price_data"],
                        self.last_successful_data["sentiment_scores"],
                    )
                return None, None

        except Exception as e:
            logger.error(f"❌ Data fetch failed: {e}")
            if self.last_successful_data:
                return (
                    self.last_successful_data["price_data"],
                    self.last_successful_data["sentiment_scores"],
                )
            return None, None

    def calculate_portfolio_value(self, holdings, price_data):
        """Calculate current portfolio value with detailed logging"""
        try:
            total_value = 0
            logger.info("📊 Portfolio Value Breakdown:")
            for asset, quantity in holdings.items():
                if asset in price_data.columns:
                    current_price = price_data[asset].iloc[-1]
                    asset_value = quantity * current_price
                    total_value += asset_value
                    logger.info(f"   {asset}: {quantity} × ${current_price:.2f} = ${asset_value:.2f}")
                else:
                    logger.warning(f"   {asset}: No price data available")
        
            logger.info(f"   Total Portfolio Value: ${total_value:.2f}")
            return total_value
        except Exception as e:
            logger.error(f"Portfolio value calculation error: {e}")
            return 0

    def get_portfolio_with_retry(self, max_retries=2):  # Faster retries for competition
        """Get portfolio with faster retry logic"""
        for attempt in range(max_retries):
            try:
                current_holdings, cash_balance = bot_executor.get_current_portfolio()
                if cash_balance is not None:
                    return current_holdings, cash_balance
            except Exception as e:
                logger.warning(f"⚠️ Portfolio fetch attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Shorter sleep

        logger.error("❌ Portfolio fetch failed")
        return {}, 0.0

    def log_trade_results(self, rebalance_result):
        """Log trade results with competition context"""
        try:
            if rebalance_result.get("sell_orders"):
                for order in rebalance_result["sell_orders"]:
                    self.performance_logger.log_trade(
                        asset=order.get("asset", "UNKNOWN"),
                        action="SELL",
                        quantity=order.get("quantity", 0),
                        price=order.get("price", 0),
                        success=order.get("success", False),
                        error_msg=order.get("error", None),
                    )

            if rebalance_result.get("buy_orders"):
                for order in rebalance_result["buy_orders"]:
                    self.performance_logger.log_trade(
                        asset=order.get("asset", "UNKNOWN"),
                        action="BUY", 
                        quantity=order.get("quantity", 0),
                        price=order.get("price", 0),
                        success=order.get("success", False),
                        error_msg=order.get("error", None),
                    )
        except Exception as e:
            logger.error(f"Error logging trade results: {e}")

    def run_trading_cycle(self):
        """Competition-optimized trading cycle"""
        # Update competition parameters
        self.update_competition_parameters()

        logger.info(
            f"🔄 Trading Cycle {self.iteration_count + 1} | Day {14 - self.competition_days_remaining + 1}/14"
        )

        cycle_start = time.time()

        try:
            # 1. Fetch data in parallel
            logger.info("📥 Fetching competition data...")
            price_data, sentiment_scores = self.fetch_data_parallel()

            if price_data is None or price_data.empty:
                logger.error("❌ No market data available")
                self.consecutive_failures += 1
                return False, 10800  # Return 3-hour default

            # 2. Quick data validation
            if len(price_data) < 20:
                logger.warning("⚠️ Insufficient data points, skipping cycle")
                self.consecutive_failures += 1
                return False, 10800

            # 4. Get current portfolio BEFORE skip checks
            current_holdings, cash_balance = self.get_portfolio_with_retry()

# 5. Emergency skip checks (now current_holdings is defined)
            if self.should_skip_rebalance(current_holdings, price_data):
                logger.warning("🚨 Emergency skip triggered")
                return False, self.current_optimal_cycle

            # 5. ADAPTIVE QUBO Asset Selection
            logger.info("🎯 Running competition QUBO optimization...")

            try:
                # Try adaptive version (5 return values)
                (
                    selected_assets,
                    regime,
                    n_assets,
                    dynamic_lambda,
                    optimal_cycle_seconds,
                ) = qubo_optimizer.get_target_assets(
                    price_data, sentiment_scores, self.lambda_risk
                )

                # Store adaptive cycle time
                self.current_optimal_cycle = optimal_cycle_seconds
                self.market_state = regime
                logger.info(f"📊 Competition cycle: {optimal_cycle_seconds/3600:.1f}h")

            except ValueError as e:
                if "too many values to unpack" in str(e):
                    # Legacy version
                    selected_assets, regime, n_assets, dynamic_lambda = (
                        qubo_optimizer.get_target_assets(
                            price_data, sentiment_scores, self.lambda_risk
                        )
                    )
                    self.current_optimal_cycle = 7200  # 2 hours default
                    self.market_state = regime
                else:
                    raise e
            except Exception as e:
                logger.error(f"❌ QUBO optimization failed: {e}")
                # Competition fallback - be more aggressive
                returns = price_data.pct_change().iloc[-24:].mean()  # Shorter lookback
                selected_assets = returns.nlargest(6).index.tolist()  # More assets
                regime = "CONSERVATIVE_FALLBACK"
                n_assets = len(selected_assets)
                dynamic_lambda = self.lambda_risk * 1.2  # Lower risk aversion
                self.current_optimal_cycle = 10800

            if not selected_assets:
                selected_assets = price_data.columns[
                    :4
                ].tolist()  # More fallback assets
                regime = "ULTRA_CONSERVATIVE"

            # Update lambda for next cycle
            self.lambda_risk = dynamic_lambda

            logger.info(f"✅ Selected {len(selected_assets)} assets for competition")

            # 6. Portfolio Allocation
            logger.info("📊 Calculating competition weights...")
            try:
                portfolio_weights = allocator.get_target_weights(
                    selected_assets, price_data
                )
            except Exception as e:
                logger.error(f"❌ Portfolio allocation failed: {e}")
                # Equal weight but more concentrated
                portfolio_weights = {
                    asset: 1.0 / len(selected_assets) for asset in selected_assets
                }

            # 7. Get current portfolio
            #current_holdings, cash_balance = self.get_portfolio_with_retry()

            # Track portfolio value
            current_portfolio_value = self.calculate_portfolio_value(
                current_holdings, price_data
            )
            total_value = current_portfolio_value + cash_balance
            self.portfolio_value_history.append(total_value)

            self.performance_logger.log_portfolio_value(total_value)

            if cash_balance == 0 and not current_holdings:
                logger.error("❌ Cannot get portfolio balance")
                self.consecutive_failures += 1
                return False, self.current_optimal_cycle

            logger.info(f"💼 Portfolio: ${total_value:,.2f}")

            # 8. Execute competition rebalance
            rebalance_result = bot_executor.execute_rebalance(
                portfolio_weights,
                current_holdings,
                cash_balance,
                threshold=self.hyperparameters["trade_threshold"],
            )

            # 9. Log performance
            self.performance_logger.log_rebalance(
                timestamp=datetime.now(),
                selected_assets=selected_assets,
                weights=portfolio_weights,
                regime=f"{regime}",
                n_assets=n_assets,
                lambda_risk=dynamic_lambda,
            )

            # Log trade results
            self.log_trade_results(rebalance_result)

            cycle_time = time.time() - cycle_start
            self.iteration_count += 1

            # Calculate performance
            performance_info = self.calculate_performance()

            logger.info(
                f"✅ Cycle {self.iteration_count} completed in {cycle_time:.1f}s"
            )
            logger.info(f"📈 {performance_info}")
            logger.info(
                f"🌡️ Market: {self.market_state}, Next: {self.current_optimal_cycle/3600:.1f}h"
            )

            self.consecutive_failures = 0
            return True, self.current_optimal_cycle

        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}")
            self.consecutive_failures += 1
            return False, self.current_optimal_cycle

    def start_dashboard(self):
        """Start competition dashboard"""
        try:
            logger.info("🚀 Starting competition dashboard...")
            dashboard.start_dashboard(
                self.performance_logger, host="0.0.0.0", port=8050, debug=False
            )
        except Exception as e:
            logger.error(f"Dashboard error: {e}")

    def analyze_competition_performance(self):
        """Competition-specific performance analysis"""
        if self.iteration_count % 5 == 0:  # More frequent analysis
            logger.info("🏆 COMPETITION PERFORMANCE UPDATE")

            total_cycles = self.iteration_count
            success_rate = (
                (total_cycles - self.consecutive_failures) / total_cycles * 100
                if total_cycles > 0
                else 0
            )

            # Calculate key competition metrics
            if self.portfolio_value_history:
                initial_value = self.portfolio_value_history[0]
                current_value = self.portfolio_value_history[-1]
                total_return_pct = (
                    (current_value - initial_value) / initial_value
                ) * 100

                logger.info(
                    f"📊 Day {14 - self.competition_days_remaining + 1}/14 Performance:"
                )
                logger.info(f"   • Total Return: {total_return_pct:+.2f}%")
                logger.info(
                    f"   • Cycles: {total_cycles} (Success: {success_rate:.1f}%)"
                )
                logger.info(f"   • Current Strategy: {self.market_state}")
                logger.info(f"   • Risk Level: {self.lambda_risk:.3f}")

            # Auto-save logs more frequently
            self.performance_logger.save_logs()

    def run_competition(self):
        """Main competition loop"""
        logger.info("🏁 STARTING 2-WEEK TRADING COMPETITION")
        logger.info("⏰ Competition ends in 14 days")
        logger.info("🎯 Strategy: Aggressive first 11 days, Conservative final 3 days")

        # Start dashboard
        self.dashboard_thread = threading.Thread(
            target=self.start_dashboard, daemon=True
        )
        self.dashboard_thread.start()
        logger.info("📊 Competition dashboard: http://localhost:8050")

        while self.competition_days_remaining > 0:
            try:
                cycle_start = time.time()

                # Run trading cycle
                success, optimal_cycle = self.run_trading_cycle()

                if success:
                    self.consecutive_failures = 0
                    self.analyze_competition_performance()

                    # Adaptive sleep with competition context
                    cycle_time = time.time() - cycle_start
                    sleep_time = max(
                        optimal_cycle - cycle_time, 45
                    )  # Minimum 45 seconds

                    logger.info(
                        f"💤 Next cycle in {sleep_time/60:.1f}m | {self.competition_days_remaining} days left"
                    )
                    time.sleep(sleep_time)

                else:
                    self.consecutive_failures += 1
                    cycle_time = time.time() - cycle_start
                    sleep_time = max(optimal_cycle - cycle_time, 45)

                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error(f"🚨 Multiple failures - extended cooldown")
                        sleep_time = min(sleep_time * 3, 7200)  # Max 2 hours

                    logger.warning(f"🔄 Retrying in {sleep_time/60:.1f}m...")
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("🛑 Competition bot stopped by user")
                break
            except Exception as e:
                logger.error(f"💥 Competition error: {e}")
                self.consecutive_failures += 1
                sleep_time = min(self.current_optimal_cycle, 3600)
                time.sleep(sleep_time)

        logger.info("🏁 COMPETITION COMPLETED - Final performance report:")
        self.analyze_competition_performance()
    
    def calculate_performance(self):
        """Calculate competition performance metrics"""
        if len(self.portfolio_value_history) < 2:
            return "First cycle"

        current_value = self.portfolio_value_history[-1]
        initial_value = self.portfolio_value_history[0]

        if initial_value > 0:
            total_return = ((current_value - initial_value) / initial_value) * 100
            days_elapsed = (datetime.now() - self.competition_start_time).days + 1
            annualized_return = (
                total_return * (365 / days_elapsed) if days_elapsed > 0 else 0
            )

            return (
                f"Total: {total_return:+.2f}% | Annualized: {annualized_return:+.1f}%"
            )
        else:
            return "N/A"


def test_competition_bot():
    """Test competition bot with one cycle"""
    logger.info("🧪 Testing competition bot...")
    bot = CompetitionQUBOBot(lambda_risk=0.5)
    success, cycle_time = bot.run_trading_cycle()
    logger.info(f"🧪 Test completed: {success}, Next: {cycle_time/3600:.1f}h")
    return success


if __name__ == "__main__":
    # Start competition bot
    bot = CompetitionQUBOBot(lambda_risk=0.5)

    try:
        bot.run_competition()
    except KeyboardInterrupt:
        logger.info("🛑 Competition terminated")
    except Exception as e:
        logger.error(f"💥 Fatal competition error: {e}")
