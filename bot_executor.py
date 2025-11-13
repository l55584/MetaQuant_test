# bot_executor.py - COMPETITION OPTIMIZED WITH ANTI-WASH CONTROLS
import requests
import time
import hmac
import hashlib
import logging
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta

# Configuration
BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "d6E2kL7wP8bN0tM5gQ1lY3oV4nS6pJ9rA7fT2mC5uI8yB3zK0xW1hN4jX9vH7s"
SECRET_KEY = "aZ3mN8qP4xT7vB1CdF6hJ2K9lM5nR0sUyE7pQ1wX8cV3tG6ZoH4iL9A2bS5uD0rY"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Rate limiting variables
_last_order_time = 0
_MIN_ORDER_INTERVAL = 0.3  # Minimum 0.3 seconds between orders

# COMPETITION ANTI-WASH HYPERPARAMETERS
COMPETITION_CONFIG = {
    "MIN_HOLD_HOURS": 8,  # Minimum 8 hours between buy and sell
    "MIN_NET_PROFIT": 0.008,  # 0.8% minimum net profit after commissions
    "MAX_DAILY_TRADES_PER_ASSET": 1,  # Only 1 round-trip per asset daily
    "MAX_DAILY_TOTAL_TRADES": 3,  # Maximum 3 total trades per day
    "MIN_TRADE_VALUE": 100.0,  # Minimum $100 trade size
    "COMMISSION_RATE": 0.0001,  # 0.01% per trade
    "COOLDOWN_HOURS_AFTER_SELL": 6,  # 6 hours before buying same asset
}


class CompetitionWashController:
    """Enhanced anti-wash controller for competition compliance"""

    def __init__(self):
        self.trade_history = {}
        self.daily_trade_count = 0
        self.last_daily_reset = datetime.now().date()

    def _reset_daily_counts(self):
        """Reset daily counters if date changed"""
        today = datetime.now().date()
        if today != self.last_daily_reset:
            self.daily_trade_count = 0
            self.last_daily_reset = today

    def can_execute_trade(self, asset, action, current_price, quantity):
        """Strict trade validation to prevent wash trading"""
        self._reset_daily_counts()

        # Convert asset name if needed
        base_asset = asset.replace("-USD", "") if "-USD" in asset else asset

        # Check daily total trade limit
        if self.daily_trade_count >= COMPETITION_CONFIG["MAX_DAILY_TOTAL_TRADES"]:
            return (
                False,
                f"Daily trade limit reached ({self.daily_trade_count}/{COMPETITION_CONFIG['MAX_DAILY_TOTAL_TRADES']})",
            )

        # Check trade size
        trade_value = current_price * quantity
        if trade_value < COMPETITION_CONFIG["MIN_TRADE_VALUE"]:
            return (
                False,
                f"Trade size ${trade_value:.2f} below minimum ${COMPETITION_CONFIG['MIN_TRADE_VALUE']}",
            )

        if action == "SELL":
            return self._validate_sell_trade(base_asset, current_price, quantity)
        elif action == "BUY":
            return self._validate_buy_trade(base_asset, current_price, quantity)
        else:
            return False, f"Invalid action: {action}"

    def _validate_sell_trade(self, asset, current_price, quantity):
        """Validate sell trade against wash trading rules"""
        asset_history = self.trade_history.get(asset, [])
        today = datetime.now().date()

    # Allow selling holdings that don't have recent buy history (existing holdings)
        if not any(t["action"] == "BUY" for t in asset_history):
            return True, "Selling existing holding (no recent buy history)"
    # Find corresponding buy trades from today OR existing holdings from previous days
        today_buys = [
            t
            for t in asset_history
            if t["timestamp"].date() == today and t["action"] == "BUY"
        ]

    # If no buys today but we have the asset, allow selling (it's from previous days)
        if not today_buys:
        # Check if we have any history of this asset (means it's from before today)
            if asset_history:
            # Allow selling existing holdings with minimum hold time check
                oldest_hold = min([t["timestamp"] for t in asset_history if t["action"] == "BUY"])
                hold_time = datetime.now() - oldest_hold
                min_hold_seconds = COMPETITION_CONFIG["MIN_HOLD_HOURS"] * 3600
            
                if hold_time.total_seconds() >= min_hold_seconds:
                    return True, "Selling existing holding from previous day"
                else:
                    hours_held = hold_time.total_seconds() / 3600
                    min_hours = COMPETITION_CONFIG["MIN_HOLD_HOURS"]
                    return False, f"Hold time {hours_held:.1f}h < minimum {min_hours}h"
            else:
                return False, "No corresponding BUY trade found"

    # Use the most recent buy
        latest_buy = max(today_buys, key=lambda x: x["timestamp"])

    # Check hold time
        hold_time = datetime.now() - latest_buy["timestamp"]
        min_hold_seconds = COMPETITION_CONFIG["MIN_HOLD_HOURS"] * 3600

        if hold_time.total_seconds() < min_hold_seconds:
            hours_held = hold_time.total_seconds() / 3600
            min_hours = COMPETITION_CONFIG["MIN_HOLD_HOURS"]
            return False, f"Hold time {hours_held:.1f}h < minimum {min_hours}h"

    # Check profitability including commissions
        commission_cost = current_price * COMPETITION_CONFIG["COMMISSION_RATE"] * 2
        buy_price = latest_buy["price"]
        gross_profit = current_price - buy_price
        net_profit = gross_profit - commission_cost
        net_profit_pct = net_profit / buy_price

        min_profit = COMPETITION_CONFIG["MIN_NET_PROFIT"]

        if net_profit_pct < min_profit:
            return False, f"Net profit {net_profit_pct:.3%} < minimum {min_profit:.3%}"

    # Check if we've already traded this asset today
        today_trades = [t for t in asset_history if t["timestamp"].date() == today]
        if len(today_trades) >= COMPETITION_CONFIG["MAX_DAILY_TRADES_PER_ASSET"]:
            return False, f"Daily trade limit for {asset} reached"

        return True, f"OK - Profit: {net_profit_pct:.3%}"

    def _validate_buy_trade(self, asset, current_price, quantity):
        """Validate buy trade against wash trading rules"""
        asset_history = self.trade_history.get(asset, [])
        today = datetime.now().date()

        # Check if we've already traded this asset today
        today_trades = [t for t in asset_history if t["timestamp"].date() == today]
        if len(today_trades) >= COMPETITION_CONFIG["MAX_DAILY_TRADES_PER_ASSET"]:
            return False, f"Daily trade limit for {asset} reached"

        # Check cooldown period after previous sell
        recent_sells = [
            t
            for t in asset_history
            if t["action"] == "SELL"
            and (datetime.now() - t["timestamp"]).total_seconds()
            < COMPETITION_CONFIG["COOLDOWN_HOURS_AFTER_SELL"] * 3600
        ]

        if recent_sells:
            return False, f"Cooldown period active after recent SELL"

        return True, "OK"

    def record_trade(self, asset, action, price, quantity, success=True):
        """Record trade in history"""
        if not success:
            return

        self._reset_daily_counts()
        base_asset = asset.replace("-USD", "") if "-USD" in asset else asset

        if base_asset not in self.trade_history:
            self.trade_history[base_asset] = []

        self.trade_history[base_asset].append(
            {
                "timestamp": datetime.now(),
                "action": action,
                "price": price,
                "quantity": quantity,
                "success": success,
            }
        )

        self.daily_trade_count += 1

        # Clean old history (keep only 3 days)
        cutoff = datetime.now() - timedelta(days=3)
        self.trade_history[base_asset] = [
            t for t in self.trade_history[base_asset] if t["timestamp"] > cutoff
        ]

        logger.info(f"📝 Recorded {action} {quantity} {base_asset} @ ${price:.2f}")

    def get_trade_summary(self):
        """Get current trade statistics"""
        self._reset_daily_counts()
        today = datetime.now().date()

        today_trades = []
        for asset, trades in self.trade_history.items():
            today_trades.extend([t for t in trades if t["timestamp"].date() == today])

        return {
            "daily_trades": len(today_trades),
            "daily_limit": COMPETITION_CONFIG["MAX_DAILY_TOTAL_TRADES"],
            "assets_traded_today": len(set(t["asset"] for t in today_trades)),
        }


def _round_quantity(quantity: float, asset: str) -> float:
    """Round quantity to appropriate precision for the asset"""
    step_sizes = {
        "BTC": 0.0001,  # 6 decimal places
        "ETH": 0.0001,  # 4 decimal places
        "SOL": 0.01,  # 2 decimal places
        "LTC": 0.001,  # 3 decimal places
        "BNB": 0.001,  # 3 decimal places
        "XRP": 0.1,  # 1 decimal place
        "ADA": 1.0,  # 0 decimal places
        "DOT": 0.01,  # 2 decimal places
        "LINK": 0.01,  # 2 decimal places
        "ATOM": 0.01,  # 2 decimal places
        "ETC": 0.001,  # 3 decimal places
        "XLM": 0.1,  # 1 decimal place
        "ALGO": 0.1,  # 1 decimal place
        "UNI": 0.01,  # 2 decimal places
        "AAVE": 0.001,  # 3 decimal places
        "FIL": 0.001,  # 3 decimal places
        "EOS": 0.01,  # 2 decimal places
        "XTZ": 0.01,  # 2 decimal places
    }

    base_asset = asset.replace("-USD", "")
    step_size = step_sizes.get(base_asset, 0.001)

    # Round down to nearest step
    rounded_quantity = (quantity // step_size) * step_size

    # Ensure minimum quantity
    if rounded_quantity < step_size:
        rounded_quantity = step_size

    logger.debug(
        f"Rounded {quantity} {asset} to {rounded_quantity} (step: {step_size})"
    )
    return rounded_quantity


def _get_timestamp():
    """Returns a 13-digit millisecond timestamp as a string."""
    return str(int(time.time() * 1000))


def _generate_signature(payload: Dict) -> Tuple[Dict, Dict, str]:
    """Generate authentication signature and headers"""
    payload["timestamp"] = _get_timestamp()

    # Sort keys and create parameter string
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{key}={payload[key]}" for key in sorted_keys)

    # Create HMAC-SHA256 signature
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"), total_params.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # Create headers
    headers = {"RST-API-KEY": API_KEY, "MSG-SIGNATURE": signature}

    return headers, payload, total_params


def get_current_portfolio() -> Tuple[Dict[str, float], float]:
    """
    Get current portfolio holdings and cash balance
    """
    try:
        logger.info("Fetching current portfolio...")

        # Get authenticated headers
        headers, payload, _ = _generate_signature({})

        # Make API call
        response = requests.get(
            f"{BASE_URL}/v3/balance", headers=headers, params=payload, timeout=10
        )

        if response.status_code != 200:
            logger.error(f"Failed to get portfolio: {response.status_code}")
            return {}, 0.0

        data = response.json()

        if not data.get("Success"):
            error_msg = data.get("ErrMsg", "Unknown error")
            logger.error(f"API Error: {error_msg}")
            return {}, 0.0

        # Parse the portfolio data
        spot_wallet = data.get("SpotWallet", {})
        portfolio_dict = {}
        cash_balance = 0.0

        # Extract cash (USD) balance
        usd_data = spot_wallet.get("USD", {})
        cash_balance = float(usd_data.get("Free", 0)) + float(usd_data.get("Lock", 0))

        # Extract crypto holdings (excluding USD)
        for asset, balance_info in spot_wallet.items():
            if asset != "USD":
                free = float(balance_info.get("Free", 0))
                locked = float(balance_info.get("Lock", 0))
                total = free + locked

                # Only include assets with significant holdings
                if total > 0.000001:
                    trading_asset = f"{asset}-USD"
                    portfolio_dict[trading_asset] = total

        logger.info(
            f"✅ Portfolio parsed: {len(portfolio_dict)} assets, Cash: ${cash_balance:.2f}"
        )
        return portfolio_dict, cash_balance

    except requests.exceptions.Timeout:
        logger.error("Timeout while fetching portfolio")
        return {}, 0.0
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while fetching portfolio")
        return {}, 0.0
    except Exception as e:
        logger.error(f"Unexpected error in get_current_portfolio: {str(e)}")
        return {}, 0.0


def _get_current_prices(assets: list) -> Dict[str, float]:
    """Get current prices for portfolio assets"""
    prices = {}
    for asset in assets:
        if asset == "USD":
            prices[asset] = 1.0
            continue

        # Convert from trading format (BTC-USD) to API format (BTC/USD)
        if "-USD" in asset:
            pair = f"{asset.replace("-USD", "")}/USD"
        else:
            pair = f"{asset}/USD"

        try:
            response = requests.get(
                f"{BASE_URL}/v3/ticker",
                params={"timestamp": _get_timestamp(), "pair": pair},
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("Success"):
                    last_price = data["Data"][pair]["LastPrice"]
                    prices[asset] = float(last_price)

            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            logger.warning(f"Could not get price for {asset}: {e}")
            prices[asset] = 0.0

    return prices


def _place_order(
    pair_or_coin: str, side: str, quantity: float, order_type: str = "MARKET"
) -> Optional[Dict]:
    """Place an order - internal function for rebalancing"""
    global _last_order_time

    # Convert from trading format (BTC-USD) to API format (BTC/USD)
    if "-USD" in pair_or_coin:
        pair = f"{pair_or_coin.replace('-USD', '')}/USD"
    else:
        pair = f"{pair_or_coin}/USD"

    # Round quantity to appropriate precision
    rounded_quantity = _round_quantity(quantity, pair_or_coin)

    # Rate limiting
    current_time = time.time()
    time_since_last_order = current_time - _last_order_time
    if time_since_last_order < _MIN_ORDER_INTERVAL:
        sleep_time = _MIN_ORDER_INTERVAL - time_since_last_order
        logger.info(f"⏳ Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)

    # Create payload
    payload = {
        "pair": pair,
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": str(rounded_quantity),
    }

    headers, final_payload, total_params_string = _generate_signature(payload)
    headers["Content-Type"] = "application/x-www-form-urlencoded"

    try:
        logger.info(
            f"Placing {side} order for {rounded_quantity} {pair} (original: {quantity})..."
        )
        response = requests.post(
            f"{BASE_URL}/v3/place_order",
            headers=headers,
            data=total_params_string,
            timeout=10,
        )

        _last_order_time = time.time()

        if response.status_code == 200:
            result = response.json()
            if result and result.get("Success"):
                order_id = result.get("OrderDetail", {}).get("OrderID")
                logger.info(
                    f"✅ {side} order placed successfully! Order ID: {order_id}"
                )
                return result
            else:
                error_msg = (
                    result.get("ErrMsg", "Unknown error") if result else "No response"
                )
                logger.error(f"❌ {side} order failed: {error_msg}")
                return None
        elif response.status_code == 429:
            logger.warning("⚠️ Rate limit hit, waiting 1 second...")
            time.sleep(1)
            return _place_order(pair_or_coin, side, rounded_quantity, order_type)
        else:
            logger.error(f"Order API error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Order placement error: {e}")
        return None


def execute_rebalance(target_weights, current_portfolio, cash_balance, threshold=0.12):
    """
    Competition-optimized rebalance with strict anti-wash controls
    """
    # Initialize wash controller
    wash_controller = CompetitionWashController()

    logger.info("🚀 STARTING COMPETITION REBALANCE")
    logger.info(f"Target weights: {target_weights}")
    logger.info(f"Current portfolio: {current_portfolio}")
    logger.info(f"Cash balance: ${cash_balance:.2f}")
    logger.info(f"Rebalance threshold: {threshold*100}%")

    rebalance_result = {
        "success": False,
        "total_orders_placed": 0,
        "sell_orders": [],
        "buy_orders": [],
        "errors": [],
        "final_cash_balance": cash_balance,
        "wash_controls_applied": True,
    }

    try:
        # Step 1: Get current prices for all assets
        all_assets = set(list(target_weights.keys()) + list(current_portfolio.keys()))
        if "USD" in all_assets:
            all_assets.remove("USD")

        prices = _get_current_prices(list(all_assets))
        logger.info(f"Retrieved prices for {len(prices)} assets")

        # Step 2: Calculate current portfolio value and weights
        portfolio_value = 0.0
        for asset, quantity in current_portfolio.items():
            if asset in prices and prices[asset] > 0:
                portfolio_value += quantity * prices[asset]

        total_value = portfolio_value + cash_balance

        if total_value <= 0:
            logger.error("Total portfolio value is zero or negative - cannot rebalance")
            rebalance_result["errors"].append("Total portfolio value is zero")
            return rebalance_result

        logger.info(f"📊 Total Portfolio Value: ${total_value:.2f}")

        # Calculate current weights
        current_weights = {}
        for asset in all_assets:
            if asset in current_portfolio and asset in prices and prices[asset] > 0:
                asset_value = current_portfolio[asset] * prices[asset]
                current_weights[asset] = asset_value / total_value
            else:
                current_weights[asset] = 0.0

        # Add cash as 'USD' in weights
        current_weights["USD"] = cash_balance / total_value

        logger.info("📈 CURRENT WEIGHTS:")
        for asset, weight in current_weights.items():
            if weight > 0.001:  # Only show significant weights
                logger.info(f"   {asset}: {weight:.1%}")

        logger.info("🎯 TARGET WEIGHTS:")
        for asset, weight in target_weights.items():
            logger.info(f"   {asset}: {weight:.1%}")

        # Step 3: Calculate rebalancing needs
        rebalance_actions = []
        logger.info("🧮 Calculating rebalancing actions...")

        for asset, target_weight in target_weights.items():
            if asset == "USD":
                continue

            current_weight = current_weights.get(asset, 0.0)
            weight_diff = current_weight - target_weight
            abs_diff = abs(weight_diff)

            # Check if outside threshold
            if abs_diff > threshold:
                action_type = "SELL" if weight_diff > 0 else "BUY"

                # Calculate USD value to rebalance
                usd_value_to_rebalance = abs(weight_diff) * total_value

                # Calculate quantity to trade
                if asset in prices and prices[asset] > 0:
                    quantity = usd_value_to_rebalance / prices[asset]

                    # Apply minimum quantity checks
                    if quantity > 0.000001:
                        action = {
                            "asset": asset,
                            "action": action_type,
                            "current_weight": current_weight,
                            "target_weight": target_weight,
                            "weight_diff": weight_diff,
                            "usd_value": usd_value_to_rebalance,
                            "quantity": quantity,
                            "price": prices[asset],
                        }
                        rebalance_actions.append(action)

                        logger.info(
                            f"   {asset}: {current_weight:.1%} → {target_weight:.1%} "
                            f"({action_type} {quantity:.6f})"
                        )

        # Also handle assets that need to be completely sold (not in target weights)
        for asset, current_weight in current_weights.items():
            if (
                asset not in target_weights
                and asset != "USD"
                and current_weight > threshold
                and asset in prices
                and prices[asset] > 0
            ):
                usd_value = current_weight * total_value
                quantity = usd_value / prices[asset]

                action = {
                    "asset": asset,
                    "action": "SELL",
                    "current_weight": current_weight,
                    "target_weight": 0.0,
                    "weight_diff": current_weight,
                    "usd_value": usd_value,
                    "quantity": quantity,
                    "price": prices[asset],
                }
                rebalance_actions.append(action)

                logger.info(
                    f"   {asset}: {current_weight:.1%} → 0.0% "
                    f"(SELL {quantity:.6f} - not in target)"
                )

        logger.info(f"📋 Initial rebalance actions: {len(rebalance_actions)}")

        # Step 4: APPLY STRICT ANTI-WASH FILTERING
        filtered_actions = []
        trade_summary = wash_controller.get_trade_summary()

        logger.info(f"🔍 Applying anti-wash controls...")
        logger.info(
            f"   Daily trades: {trade_summary['daily_trades']}/{trade_summary['daily_limit']}"
        )

        for order in rebalance_actions:
            asset = order["asset"]
            action = order["action"]
            price = order["price"]
            quantity = order["quantity"]
            usd_value = order["usd_value"]

            # Check if trade passes wash controls
            can_trade, reason = wash_controller.can_execute_trade(
                asset, action, price, quantity
            )

            if can_trade:
                filtered_actions.append(order)
                logger.info(f"✅ APPROVED {action} {asset}: {reason}")
            else:
                logger.warning(f"🚫 BLOCKED {action} {asset}: {reason}")

        rebalance_actions = filtered_actions
        logger.info(f"📊 After anti-wash filtering: {len(rebalance_actions)} actions")

        # Step 5: Get updated cash balance
        logger.info("🔄 Getting updated cash balance...")
        updated_portfolio, updated_cash = get_current_portfolio()
        rebalance_result["final_cash_balance"] = updated_cash
        logger.info(f"💰 Updated cash balance: ${updated_cash:.2f}")

        # Step 6: Execute SELL orders first
        sell_orders = [
            action for action in rebalance_actions if action["action"] == "SELL"
        ]
        if sell_orders:
            logger.info(f"📤 Executing {len(sell_orders)} SELL orders")

            executed_sells = 0
            for order in sell_orders:
                result = _place_order(order["asset"], "SELL", order["quantity"])

                order_result = {
                    "asset": order["asset"],
                    "action": "SELL",
                    "quantity": order["quantity"],
                    "price": order["price"],
                    "usd_value": order["usd_value"],
                    "success": result is not None and result.get("Success", False),
                    "api_response": result,
                    "timestamp": datetime.now(),
                }

                if order_result["success"]:
                    order_id = result.get("OrderDetail", {}).get("OrderID")
                    order_result["order_id"] = order_id
                    executed_sells += 1

                    # Record successful trade
                    wash_controller.record_trade(
                        order["asset"], "SELL", order["price"], order["quantity"], True
                    )
                else:
                    error_msg = (
                        result.get("ErrMsg", "Unknown error")
                        if result
                        else "No response"
                    )
                    order_result["error"] = error_msg

                    # Record failed trade
                    wash_controller.record_trade(
                        order["asset"], "SELL", order["price"], order["quantity"], False
                    )

                rebalance_result["sell_orders"].append(order_result)

            logger.info(f"✅ Executed {executed_sells} sell orders")

        # Step 7: Execute BUY orders
        buy_orders = [
            action for action in rebalance_actions if action["action"] == "BUY"
        ]
        if buy_orders:
            # Recalculate buy quantities based on actual available cash
            total_buy_value = sum(order["usd_value"] for order in buy_orders)

            if total_buy_value > updated_cash:
                # Scale down buy orders proportionally
                scale_factor = updated_cash / total_buy_value
                logger.warning(
                    f"⚠️ Insufficient cash. Scaling buy orders by {scale_factor:.1%}"
                )

                adjusted_buy_orders = []
                for order in buy_orders:
                    adjusted_usd = order["usd_value"] * scale_factor
                    adjusted_quantity = adjusted_usd / prices[order["asset"]]

                    adjusted_order = order.copy()
                    adjusted_order["usd_value"] = adjusted_usd
                    adjusted_order["quantity"] = adjusted_quantity

                    adjusted_buy_orders.append(adjusted_order)

                    logger.info(
                        f"   {order['asset']}: {order['quantity']:.6f} → {adjusted_quantity:.6f}"
                    )

                buy_orders = adjusted_buy_orders

            logger.info(f"📥 Executing {len(buy_orders)} BUY orders")

            executed_buys = 0
            for order in buy_orders:
                # Final validation
                can_trade, reason = wash_controller.can_execute_trade(
                    order["asset"], "BUY", order["price"], order["quantity"]
                )

                if not can_trade:
                    logger.warning(f"🚫 Skipping {order['asset']} BUY: {reason}")
                    continue

                result = _place_order(order["asset"], "BUY", order["quantity"])

                order_result = {
                    "asset": order["asset"],
                    "action": "BUY",
                    "quantity": order["quantity"],
                    "price": order["price"],
                    "usd_value": order["usd_value"],
                    "success": result is not None and result.get("Success", False),
                    "api_response": result,
                    "timestamp": datetime.now(),
                }

                if order_result["success"]:
                    order_id = result.get("OrderDetail", {}).get("OrderID")
                    order_result["order_id"] = order_id
                    executed_buys += 1

                    # Record successful trade
                    wash_controller.record_trade(
                        order["asset"], "BUY", order["price"], order["quantity"], True
                    )
                else:
                    error_msg = (
                        result.get("ErrMsg", "Unknown error")
                        if result
                        else "No response"
                    )
                    order_result["error"] = error_msg

                    # Record failed trade
                    wash_controller.record_trade(
                        order["asset"], "BUY", order["price"], order["quantity"], False
                    )

                rebalance_result["buy_orders"].append(order_result)

            logger.info(f"✅ Executed {executed_buys} buy orders")

        # Update total orders count
        rebalance_result["total_orders_placed"] = len(
            rebalance_result["sell_orders"]
        ) + len(rebalance_result["buy_orders"])

        # Final validation
        success_orders = len(
            [
                o
                for o in rebalance_result["sell_orders"]
                + rebalance_result["buy_orders"]
                if o.get("success", False)
            ]
        )

        if success_orders > 0:
            rebalance_result["success"] = True
            logger.info(f"✅ REBALANCE COMPLETED: {success_orders} orders executed")

            # Log final trade summary
            final_summary = wash_controller.get_trade_summary()
            logger.info(
                f"📊 Final trade summary: {final_summary['daily_trades']}/{final_summary['daily_limit']} daily trades"
            )
        else:
            logger.warning("⚠️ Rebalance completed but no orders were filled")

        return rebalance_result

    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR during rebalance: {str(e)}")
        rebalance_result["errors"].append(f"Critical error: {str(e)}")
        return rebalance_result


def test_anti_wash_controls():
    """Test the anti-wash controls"""
    print("🧪 Testing Anti-Wash Controls")
    print("=" * 50)

    wash_controller = CompetitionWashController()

    # Test 1: Basic trade validation
    print("1. Testing basic trade validation...")
    can_trade, reason = wash_controller.can_execute_trade("BTC-USD", "BUY", 50000, 0.1)
    print(f"   BUY BTC: {can_trade} - {reason}")

    # Test 2: Record a buy
    print("2. Recording BUY trade...")
    wash_controller.record_trade("BTC-USD", "BUY", 50000, 0.1, True)

    # Test 3: Try immediate sell (should fail)
    print("3. Testing immediate SELL (should fail)...")
    can_trade, reason = wash_controller.can_execute_trade("BTC-USD", "SELL", 51000, 0.1)
    print(f"   SELL BTC: {can_trade} - {reason}")

    # Test 4: Check daily limits
    print("4. Testing daily limits...")
    for i in range(5):
        can_trade, reason = wash_controller.can_execute_trade(
            f"ASSET{i}-USD", "BUY", 100, 1
        )
        print(f"   Trade {i+1}: {can_trade} - {reason}")
        if can_trade:
            wash_controller.record_trade(f"ASSET{i}-USD", "BUY", 100, 1, True)

    print("✅ Anti-wash controls test completed")


if __name__ == "__main__":
    # Run anti-wash tests
    test_anti_wash_controls()

    # Run portfolio test
    print("\n" + "=" * 50)
    portfolio, cash = get_current_portfolio()
    print(f"Portfolio: {len(portfolio)} assets, Cash: ${cash:.2f}")