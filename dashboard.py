# dashboard.py
from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Basic authentication (optional - remove if not needed)
VALID_USERS = {
    os.getenv('DASHBOARD_USER', 'admin'): os.getenv('DASHBOARD_PASSWORD', 'quant123')
}

@app.before_request
def require_auth():
    if request.endpoint != 'health_check':
        auth = request.authorization
        if not auth or VALID_USERS.get(auth.username) != auth.password:
            return 'Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Trading Dashboard"'}

# Simplified HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>🤖 Trading Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .chart-container { height: 250px; margin-top: 15px; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Trading Bot Dashboard</h1>
            <button class="refresh-btn" onclick="loadData()">🔄 Refresh</button>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📊 Performance</h2>
                <div id="summaryMetrics">
                    <div class="metric"><span>Total Rebalances:</span><span id="totalRebalances">-</span></div>
                    <div class="metric"><span>Current Regime:</span><span id="currentRegime">-</span></div>
                    <div class="metric"><span>Avg Portfolio Size:</span><span id="avgPortfolioSize">-</span></div>
                    <div class="metric"><span>Current Lambda:</span><span id="currentLambda">-</span></div>
                </div>
            </div>

            <div class="card">
                <h2>📈 Market Regimes</h2>
                <div class="chart-container"><canvas id="regimeChart"></canvas></div>
            </div>

            <div class="card">
                <h2>📦 Portfolio Size</h2>
                <div class="chart-container"><canvas id="portfolioChart"></canvas></div>
            </div>
        </div>

        <div class="card">
            <h2>🔄 Recent Activity</h2>
            <div id="recentTrades">Loading...</div>
        </div>

        <div style="text-align: center; margin-top: 20px; color: #666;">
            Last updated: <span id="lastUpdate">-</span>
        </div>
    </div>

    <script>
        let regimeChart, portfolioChart;

        async function loadData() {
            try {
                const response = await fetch('/api/performance');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error:', error);
                alert('Error loading data');
            }
        }

        function updateDashboard(data) {
            // Update summary
            const s = data.summary;
            document.getElementById('totalRebalances').textContent = s.totalRebalances;
            document.getElementById('currentRegime').textContent = s.currentRegime;
            document.getElementById('avgPortfolioSize').textContent = s.avgPortfolioSize.toFixed(1);
            document.getElementById('currentLambda').textContent = s.currentLambda.toFixed(3);
            document.getElementById('lastUpdate').textContent = s.lastUpdate;

            // Update charts
            updateCharts(data);
            updateRecentTrades(data.recentTrades);
        }

        function updateCharts(data) {
            // Regime Chart
            if (regimeChart) regimeChart.destroy();
            regimeChart = new Chart(document.getElementById('regimeChart'), {
                type: 'doughnut',
                data: {
                    labels: Object.keys(data.regimeDistribution),
                    datasets: [{
                        data: Object.values(data.regimeDistribution),
                        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56']
                    }]
                }
            });

            // Portfolio Chart
            if (portfolioChart) portfolioChart.destroy();
            const history = data.performanceHistory || [];
            portfolioChart = new Chart(document.getElementById('portfolioChart'), {
                type: 'line',
                data: {
                    labels: history.map((_, i) => `Cycle ${i + 1}`),
                    datasets: [{
                        label: 'Portfolio Size',
                        data: history.map(h => h.portfolio_size || 0),
                        borderColor: '#36A2EB',
                        tension: 0.4
                    }]
                }
            });
        }

        function updateRecentTrades(trades) {
            const container = document.getElementById('recentTrades');
            if (!trades || trades.length === 0) {
                container.innerHTML = '<p>No recent trades</p>';
                return;
            }

            let html = '<table style="width: 100%; border-collapse: collapse;">';
            html += '<tr><th>Time</th><th>Asset</th><th>Action</th><th>Status</th></tr>';
            
            trades.forEach(trade => {
                html += `<tr>
                    <td>${trade.timestamp || 'N/A'}</td>
                    <td>${trade.asset || 'UNKNOWN'}</td>
                    <td>${trade.action || 'N/A'}</td>
                    <td>${trade.success ? '✅' : '❌'}</td>
                </tr>`;
            });
            
            html += '</table>';
            container.innerHTML = html;
        }

        // Load on start and refresh every 30 seconds
        document.addEventListener('DOMContentLoaded', () => {
            loadData();
            setInterval(loadData, 30000);
        });
    </script>
</body>
</html>
"""

class TeamDashboard:
    def __init__(self, performance_logger):
        self.logger = performance_logger
    
    def get_dashboard_data(self):
        """Get data for dashboard"""
        try:
            if hasattr(self.logger, 'generate_performance_report'):
                report = self.logger.generate_performance_report()
            else:
                report = self._generate_basic_report()
            
            return {
                'summary': {
                    'totalRebalances': report.get('total_rebalances', 0),
                    'currentRegime': report.get('current_regime', 'NO_DATA'),
                    'avgPortfolioSize': report.get('avg_portfolio_size', 0),
                    'currentLambda': report.get('latest_lambda', 0),
                    'lastUpdate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'regimeDistribution': report.get('regime_distribution', {'NO_DATA': 1}),
                'recentTrades': report.get('recent_trades', []),
                'performanceHistory': self.get_performance_history()
            }
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return self.get_fallback_data()
    
    def _generate_basic_report(self):
        """Generate basic report"""
        if not hasattr(self.logger, 'regime_history') or not self.logger.regime_history:
            return self.get_fallback_data()
        
        regime_history = self.logger.regime_history
        total_rebalances = len(regime_history)
        
        # Count regimes
        regime_counts = {}
        for item in regime_history:
            regime = item.get('regime', 'UNKNOWN')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'total_rebalances': total_rebalances,
            'current_regime': regime_history[-1].get('regime', 'UNKNOWN') if regime_history else 'UNKNOWN',
            'avg_portfolio_size': sum(item.get('portfolio_size', 0) for item in regime_history) / total_rebalances if total_rebalances > 0 else 0,
            'latest_lambda': regime_history[-1].get('lambda_risk', 0) if regime_history else 0,
            'regime_distribution': regime_counts,
            'recent_trades': []
        }
    
    def get_performance_history(self):
        """Get historical performance data"""
        if not hasattr(self.logger, 'regime_history'):
            return []
        
        return [
            {
                'portfolio_size': item.get('portfolio_size', 0),
                'lambda_risk': float(item.get('lambda_risk', 0))
            }
            for item in getattr(self.logger, 'regime_history', [])
        ]
    
    def get_fallback_data(self):
        """Return fallback data"""
        return {
            'summary': {
                'totalRebalances': 0,
                'currentRegime': 'NO_DATA',
                'avgPortfolioSize': 0,
                'currentLambda': 0,
                'lastUpdate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'regimeDistribution': {'NO_DATA': 1},
            'recentTrades': [],
            'performanceHistory': []
        }

# Global dashboard instance
dashboard = None

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/performance')
def get_performance():
    data = dashboard.get_dashboard_data() if dashboard else {}
    return jsonify(data)

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

def start_dashboard(performance_logger, host='0.0.0.0', port=8050, debug=False):
    global dashboard
    dashboard = TeamDashboard(performance_logger)
    
    logger.info(f"🚀 Starting dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)

# For testing
if __name__ == "__main__":
    class MockLogger:
        def __init__(self):
            self.regime_history = [
                {
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'regime': ['LOW', 'NORMAL', 'HIGH'][i % 3],
                    'portfolio_size': 4 + (i % 3),
                    'lambda_risk': 0.5 + (i * 0.1)
                }
                for i in range(10)
            ]
    
    print("🧪 Starting test dashboard...")
    start_dashboard(MockLogger(), debug=True)