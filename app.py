

from flask import Flask, render_template, jsonify

app = Flask(__name__)


@app.route('/')
def home():
    # You can pass variables from Python to HTML here
    user_name = "Explorer"
    return render_template('dashboard.html', name=user_name)


@app.route('/api/trades')
def api_trades():
    # Simple sample trade data for the dashboard to consume
    trades = [
        {"t": "14:31:07", "sym": "XAUUSD", "dir": "BUY",  "lot": "0.10", "ep": "2641.0", "pnl": "+$24.5"},
        {"t": "14:28:43", "sym": "EURUSD", "dir": "BUY",  "lot": "0.50", "ep": "1.0842", "pnl": "+$18.0"},
        {"t": "14:21:15", "sym": "BTCUSD", "dir": "BUY",  "lot": "0.05", "ep": "97280", "pnl": "+$113.0"},
        {"t": "14:15:02", "sym": "GBPUSD", "dir": "SELL", "lot": "0.30", "ep": "1.2671", "pnl": "-$41.0"},
        {"t": "14:08:33", "sym": "US500",  "dir": "BUY",  "lot": "1.00", "ep": "5888.0", "pnl": "+$75.0"},
        {"t": "13:55:47", "sym": "USDJPY", "dir": "SELL", "lot": "0.40", "ep": "151.85", "pnl": "-$28.0"},
        {"t": "13:41:20", "sym": "USDCAD", "dir": "SELL", "lot": "0.20", "ep": "1.3660", "pnl": "+$16.0"},
        {"t": "13:30:05", "sym": "EURUSD", "dir": "BUY",  "lot": "0.40", "ep": "1.0831", "pnl": "+$56.0"},
        {"t": "13:18:47", "sym": "XAUUSD", "dir": "SELL", "lot": "0.10", "ep": "2638.5", "pnl": "-$12.0"},
        {"t": "13:05:13", "sym": "GBPUSD", "dir": "BUY",  "lot": "0.20", "ep": "1.2655", "pnl": "+$34.0"}
    ]
    return jsonify(trades)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)