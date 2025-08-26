from flask import Flask, render_template, jsonify, request
import datetime
from ml_model import get_ai_signal

app = Flask(__name__)

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
}

signal_history = []
last_signal = None
pending_signal = None
pending_time = None

@app.route("/api/signal")
def api_signal():
    global last_signal, pending_signal, pending_time

    asset = request.args.get("asset", "EUR/USD")
    symbol = ASSETS[asset]

    # AI сигнал (BUY/SELL/NONE + confidence)
    signal, confidence = get_ai_signal(symbol)

    if confidence >= 0.7 and signal in ["BUY", "SELL"]:
        if not pending_signal:
            pending_signal = signal
            pending_time = datetime.datetime.now()
            signal_history.insert(0, f"{pending_time.strftime('%H:%M:%S')} - PENDING {signal} ({confidence*100:.1f}%)")
        elif pending_signal == signal:
            if (datetime.datetime.now() - pending_time).seconds >= 20:
                last_signal = signal
                pending_signal = None
                signal_history.insert(0, f"{datetime.datetime.now().strftime('%H:%M:%S')} - {signal} ({confidence*100:.1f}%)")

    return jsonify({
        "asset": asset,
        "signal": last_signal if last_signal else "NONE",
        "pending": pending_signal if pending_signal else "NONE",
        "history": signal_history[:10]
    })

@app.route("/")
def dashboard():
    return render_template("index.html", assets=list(ASSETS.keys()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
