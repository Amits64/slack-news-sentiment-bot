import backtrader as bt
import logging
import traceback
import matplotlib.pyplot as plt

# Set up root logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def unwrap_sentiment_label(label):
    original = label
    max_depth = 5
    depth = 0

    while depth < max_depth and not isinstance(label, str):
        logger.debug(f"🧪 Unwrapping level {depth}: {repr(label)} ({type(label)})")
        if isinstance(label, (list, tuple)) and label:
            label = label[0]
        elif isinstance(label, dict):
            label = label.get("label", label)
        else:
            break
        depth += 1

    if isinstance(label, str):
        logger.debug(f"✅ Final unwrapped label: {repr(label)}")
        return label.lower()
    else:
        logger.error(f"❌ CRITICAL: Could not unwrap. Final value: {repr(label)} ({type(label)})")
        raise TypeError(f"Cannot call .lower() on type: {type(label)})")


class SentimentStrategy(bt.Strategy):
    params = dict(sentiment_data=None)

    def __init__(self):
        self.sentiments = self.p.sentiment_data
        self.dataclose = self.datas[0].close
        self.entry_price = None
        self.order = None
        self.leverage = 100
        self.tp_pct = 0.02
        self.sl_pct = 0.01

        # For plotting entries and exits
        self.trades = []

    def log(self, msg):
        dt = self.datas[0].datetime.datetime(0)
        logger.info(f"{dt.isoformat()} - {msg}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            price = order.executed.price
            self.log(f"{action} EXECUTED at {price:.2f}")

    def next(self):
        current_dt = self.datas[0].datetime.datetime(0)
        timestamp = current_dt.strftime('%Y-%m-%d %H:00')

        raw_entry = self.sentiments.get(timestamp, {"label": "neutral", "score": 0.0})

        # Extract label and score
        label = raw_entry.get("label", "neutral") if isinstance(raw_entry, dict) else raw_entry
        score = raw_entry.get("score", 0.0) if isinstance(raw_entry, dict) else 0.0

        label = label.lower() if isinstance(label, str) else "neutral"
        score = float(score)

        price = self.dataclose[0]

        # Manage existing position
        if self.position:
            if self.position.size > 0:  # Long
                if price >= self.entry_price * (1 + self.tp_pct):
                    self.log(f"TP HIT: Long exit at {price:.2f}")
                    pnl = (price - self.entry_price) * self.position.size
                    self.log(f"PNL: +{pnl:.2f}")
                    self.trades.append((self.entry_price, price, 'long'))
                    self.close()
                elif price <= self.entry_price * (1 - self.sl_pct):
                    self.log(f"SL HIT: Long exit at {price:.2f}")
                    pnl = (price - self.entry_price) * self.position.size
                    self.log(f"PNL: {pnl:.2f}")
                    self.trades.append((self.entry_price, price, 'long'))
                    self.close()
            elif self.position.size < 0:  # Short
                if price <= self.entry_price * (1 - self.tp_pct):
                    self.log(f"TP HIT: Short exit at {price:.2f}")
                    pnl = (self.entry_price - price) * abs(self.position.size)
                    self.log(f"PNL: +{pnl:.2f}")
                    self.trades.append((self.entry_price, price, 'short'))
                    self.close()
                elif price >= self.entry_price * (1 + self.sl_pct):
                    self.log(f"SL HIT: Short exit at {price:.2f}")
                    pnl = (self.entry_price - price) * abs(self.position.size)
                    self.log(f"PNL: {pnl:.2f}")
                    self.trades.append((self.entry_price, price, 'short'))
                    self.close()
            return

        # Open new trade if valid
        if label in ("bullish", "bearish") and score >= 0.9:
            stake = (self.broker.get_cash() * self.leverage) / price
            if label == "bullish":
                self.log(f"Signal: Bullish (score: {score:.2f}) → BUY at {price:.2f}")
                self.buy(size=stake)
                self.entry_price = price
            elif label == "bearish":
                self.log(f"Signal: Bearish (score: {score:.2f}) → SELL at {price:.2f}")
                self.sell(size=stake)
                self.entry_price = price


def run_backtest(price_df, sentiment_df):
    sentiment_map = {}

    for dt, row in sentiment_df.iterrows():
        timestamp = dt.strftime('%Y-%m-%d %H:00')
        label = row.get("label")
        score = row.get("score", 0.0)

        try:
            clean_label = unwrap_sentiment_label(label)
            sentiment_map[timestamp] = {
                "label": clean_label,
                "score": float(score)
            }
        except Exception as e:
            logger.error(f"[run_backtest] Failed to unwrap label at {timestamp}: {repr(label)} — {e}")
            traceback.print_exc()
            sentiment_map[timestamp] = {"label": "neutral", "score": 0.0}

    logger.info(f"Sentiment Map Sample: {list(sentiment_map.items())[:5]}")

    data = bt.feeds.PandasData(dataname=price_df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SentimentStrategy, sentiment_data=sentiment_map)
    cerebro.broker.set_cash(100000)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    try:
        results = cerebro.run()
        strat = results[0]
    except Exception as e:
        logger.error("❌ Cerebro failed to execute strategy.")
        traceback.print_exc()
        return {}

    # Plot with entries and exits
    if hasattr(strat, 'trades'):
        fig, ax = plt.subplots()
        price_df['Close'].plot(ax=ax, label='Price')
        for entry, exit_, side in strat.trades:
            ax.plot(entry, 'g^' if side == 'long' else 'rv', label=f'{side.upper()} Entry')
            ax.plot(exit_, 'ko', label='Exit')
        ax.set_title("Trade Entries/Exits")
        plt.legend()
        plt.grid(True)
        plt.show()

    stats = {
        "Final Portfolio Value": cerebro.broker.getvalue(),
        "Sharpe Ratio": strat.analyzers.sharpe.get_analysis(),
        "Trade Analysis": strat.analyzers.trades.get_analysis(),
    }

    return stats
