import backtrader as bt
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SentimentStrategy(bt.Strategy):
    params = dict(sentiment_data=None)

    def __init__(self):
        self.sentiments = self.p.sentiment_data
        self.dataclose = self.datas[0].close

    def next(self):
        current_dt = self.datas[0].datetime.datetime(0)
        timestamp = current_dt.strftime('%Y-%m-%d %H:00')
        sentiment_score = self.sentiments.get(timestamp, "neutral").lower()

        if not self.position:
            if sentiment_score == "bullish":
                self.buy()
            elif sentiment_score == "bearish":
                self.sell()
        else:
            if (self.position.size > 0 and sentiment_score == "bearish") or \
               (self.position.size < 0 and sentiment_score == "bullish"):
                self.close()

def run_backtest(price_df, sentiment_df):
    sentiment_map = {}

    for dt, row in sentiment_df.iterrows():
        timestamp = dt.strftime('%Y-%m-%d %H:00')
        label = row.get("label")

        if isinstance(label, (list, tuple)):
            label = label[0]
        elif isinstance(label, dict):
            label = label.get("label")

        if isinstance(label, str):
            sentiment_map[timestamp] = label.lower()
        else:
            logger.warning(f"Invalid sentiment label at {timestamp}: {label}")

    logger.info(f"Sentiment Map Sample: {list(sentiment_map.items())[:5]}")

    data = bt.feeds.PandasData(dataname=price_df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SentimentStrategy, sentiment_data=sentiment_map)
    cerebro.broker.set_cash(100000)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    result = cerebro.run()
    strat = result[0]

    stats = {
        "Final Portfolio Value": cerebro.broker.getvalue(),
        "Sharpe Ratio": strat.analyzers.sharpe.get_analysis(),
        "Trade Analysis": strat.analyzers.trades.get_analysis(),
    }

    return stats
