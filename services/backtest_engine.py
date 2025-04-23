import backtrader as bt
import pandas as pd
import logging
import matplotlib.pyplot as plt
import uuid
import os
from typing import Optional, Dict, Any
from slack_sdk import WebClient

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def unwrap_sentiment_label(label: Any) -> str:
    """Recursively unwrap nested sentiment label structures to a simple lowercase string."""
    max_depth = 5
    depth = 0
    while depth < max_depth and not isinstance(label, str):
        if isinstance(label, (list, tuple)) and label:
            label = label[0]
        elif isinstance(label, dict):
            label = label.get('label', label)
        else:
            break
        depth += 1
    if isinstance(label, str):
        return label.lower()
    raise TypeError(f"Unable to unwrap sentiment label of type {type(label)}: {repr(label)}")


class SentimentStrategy(bt.Strategy):
    params = dict(
        sentiment_data=None,
        leverage=100,
        tp_pct=0.02,
        sl_pct=0.01,
        min_sentiment_score=0.9
    )

    def __init__(self):
        self.sentiments = self.p.sentiment_data
        self.dataclose = self.datas[0].close
        self.entry_price = None
        self.order = None
        self.trades_list = []

    def log(self, msg: str) -> None:
        dt = self.datas[0].datetime.datetime(0)
        logger.info(f"{dt.isoformat()} - {msg}")

    def next(self) -> None:
        current_dt = self.datas[0].datetime.datetime(0)
        timestamp = current_dt.strftime('%Y-%m-%d %H:00')

        raw = self.sentiments.get(timestamp, {'label': 'neutral', 'score': 0.0})
        label = raw.get('label') if isinstance(raw, dict) else raw
        score = raw.get('score', 0.0) if isinstance(raw, dict) else 0.0

        try:
            label = unwrap_sentiment_label(label)
        except Exception:
            label = 'neutral'
            score = 0.0

        score = float(score)
        price = self.dataclose[0]

        # Manage open position
        if self.position:
            if self.position.size > 0:  # Long position
                if price >= self.entry_price * (1 + self.p.tp_pct):
                    self.close_position('long', price, current_dt)
                    return
                if price <= self.entry_price * (1 - self.p.sl_pct):
                    self.close_position('long', price, current_dt)
                    return
            elif self.position.size < 0:  # Short position
                if price <= self.entry_price * (1 - self.p.tp_pct):
                    self.close_position('short', price, current_dt)
                    return
                if price >= self.entry_price * (1 + self.p.sl_pct):
                    self.close_position('short', price, current_dt)
                    return
            return

        # No open position: check new signal
        if score >= self.p.min_sentiment_score and label in ('bullish', 'bearish'):
            size = (self.broker.get_cash() * self.p.leverage) / price
            if label == 'bullish':
                self.log(f"Signal Bullish (score {score:.2f}) → BUY @ {price:.2f}")
                self.buy(size=size)
            else:
                self.log(f"Signal Bearish (score {score:.2f}) → SELL @ {price:.2f}")
                self.sell(size=size)

            self.entry_price = price
            self.trades_list.append({
                'entry_dt': current_dt,
                'side': label,
                'entry_price': price,
                'exit_dt': None,
                'exit_price': None,
                'pnl': None
            })

    def close_position(self, side: str, price: float, dt: pd.Timestamp) -> None:
        last_trade = self.trades_list[-1]
        entry_price = last_trade['entry_price']
        pnl = (price - entry_price) * abs(self.position.size) if side == 'long' else (entry_price - price) * abs(
            self.position.size)

        self.close()
        self.log(f"{side.title()} exit @ {price:.2f} → PnL {pnl:.2f}")

        last_trade.update({
            'exit_dt': dt,
            'exit_price': price,
            'pnl': pnl
        })


def run_backtest(
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        slack_client: Optional[WebClient] = None,
        channel: Optional[str] = None,
        initial_cash: float = 100000.0
) -> Dict[str, Any]:
    """
    Run backtest with sentiment data and optionally post results to Slack.

    Args:
        price_df: DataFrame with OHLCV price data
        sentiment_df: DataFrame with sentiment data (must have 'label' and 'score' columns)
        slack_client: Optional Slack WebClient for posting results
        channel: Optional Slack channel ID
        initial_cash: Initial cash for backtest

    Returns:
        Dictionary containing backtest results and metrics
    """
    # Build sentiment map
    sentiment_map = {}
    for dt, row in sentiment_df.iterrows():
        ts = dt.strftime('%Y-%m-%d %H:00')
        label = row.get('label')
        score = row.get('score', 0.0)
        try:
            clean = unwrap_sentiment_label(label)
        except Exception:
            clean = 'neutral'
            score = 0.0
        sentiment_map[ts] = {'label': clean, 'score': float(score)}

    logger.info(f"Backtest initialized with {len(sentiment_map)} sentiment data points")

    # Prepare backtrader
    datafeed = bt.feeds.PandasData(dataname=price_df)
    cerebro = bt.Cerebro()
    cerebro.adddata(datafeed)
    cerebro.addstrategy(SentimentStrategy, sentiment_data=sentiment_map)
    cerebro.broker.set_cash(initial_cash)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtest
    try:
        results = cerebro.run()
        strat = results[0]
    except Exception as e:
        logger.error("Backtest execution failed:", exc_info=e)
        return {}

    # Extract results
    final_value = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    trade_analysis = strat.analyzers.trades.get_analysis()

    trades = pd.DataFrame(strat.trades_list)
    if not trades.empty:
        best_trade = trades.loc[trades['pnl'].idxmax()].to_dict()
        worst_trade = trades.loc[trades['pnl'].idxmin()].to_dict()
        avg_pnl = trades['pnl'].mean()
        win_rate = len(trades[trades['pnl'] > 0]) / len(trades)
    else:
        best_trade = worst_trade = {}
        avg_pnl = win_rate = 0.0

    # Generate plot
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 6))

    price_df['Close'].plot(ax=ax, label='Price', color='navy', alpha=0.7)

    for _, row in trades.iterrows():
        entry_color = 'green' if row['side'] == 'bullish' else 'red'
        exit_color = 'darkgreen' if row['side'] == 'bullish' else 'darkred'

        ax.scatter(row['entry_dt'], row['entry_price'],
                   marker='^', s=100, color=entry_color, label='Entry')
        ax.scatter(row['exit_dt'], row['exit_price'],
                   marker='v', s=100, color=exit_color, label='Exit')

    ax.set_title('Price with Trade Entries/Exits', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend()
    ax.grid(True)

    # Save plot to temp file
    image_path = f"data/backtest_result_{uuid.uuid4().hex}.png"
    plt.savefig(image_path, bbox_inches='tight', dpi=100)
    plt.close()

    # Upload to Slack if configured
    image_url = None
    if slack_client and channel:
        try:
            response = slack_client.files_upload(
                channels=channel,
                file=image_path,
                title="Backtest Results",
                initial_comment="Here's the trade execution chart:"
            )
            image_url = response['file']['permalink']
        except Exception as e:
            logger.error(f"Failed to upload plot to Slack: {e}")
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    # Prepare results
    results = {
        'final_value': final_value,
        'sharpe_ratio': sharpe,
        'total_trades': len(trades),
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'trade_analysis': trade_analysis,
        'all_trades': trades,
        'plot_url': image_url
    }

    # Print summary
    logger.info("\n===== Backtest Summary =====")
    logger.info(f"Final Portfolio Value: {final_value:.2f}")
    logger.info(f"Sharpe Ratio: {sharpe:.4f}" if sharpe else "Sharpe Ratio: N/A")
    logger.info(f"Total Trades: {len(trades)}")
    logger.info(f"Win Rate: {win_rate:.1%}")
    logger.info(f"Average PnL per Trade: {avg_pnl:.2f}")

    return results
