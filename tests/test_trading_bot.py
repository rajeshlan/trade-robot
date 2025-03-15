# python -m tests.test_trading_bot 

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import ccxt
import ta
from trading.tradingbot import TradingBot


class TestTradingFunctions(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.api_secret = "test_api_secret"
        self.trading_bot = TradingBot(self.api_key, self.api_secret)

    @patch("trading.tradingbot.ccxt.bybit")
    def test_initialize_exchange(self, mock_bybit):
        mock_exchange = MagicMock()
        mock_bybit.return_value = mock_exchange
        self.trading_bot.initialize_exchange()
        self.assertEqual(self.trading_bot.exchange, mock_exchange)

    @patch("trading.tradingbot.ntplib.NTPClient")
    def test_synchronize_time(self, mock_ntp_client):
        mock_response = MagicMock()
        mock_response.offset = 0.5
        mock_ntp_client.return_value.request.return_value = mock_response
        time_offset = self.trading_bot.synchronize_time()
        self.assertEqual(time_offset, 0.5)

    @patch("trading.tradingbot.ccxt.bybit")
    def test_fetch_data(self, mock_bybit):
        mock_exchange = MagicMock()
        mock_bybit.return_value = mock_exchange
        self.trading_bot.exchange = mock_exchange

        mock_ohlcv = [
            [1625097600000, 34000, 35000, 33000, 34500, 100],
            [1625184000000, 34500, 35500, 34000, 35000, 150],
        ]
        mock_exchange.fetch_ohlcv.return_value = mock_ohlcv

        df = self.trading_bot.fetch_data("BTCUSDT", "1h", 2)
        self.assertEqual(len(df), 2)
        self.assertIn("close", df.columns)

    def calculate_indicators(self, df):
        """
        Calculates technical indicators (SMA and MACD) for the given DataFrame.
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty. Cannot calculate indicators.")

            if len(df) < 26:  # Ensure enough data for MACD calculation
                raise ValueError("Not enough data to calculate MACD. Need at least 26 rows.")

            df["SMA_50"] = df["close"].rolling(window=50, min_periods=1).mean()  # SMA with min periods

            # Calculate MACD using ta library
            macd = ta.trend.MACD(df["close"])
            if macd is not None:
                df["MACD"] = macd.macd()  # FIXED: Use method instead of dictionary access
                df["MACD_signal"] = macd.macd_signal()  # FIXED: Use method instead of dictionary access
            else:
                raise ValueError("MACD calculation returned None.")

            return df

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df  # Return original DataFrame if indicators fail

    def test_calculate_indicators(self):
        df = pd.DataFrame({
            "close": list(range(35000, 35100)),  # 100 rows
            "high": list(range(35100, 35200)),
            "low": list(range(34900, 35000))
        })
        df = self.trading_bot.calculate_indicators(df)
        self.assertIn("SMA_50", df.columns)
        self.assertIn("MACD", df.columns)


    @patch("trading.tradingbot.ccxt.bybit")
    def test_place_order(self, mock_bybit):
        mock_exchange = MagicMock()
        mock_bybit.return_value = mock_exchange
        self.trading_bot.exchange = mock_exchange

        order = self.trading_bot.place_order("buy", 50000, "BTCUSDT", 0.001)
        self.assertIsNotNone(order)


if __name__ == "__main__":
    unittest.main()
