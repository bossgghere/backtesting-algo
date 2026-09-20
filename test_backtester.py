import unittest
import numpy as np
import pandas as pd
from backtester import run_backtest_simulation

class BacktestTests(unittest.TestCase):
    def run_case(self, opens, closes, signals, **kwargs):
        data = pd.DataFrame({'open': opens, 'close': closes}, index=pd.date_range('2024-01-01', periods=len(opens)))
        settings = dict(initial_capital=1000, brokerage_per_trade=0, slippage_pct=0, stt_tax_pct=0)
        settings.update(kwargs)
        return run_backtest_simulation(data, lambda d, p: pd.Series(signals, index=d.index), **settings)

    def test_hand_calculated_profit_and_next_open(self):
        r = self.run_case([50,100,110],[50,150,90],[1,-1,0],quantity=10)
        self.assertEqual(r['summary']['final_capital'],1100)
        self.assertEqual(r['trades'][0]['net_pnl'],100)
        self.assertEqual(r['trades'][0]['exit_price'],110)

    def test_hold_and_open_position_valuation(self):
        r = self.run_case([100]*4,[100,100,110,120],[1,0,0,0],quantity=10)
        self.assertEqual(r['summary']['final_capital'],1200)
        self.assertEqual(r['summary']['cash_balance'],0)
        self.assertEqual(r['summary']['total_trades'],0)
        self.assertEqual(r['open_position']['unrealized_pnl'],200)

    def test_fees_tax_turnover_and_losing_trade(self):
        r = self.run_case([100,100,101],[100,100,101],[1,-1,0],quantity=5,brokerage_per_trade=3,stt_tax_pct=.01)
        # Gain 5 minus entry 3 minus exit (3 + 505*.01) = -6.05.
        self.assertAlmostEqual(r['trades'][0]['net_pnl'],-6.05)
        self.assertAlmostEqual(r['summary']['final_capital'],993.95)
        self.assertEqual(r['summary']['winning_trades'],0)

    def test_slippage_both_sides(self):
        r=self.run_case([100]*3,[100]*3,[1,-1,0],quantity=5,slippage_pct=.01)
        self.assertEqual(r['trades'][0]['entry_price'],101)
        self.assertEqual(r['trades'][0]['exit_price'],99)
        self.assertEqual(r['summary']['final_capital'],990)

    def test_sizing_reserves_fee_and_rounds_lots(self):
        r=self.run_case([100]*2,[100]*2,[1,0],lot_size=3,brokerage_per_trade=10)
        self.assertEqual(r['open_position']['quantity'],9)
        self.assertEqual(r['summary']['cash_balance'],90)

    def test_insufficient_funds_no_fee(self):
        r=self.run_case([100]*2,[100]*2,[1,0],quantity=10,brokerage_per_trade=1)
        self.assertEqual(r['summary']['final_capital'],1000)
        self.assertEqual(len(r['rejected_orders']),1)

    def test_sell_while_flat_does_not_short(self):
        r=self.run_case([100]*3,[100,90,80],[-1,-1,0])
        self.assertEqual(r['summary']['final_capital'],1000)
        self.assertIsNone(r['summary']['profit_factor'])
        self.assertIsNone(r['summary']['sharpe_ratio'])

    def test_last_signal_has_no_future_execution(self):
        r=self.run_case([100]*2,[100]*2,[0,1])
        self.assertIsNone(r['open_position'])

    def test_drawdown_includes_capital_and_entry_fee(self):
        r=self.run_case([100]*2,[100]*2,[1,0],quantity=5,brokerage_per_trade=10)
        self.assertAlmostEqual(r['summary']['max_drawdown_pct'],-1)

    def test_accounting_reconciles_closed_and_open_trades(self):
        r=self.run_case([100,100,110,100,120],[100,100,110,100,120],[1,-1,1,0,0],quantity=5,brokerage_per_trade=2)
        pnl=sum(t['net_pnl'] for t in r['trades'])+r['open_position']['net_pnl_to_date']
        self.assertAlmostEqual(r['summary']['final_capital']-1000,pnl)
        self.assertEqual(r['summary']['final_capital'],r['equity_curve']['equity'].iloc[-1])
        self.assertTrue(np.isinf(r['summary']['profit_factor']))

    def test_bad_input_rejected(self):
        for signals in ([1,.5],[1,np.nan],[1,2]):
            with self.subTest(signals=signals), self.assertRaises(ValueError):
                self.run_case([100]*2,[100]*2,signals)
        for kwargs in (dict(quantity=3,lot_size=2),dict(initial_capital=0),dict(slippage_pct=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_case([100]*2,[100]*2,[1,0],**kwargs)
        for opens in ([0,100],[np.nan,100]):
            with self.assertRaises(ValueError): self.run_case(opens,[100]*2,[1,0])

    def test_index_mismatch_and_empty_data(self):
        data=pd.DataFrame({'open':[100,100],'close':[100,100]},index=[1,2])
        for frame, fn in ((data, lambda d,p: pd.Series([1,0],index=[2,3])),(data.iloc[:0],lambda d,p: []),(data.iloc[::-1],lambda d,p: [1,0])):
            with self.assertRaises(ValueError): run_backtest_simulation(frame,fn)

    def test_sharpe_annualization_is_explicit(self):
        a=self.run_case([100]*4,[100,101,99,105],[1,0,0,0],periods_per_year=252)
        b=self.run_case([100]*4,[100,101,99,105],[1,0,0,0],periods_per_year=1008)
        self.assertAlmostEqual(b['summary']['sharpe_ratio'],2*a['summary']['sharpe_ratio'])

if __name__ == '__main__': unittest.main()
