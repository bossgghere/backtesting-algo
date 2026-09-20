import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import data_loader as dl
from backtester import run_backtest_simulation

class DataTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'open':[100,100,110], 'high':[101,106,111], 'low':[99,99,109], 'close':[100,105,110], 'volume':[0,20,30]}, index=pd.date_range('2024-01-01',periods=3))

    def test_validation(self):
        good=self.frame()
        pd.testing.assert_frame_equal(dl.validate_market_data(good.iloc[::-1]),good)
        for col,value in [('high',1),('low',200),('close',float('nan')),('volume',-1)]:
            bad=good.copy(); bad.loc[bad.index[0],col]=value
            with self.subTest(col=col), self.assertRaises(ValueError): dl.validate_market_data(bad)
        for bad in (good.iloc[:0],pd.concat([good,good]),good.drop(columns='volume')):
            with self.assertRaises(ValueError): dl.validate_market_data(bad)

    def test_csv_to_engine_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.frame().to_csv(root/'prices.csv',index_label='timestamp')
            with patch.object(dl,'ROOT',root),patch.dict(os.environ,{'DATA_PROVIDER':'csv','HISTORICAL_CSV_PATH':'prices.csv','HISTORICAL_CSV_SYMBOL':'TEST'}):
                data=dl.get_market_data('TEST')
                again=dl.get_market_data('TEST')
                self.assertEqual(data.attrs['snapshot_id'],again.attrs['snapshot_id'])
                self.assertTrue((root/'.data_cache/snapshots'/f'{data.attrs["snapshot_id"]}.csv').exists())
                result=run_backtest_simulation(data,lambda d,p:pd.Series([1,-1,0],index=d.index),initial_capital=1000,quantity=10,brokerage_per_trade=0,slippage_pct=0,stt_tax_pct=0)
                self.assertEqual(result['summary']['final_capital'],1100)
                with self.assertRaises(ValueError): dl.get_market_data('WRONG')

    def test_provider_swap_preserves_engine_input(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(dl,'ROOT',Path(tmp)),patch.object(dl.YahooProvider,'fetch',return_value=self.frame()) as fetch:
            data=dl.get_market_data('TEST',period='6mo',provider='yahoo')
            fetch.assert_called_once_with('TEST','6mo','1d')
            self.assertEqual(list(data.columns),dl.COLUMNS)
            self.assertEqual(data.attrs['provider'],'yahoo')

    def test_reject_unsupported_source_and_interval(self):
        with self.assertRaises(ValueError): dl.get_market_data(provider='unknown')
        with self.assertRaises(ValueError): dl.get_market_data(interval='1m')
        frame=self.frame(); frame.index=pd.date_range('2024-01-01',periods=3,freq='min')
        with self.assertRaises(ValueError): dl.validate_market_data(frame)

if __name__=='__main__': unittest.main()
