import pandas as pd
import numpy as np

def create_cv_dataset() -> pd.DataFrame:
    stationary_features = [
    'Return', 'Return_5d', 'Return_20d', 'Return_Smooth', 
    'RSI_14', 'MACD_Hist', 'Drawdown', 
    'VIX', 'VIX_Change', 'VIX_Change_5d', 
    'GDP_YoY', 'Core_Inflation_YoY', 'M2_YoY', 'Unemployment',
    'Risk_Adj_Return_20d','Relative_Volume',
    'MACD_Hist_Accel'
    ]   

    data = pd.read_csv("data/labeled_dataset.csv")

    data['regime_binary']= np.where(data['regime']=='bull',1,0)
    data['Risk_Adj_Return_20d'] = data['Return_20d'] / (data['VIX'] + 0.00001)
    data['SPY_Volume_20d_MA'] = data['SPY Volume'].rolling(window=20).mean()
    data['Relative_Volume'] = data['SPY Volume'] / data['SPY_Volume_20d_MA']
    data['MACD_Hist_Accel'] = data['MACD_Hist'] - data['MACD_Hist'].shift(1)
    data = data.drop(columns=['SPY_Volume_20d_MA']).dropna()

    return data[stationary_features + ['regime_binary']]