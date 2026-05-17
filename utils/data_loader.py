# utils/data_loader.py
import pandas as pd
import numpy as np
import os
import streamlit as st

@st.cache_data
def load_data():
    if os.path.exists('data/real_pump_data.csv'):
        try:
            df = pd.read_csv('data/real_pump_data.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except: pass

    # 模拟数据生成
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='10min')
    df = pd.DataFrame({'timestamp': dates})
    t = np.arange(1000)
    df['sensor_00'] = 2000 + 800 * np.sin(t/50) + np.random.normal(0, 50, 1000)
    df['sensor_01'] = 55 - 10 * np.sin(t/50) + np.random.normal(0, 1, 1000)
    df['sensor_04'] = 1.0 + np.random.normal(0, 0.1, 1000)
    return df