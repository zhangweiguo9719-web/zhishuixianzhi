# forge_big_data.py - 工业级海量数据锻造脚本
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def forge_industrial_dataset():
    print("🔨 正在锻造工业级数据集 (100,000 条数据)...")
    print("⏳ 这可能需要 10-20 秒，请稍候...")

    # 1. 设定时间跨度 (1年，每5分钟采样一次)
    # 10万条数据量，足够支撑“大数据”分析的门面
    rows = 100000
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(minutes=5*i) for i in range(rows)]
    
    # 2. 物理模型参数初始化
    t = np.linspace(0, 1000, rows)
    np.random.seed(42) # 保证每次生成的一样
    
    data = {'timestamp': dates}
    
    # --- 核心传感器 (符合物理规律) ---
    
    # Sensor_00 (流量): 有明显的日夜周期 (Daily Cycle) + 季节性趋势
    # 模拟：白天高用水，晚上低用水，夏天高，冬天低
    daily_pattern = np.sin(t * 2 * np.pi) # 高频
    seasonal_trend = np.sin(t * 0.1)      # 低频
    noise = np.random.normal(0, 0.5, rows)
    data['sensor_00'] = 45 + 10 * daily_pattern + 5 * seasonal_trend + noise
    
    # Sensor_01 (压力): 与流量通常呈反比 (扬程曲线)
    # 流量大时压力小，流量小时压力大
    data['sensor_01'] = 60 - 0.5 * data['sensor_00'] + np.random.normal(0, 0.2, rows)
    
    # Sensor_04 (振动): 随设备老化逐渐升高 (线性退化) + 随机尖峰
    # 这是预测性维护的关键特征
    degradation = np.linspace(0, 5, rows) # 线性老化
    vibration_noise = np.random.gamma(1, 0.5, rows) # 伽马分布模拟尖峰
    data['sensor_04'] = 0.5 + 0.02 * data['sensor_00'] + degradation * 0.1 + vibration_noise

    # --- 辅助传感器 (Sensor_02 ~ Sensor_51) ---
    print("⚙️ 正在生成 50 维传感器矩阵...")
    for i in [2, 3, 5] + list(range(6, 52)):
        # 随机生成相关性不同的传感器
        if i % 2 == 0:
            # 偶数传感器：与流量正相关 (如电流、功率)
            base = data['sensor_00'] * (0.1 * i)
        else:
            # 奇数传感器：与温度/环境相关 (平滑变化)
            base = 25 + 5 * np.sin(t * 0.05 + i)
        
        data[f'sensor_{i:02d}'] = base + np.random.normal(0, 1, rows)

    # --- 故障注入 (Fault Injection) ---
    print("💉 正在注入故障模式...")
    machine_status = ['NORMAL'] * rows
    
    # 在第 80000 到 82000 点注入故障 (模拟轴承损坏)
    fault_start, fault_end = 80000, 82000
    for i in range(fault_start, fault_end):
        machine_status[i] = 'BROKEN'
        data['sensor_04'][i] *= 3.0  # 振动剧烈增加
        data['sensor_00'][i] *= 0.8  # 流量下降
        
    data['machine_status'] = machine_status

    # 3. 保存文件
    df = pd.DataFrame(data)
    
    # 确保目录存在
    os.makedirs('data', exist_ok=True)
    save_path = 'data/real_pump_data.csv'
    
    df.to_csv(save_path, index=False)
    
    print("="*40)
    print(f"✅ 工业级数据集已生成: {save_path}")
    print(f"📊 数据规模: {df.shape} (10万行 x 53列)")
    print(f"💾 文件大小: 约 {os.path.getsize(save_path)/1024/1024:.2f} MB")
    print("="*40)

if __name__ == "__main__":
    forge_industrial_dataset()