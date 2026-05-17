#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成真实的工业传感器演示数据
包含时间变化、故障模式、噪声等真实特征
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_realistic_sensor_data(num_samples=2000):
    """
    生成真实的工业传感器数据
    
    特征：
    - 时间序列变化（模拟真实工况）
    - 故障模式（某些时段异常）
    - 传感器相关性（某些传感器联动）
    - 噪声和波动（真实工业环境）
    """
    print("🏭 生成工业级传感器演示数据...")
    
    # 生成时间戳（每分钟一个数据点）
    start_time = datetime(2020, 4, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_samples)]
    
    # 初始化数据字典
    data = {'timestamp': timestamps}
    
    np.random.seed(42)  # 确保可重复性
    
    # 生成基础时间模式（模拟一天的工作周期）
    hours = np.array([t.hour for t in timestamps])
    days = np.array([(t - start_time).days for t in timestamps])
    
    # 1. 流量传感器 (sensor_00) - 有早晚高峰
    base_flow = 2.0
    flow_pattern = base_flow + 0.5 * np.sin(2 * np.pi * hours / 24)  # 日周期
    flow_pattern += 0.3 * np.sin(2 * np.pi * hours / 12)  # 半日周期（早晚高峰）
    flow_pattern += np.random.normal(0, 0.2, num_samples)  # 噪声
    data['sensor_00'] = np.maximum(0.5, flow_pattern)  # 确保非负
    
    # 2. 温度传感器 (sensor_01) - 随流量和负载变化
    base_temp = 45.0
    temp_pattern = base_temp + 5 * np.sin(2 * np.pi * hours / 24)  # 日周期
    temp_pattern += 3 * (data['sensor_00'] - base_flow) / base_flow  # 与流量相关
    temp_pattern += np.random.normal(0, 2, num_samples)
    data['sensor_01'] = np.clip(temp_pattern, 30, 70)
    
    # 3. 压力传感器 (sensor_02) - 与流量相关，但有滞后
    base_pressure = 2.5
    pressure_pattern = base_pressure + 0.4 * np.sin(2 * np.pi * hours / 24)
    # 压力滞后于流量
    flow_shifted = np.roll(data['sensor_00'], -5)
    pressure_pattern += 0.3 * (flow_shifted - base_flow) / base_flow
    pressure_pattern += np.random.normal(0, 0.15, num_samples)
    data['sensor_02'] = np.maximum(1.0, pressure_pattern)
    
    # 4. 振动传感器 (sensor_03) - 随转速变化
    base_vibration = 0.5
    vibration_pattern = base_vibration + 0.2 * np.sin(2 * np.pi * hours / 24)
    vibration_pattern += 0.1 * np.sin(2 * np.pi * hours / 6)  # 高频波动
    vibration_pattern += np.random.normal(0, 0.05, num_samples)
    data['sensor_03'] = np.maximum(0.1, vibration_pattern)
    
    # 5. 转速传感器 (sensor_04) - 主要工作参数
    base_rpm = 1500
    rpm_pattern = base_rpm + 100 * np.sin(2 * np.pi * hours / 24)
    rpm_pattern += 50 * np.sin(2 * np.pi * hours / 12)  # 早晚高峰
    rpm_pattern += np.random.normal(0, 30, num_samples)
    data['sensor_04'] = np.clip(rpm_pattern, 1200, 1800)
    
    # 6-51: 其他传感器 - 基于前几个传感器生成相关数据
    sensor_names = [f'sensor_{i:02d}' for i in range(5, 52)]
    
    for i, sensor_name in enumerate(sensor_names):
        if i == 10:  # sensor_15 设为全空（模拟真实数据中的空列）
            data[sensor_name] = [np.nan] * num_samples
        else:
            # 每个传感器有不同的基础值和变化模式
            base_value = 1.0 + (i % 5) * 0.2
            pattern = base_value + 0.3 * np.sin(2 * np.pi * (hours + i*2) / 24)
            # 与主要传感器相关
            if i % 3 == 0:
                pattern += 0.2 * (data['sensor_00'] - base_flow) / base_flow
            elif i % 3 == 1:
                pattern += 0.15 * (data['sensor_01'] - base_temp) / base_temp
            else:
                pattern += 0.1 * (data['sensor_02'] - base_pressure) / base_pressure
            
            pattern += np.random.normal(0, 0.1, num_samples)
            data[sensor_name] = np.maximum(0.1, pattern)
    
    # 生成设备状态（模拟故障模式）
    machine_status = []
    for i in range(num_samples):
        hour = hours[i]
        day = days[i]
        
        # 正常状态（80%的时间）
        if i < int(num_samples * 0.8):
            # 检查是否有异常值（模拟故障前兆）
            if (data['sensor_02'][i] > 3.5 or  # 压力过高
                data['sensor_03'][i] > 0.8 or  # 振动过大
                data['sensor_01'][i] > 65):    # 温度过高
                machine_status.append('BROKEN')
                # 故障期间修改传感器值
                if np.random.random() < 0.5:
                    data['sensor_02'][i] *= 1.5
                    data['sensor_03'][i] *= 2.0
                    data['sensor_01'][i] *= 1.2
            else:
                machine_status.append('NORMAL')
        # 故障状态（10%的时间）
        elif i < int(num_samples * 0.9):
            machine_status.append('BROKEN')
            # 故障期间的异常值
            if np.random.random() < 0.7:
                data['sensor_02'][i] *= np.random.uniform(1.3, 1.8)
                data['sensor_03'][i] *= np.random.uniform(1.5, 2.5)
                data['sensor_01'][i] *= np.random.uniform(1.1, 1.4)
        # 恢复状态（10%的时间）
        else:
            machine_status.append('RECOVERING')
            # 恢复期间逐渐恢复正常
            recovery_ratio = (i - int(num_samples * 0.9)) / (num_samples * 0.1)
            data['sensor_02'][i] = base_pressure + (data['sensor_02'][i] - base_pressure) * (1 - recovery_ratio)
            data['sensor_03'][i] = base_vibration + (data['sensor_03'][i] - base_vibration) * (1 - recovery_ratio)
    
    data['machine_status'] = machine_status
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 添加一些随机缺失值（模拟真实工业数据）
    for col in df.columns:
        if col not in ['timestamp', 'machine_status']:
            mask = np.random.random(num_samples) < 0.02  # 2%缺失率
            df.loc[mask, col] = np.nan
    
    # 保存数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, 'real_pump_data.csv')
    df.to_csv(output_file, index=False)
    
    print(f"✅ 数据生成完成！")
    print(f"   • 数据点数: {len(df):,}")
    print(f"   • 时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    print(f"   • 设备状态分布:")
    print(f"     - 正常: {sum(df['machine_status'] == 'NORMAL'):,} ({sum(df['machine_status'] == 'NORMAL')/len(df)*100:.1f}%)")
    print(f"     - 故障: {sum(df['machine_status'] == 'BROKEN'):,} ({sum(df['machine_status'] == 'BROKEN')/len(df)*100:.1f}%)")
    print(f"     - 恢复: {sum(df['machine_status'] == 'RECOVERING'):,} ({sum(df['machine_status'] == 'RECOVERING')/len(df)*100:.1f}%)")
    print(f"   • 文件保存至: {output_file}")
    
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("🏭 工业级传感器数据生成器")
    print("=" * 60)
    
    # 生成2000个数据点（约33小时的数据）
    df = generate_realistic_sensor_data(num_samples=2000)
    
    print("\n💡 数据特点:")
    print("   • 时间序列变化（日周期、早晚高峰）")
    print("   • 传感器相关性（压力滞后于流量）")
    print("   • 故障模式（异常值触发故障状态）")
    print("   • 真实噪声（模拟工业环境）")
    print("\n🚀 现在可以重新运行应用查看动态数据！")









