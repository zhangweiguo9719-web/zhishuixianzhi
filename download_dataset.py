#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pump Sensor Data 数据集下载脚本

使用合成方式生成符合泵站特性的工业级数据。
真实 Kaggle 数据需要 Kaggle API key，此处自动生成等效的演示数据。
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_directories():
    """创建项目目录结构"""
    dirs = ['data', 'models', 'logs', 'checkpoints']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_dataset(num_samples=100000):
    """生成符合泵站物理特性的传感器数据"""
    print("[GEN] 开始生成工业级数据集...")

    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(minutes=5 * i) for i in range(num_samples)]

    np.random.seed(42)
    t = np.linspace(0, 1000, num_samples)

    data = {'timestamp': dates}

    # Sensor_00: 流量 — 日夜周期 + 季节趋势
    daily = np.sin(2 * np.pi * t / (num_samples / 4))
    seasonal = np.sin(2 * np.pi * t / (num_samples / 6))
    data['sensor_00'] = 45 + 10 * daily + 5 * seasonal + np.random.normal(0, 0.5, num_samples)

    # Sensor_01: 温度 — 随流量和负载变化
    data['sensor_01'] = 60 - 0.5 * data['sensor_00'] + np.random.normal(0, 0.2, num_samples)

    # Sensor_04: 振动 — 随设备老化线性增长（预测性维护关键特征）
    degradation = np.linspace(0, 5, num_samples)
    vibration_noise = np.random.gamma(1, 0.5, num_samples)
    data['sensor_04'] = 0.5 + 0.02 * data['sensor_00'] + degradation * 0.1 + vibration_noise

    # 其余传感器 (sensor_02, 03, 05~51)
    print("[GEN] 生成 50 维传感器矩阵...")
    for i in [2, 3] + list(range(5, 52)):
        if i % 2 == 0:
            base = data['sensor_00'] * (0.1 * i)
        else:
            base = 25 + 5 * np.sin(t * 0.05 + i)
        data[f'sensor_{i:02d}'] = base + np.random.normal(0, 1, num_samples)

    # 故障注入 (第 80000 ~ 82000 点，模拟轴承损坏)
    machine_status = ['NORMAL'] * num_samples
    fault_start, fault_end = 80000, 82000
    for i in range(fault_start, fault_end):
        machine_status[i] = 'BROKEN'
        data['sensor_04'][i] *= 3.0
        data['sensor_00'][i] *= 0.8

    data['machine_status'] = machine_status

    # 添加 5% 随机缺失值
    for col in data.keys():
        if col not in ['timestamp', 'machine_status']:
            mask = np.random.random(len(data[col])) < 0.05
            data[col] = np.where(mask, np.nan, data[col])

    df = pd.DataFrame(data)
    return df


def main():
    print("=" * 60)
    print("智水先知 - 数据集生成器")
    print("=" * 60)

    create_directories()

    df = generate_dataset(num_samples=100000)

    save_path = 'data/real_pump_data.csv'
    df.to_csv(save_path, index=False)

    print(f"[OK] 数据集已生成: {save_path}")
    print(f"     数据规模: {df.shape[0]:,} 行 x {df.shape[1]} 列")
    print(f"     故障区间: 第 80000 ~ 82000 点")
    print(f"     文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    print()
    print("下一步:")
    print("  1. python preprocessing.py   (数据预处理)")
    print("  2. python train.py          (训练模型)")
    print("  3. python -m streamlit run app.py  (启动界面)")
    print()

    input("按回车键退出...")


if __name__ == "__main__":
    main()
