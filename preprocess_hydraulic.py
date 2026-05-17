"""
预处理脚本：UCI 液压系统状态监测数据集
============================================
数据来源：ZeMA gGmbH 工业液压测试台架
原始链接：https://archive.ics.uci.edu/dataset/444/

目标：生成适配"智水先知"项目的训练数据
诊断目标：内部泵泄漏（Internal Pump Leakage）三分类
  - 0: 无泄漏（正常）
  - 1: 轻微泄漏
  - 2: 严重泄漏

传感器通道（不同采样率）：
  PS1-6   : 压力传感器  100Hz × 6000点/周期
  EPS1    : 电机功率    100Hz × 6000点/周期
  FS1-2   : 流量传感器   10Hz ×  600点/周期
  TS1-4   : 温度传感器    1Hz ×   60点/周期
  VS1     : 振动传感器    1Hz ×   60点/周期
  SE      : 系统效率      1Hz ×   60点/周期
  CE      : 冷却效率      1Hz ×   60点/周期
  CP      : 冷却功率      1Hz ×   60点/周期
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 配置路径
# ============================================================================
HYDRAULIC_DATA_DIR = r"data/condition+monitoring+of+hydraulic+systems"
OUTPUT_DATA_PATH = "data/hydraulic_pump_data.csv"
OUTPUT_MODEL_DIR = "models_hydraulic"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# 传感器文件列表
SENSOR_FILES = {
    # 100Hz 高频信号 -> 降采样到 10Hz（每秒取10个点，60秒=600点）
    "PS1": ("PS1.txt", 6000, 10),   # 压力传感器1
    "PS2": ("PS2.txt", 6000, 10),   # 压力传感器2
    "PS3": ("PS3.txt", 6000, 10),   # 压力传感器3
    "PS4": ("PS4.txt", 6000, 10),   # 压力传感器4
    "PS5": ("PS5.txt", 6000, 10),   # 压力传感器5
    "PS6": ("PS6.txt", 6000, 10),   # 压力传感器6
    "EPS1": ("EPS1.txt", 6000, 10), # 电机功率

    # 10Hz 中频信号 -> 保持（600点）
    "FS1": ("FS1.txt", 600, 1),     # 流量传感器1
    "FS2": ("FS2.txt", 600, 1),     # 流量传感器2

    # 1Hz 低频信号 -> 上采样到 10Hz（插值到600点）
    "TS1": ("TS1.txt", 60, 10),     # 温度传感器1
    "TS2": ("TS2.txt", 60, 10),     # 温度传感器2
    "TS3": ("TS3.txt", 60, 10),     # 温度传感器3
    "TS4": ("TS4.txt", 60, 10),     # 温度传感器4
    "VS1": ("VS1.txt", 60, 10),     # 振动传感器
    "SE":  ("SE.txt", 60, 10),      # 系统效率
    "CE":  ("CE.txt", 60, 10),      # 冷却效率
    "CP":  ("CP.txt", 60, 10),      # 冷却功率
}

def load_and_resample(filepath, n_points, target_rate_ratio):
    """
    加载传感器数据并重采样到统一长度

    参数:
        filepath: 文件路径
        n_points: 原始点数
        target_rate_ratio: 目标采样倍率
            >1 表示降采样（如100Hz->10Hz取1/10）
            =1 表示保持（如10Hz本身）
            <1 表示上采样（如1Hz->10Hz插值10倍）
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            values = line.strip().split('\t')
            row = [float(v) for v in values if v.strip()]
            data.append(row)

    data = np.array(data)

    if target_rate_ratio > 1:
        # 降采样：每隔 N 个点取一个
        step = target_rate_ratio
        resampled = data[:, ::step]  # shape: (n_samples, 6000/step)
    elif target_rate_ratio < 1:
        # 上采样：线性插值到统一长度 600
        step = int(1 / target_rate_ratio)  # 上采样倍率
        n_target = 600  # 60秒 × 10Hz
        resampled = np.zeros((data.shape[0], n_target))
        for i in range(data.shape[0]):
            x_orig = np.arange(len(data[i]))
            x_new = np.linspace(0, len(data[i])-1, n_target)
            resampled[i] = np.interp(x_new, x_orig, data[i])
    else:
        # 保持原样
        resampled = data

    return resampled

def extract_cycle_features(cycle_data_dict):
    """
    从一个60秒周期的多传感器数据中提取统计特征

    参数:
        cycle_data_dict: {传感器名: np.array(周期数, 点数)}

    返回:
        feature_dict: {传感器名: [均值, 标准差, 最大, 最小, RMS, 峰峰值]}
    """
    features = {}

    for sensor_name, data in cycle_data_dict.items():
        # data shape: (n_samples_per_cycle, n_points)
        # 对每个样本计算统计量，然后取平均
        mean_vals = np.mean(data, axis=1)
        std_vals = np.std(data, axis=1)
        max_vals = np.max(data, axis=1)
        min_vals = np.min(data, axis=1)

        # RMS
        rms_vals = np.sqrt(np.mean(data**2, axis=1))
        # 峰峰值
        range_vals = max_vals - min_vals

        # 聚合：对该周期内的所有采样取平均（平滑噪声）
        features[f"{sensor_name}_mean"] = np.mean(mean_vals)
        features[f"{sensor_name}_std"] = np.mean(std_vals)
        features[f"{sensor_name}_max"] = np.mean(max_vals)
        features[f"{sensor_name}_min"] = np.mean(min_vals)
        features[f"{sensor_name}_rms"] = np.mean(rms_vals)
        features[f"{sensor_name}_range"] = np.mean(range_vals)

    return features

def preprocess_pipeline():
    """
    主处理流程
    """
    print("=" * 60)
    print("UCI 液压系统数据集预处理")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 加载 profile 标签文件
    # -------------------------------------------------------------------------
    print("\n[1/6] 加载标签文件...")
    profile_path = os.path.join(HYDRAULIC_DATA_DIR, "profile.txt")
    profile_data = []
    with open(profile_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                cooler = int(parts[0])
                valve = int(parts[1])
                pump_leak = int(parts[2])      # 内部泵泄漏 ← 目标标签
                accumulator = int(parts[3])
                stable = int(parts[4])
                profile_data.append({
                    'cooler': cooler,
                    'valve': valve,
                    'pump_leakage': pump_leak,
                    'accumulator': accumulator,
                    'stable': stable
                })

    profile_df = pd.DataFrame(profile_data)
    n_cycles = len(profile_df)
    print(f"   -> 总周期数: {n_cycles}")
    print(f"   -> 内部泵泄漏分布:\n{profile_df['pump_leakage'].value_counts().sort_index()}")

    # -------------------------------------------------------------------------
    # 2. 加载并重采样所有传感器数据
    # -------------------------------------------------------------------------
    print("\n[2/6] 加载并重采样传感器数据...")

    sensor_arrays = {}
    for sensor_name, (filename, n_points, rate_ratio) in SENSOR_FILES.items():
        filepath = os.path.join(HYDRAULIC_DATA_DIR, filename)
        print(f"   加载 {sensor_name} ({filename})...", end=" ")
        arr = load_and_resample(filepath, n_points, rate_ratio)
        sensor_arrays[sensor_name] = arr
        print(f"shape={arr.shape}")

    # -------------------------------------------------------------------------
    # 3. 为每个周期提取特征
    # -------------------------------------------------------------------------
    print("\n[3/6] 提取周期特征...")

    all_features = []
    all_labels = []

    # 对高频信号做均值平滑（每秒一个统计量）
    # PS1-6, EPS1: 6000点/周期 -> 600点(10Hz) -> 60点(1Hz)
    # FS1-2: 600点/周期 -> 60点(1Hz)
    # TS1-4, VS1, SE, CE, CP: 60点/周期 -> 保持

    # 先把高频信号降采样到 1Hz（60点/周期）以统一时序长度
    high_freq_sensors = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"]
    low_freq_sensors = ["FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "SE", "CE", "CP"]

    # 降高频信号到 1Hz
    for sname in high_freq_sensors:
        arr = sensor_arrays[sname]  # (2205, 600)
        # 每10个点取均值 -> (2205, 60)
        arr_1hz = arr.reshape(arr.shape[0], 60, 10).mean(axis=2)
        sensor_arrays[sname + "_1hz"] = arr_1hz

    # 低频信号直接reshape到60点（已经是60或600）
    for sname in low_freq_sensors:
        arr = sensor_arrays[sname]
        if arr.shape[1] == 600:
            arr_1hz = arr.reshape(arr.shape[0], 60, 10).mean(axis=2)
            sensor_arrays[sname + "_1hz"] = arr_1hz
        else:
            sensor_arrays[sname + "_1hz"] = arr

    # -------------------------------------------------------------------------
    # 3a. 方案A：统计特征（适用于传统ML和简单LSTM）
    # -------------------------------------------------------------------------
    print("   提取统计特征...")
    for cycle_idx in range(n_cycles):
        if cycle_idx % 200 == 0:
            print(f"   进度: {cycle_idx}/{n_cycles}")

        cycle_features = {}

        # 所有1Hz传感器
        all_1hz_sensors = (
            [s + "_1hz" for s in high_freq_sensors] +
            [s + "_1hz" for s in low_freq_sensors]
        )

        for sname in all_1hz_sensors:
            base_name = sname.replace("_1hz", "")
            arr = sensor_arrays[sname][cycle_idx]  # (60,)
            cycle_features[f"{base_name}_mean"] = np.mean(arr)
            cycle_features[f"{base_name}_std"] = np.std(arr)
            cycle_features[f"{base_name}_max"] = np.max(arr)
            cycle_features[f"{base_name}_min"] = np.min(arr)
            cycle_features[f"{base_name}_rms"] = np.sqrt(np.mean(arr**2))
            cycle_features[f"{base_name}_range"] = np.max(arr) - np.min(arr)
            # 新增：斜率（趋势）
            x = np.arange(len(arr))
            slope = np.polyfit(x, arr, 1)[0]
            cycle_features[f"{base_name}_slope"] = slope

        all_features.append(cycle_features)
        all_labels.append(profile_df['pump_leakage'].iloc[cycle_idx])

    features_df = pd.DataFrame(all_features)
    labels = np.array(all_labels)

    print(f"   -> 特征维度: {features_df.shape}")
    print(f"   -> 标签分布: 0(正常)={np.sum(labels==0)}, 1(轻微)={np.sum(labels==1)}, 2(严重)={np.sum(labels==2)}")

    # -------------------------------------------------------------------------
    # 4. 归一化 + 标签编码
    # -------------------------------------------------------------------------
    print("\n[4/6] 归一化与编码...")

    # 标签映射（符合项目书描述）
    label_map = {0: 'NORMAL', 1: 'WEAK_LEAK', 2: 'SEVERE_LEAK'}
    label_str = [label_map[l] for l in labels]

    # 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features_df.values)

    feature_names = list(features_df.columns)
    print(f"   -> 特征数量: {len(feature_names)}")

    # -------------------------------------------------------------------------
    # 5. 保存数据
    # -------------------------------------------------------------------------
    print("\n[5/6] 保存处理后的数据...")

    # 构建DataFrame
    output_df = pd.DataFrame(X_scaled, columns=feature_names)
    output_df['timestamp'] = [f"cycle_{i}" for i in range(n_cycles)]
    output_df['machine_status'] = label_str
    output_df['pump_leak_label'] = labels  # 数值标签

    output_df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"   -> 保存至: {OUTPUT_DATA_PATH}")

    # 保存归一化器
    import joblib
    joblib.dump(scaler, f"{OUTPUT_MODEL_DIR}/hydraulic_scaler.pkl")
    print(f"   -> 归一化器保存至: {OUTPUT_MODEL_DIR}/hydraulic_scaler.pkl")

    # -------------------------------------------------------------------------
    # 6. 划分训练集/测试集
    # -------------------------------------------------------------------------
    print("\n[6/6] 划分数据集...")

    # 统一转为数值标签用于训练
    le = LabelEncoder()
    y_encoded = le.fit_transform(label_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"   -> 训练集: {len(X_train)} 样本")
    print(f"   -> 测试集: {len(X_test)} 样本")
    print(f"   -> 类别: {list(le.classes_)}")

    # 保存划分后的数据
    np.save(f"{OUTPUT_MODEL_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_MODEL_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_MODEL_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_MODEL_DIR}/y_test.npy", y_test)
    np.save(f"{OUTPUT_MODEL_DIR}/feature_names.npy", np.array(feature_names))
    import joblib as jbl
    jbl.dump(le, f"{OUTPUT_MODEL_DIR}/label_encoder.pkl")

    print("\n" + "=" * 60)
    print("预处理完成！")
    print(f"  数据文件: {OUTPUT_DATA_PATH}")
    print(f"  模型文件: {OUTPUT_MODEL_DIR}/")
    print("=" * 60)

    return output_df, scaler, le

if __name__ == "__main__":
    df, scaler, le = preprocess_pipeline()
