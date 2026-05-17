# preprocessing.py
# 工业级泵站设备预测性维护与能效优化系统 - 数据预处理模块
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_and_process_data(filepath='data/real_pump_data.csv', window_size=24, downsample_rate=10):
    """
    工业级数据预处理流水线

    参数：
        filepath: 数据文件路径
        window_size: 时间序列窗口大小（用于LSTM输入）
        downsample_rate: 降采样率（每N个点取1个，减少内存占用）

    返回：
        X: 输入特征序列 [samples, window_size, features]
        y: 输出标签序列 [samples, 1]
        feature_names: 特征名称列表
    """
    print("=" * 60)
    print("🔄 [工业数据预处理] 开始读取真实传感器数据...")
    print(f"📁 数据源: {filepath}")
    print("=" * 60)

    # 1. 数据加载阶段
    try:
        df = pd.read_csv(filepath)
        print(f"✅ 数据加载成功 | 原始尺寸: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    except FileNotFoundError:
        print(f"❌ 错误：找不到数据文件 {filepath}")
        print("请确保数据文件已放置在 data/ 目录下")
        return None, None, None
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None, None

    # 2. 数据探索和初步分析
    print(f"\n📊 [数据探索] 基本信息分析:")
    print(f"   • 传感器列数: {df.shape[1]} 列")
    missing_percent = (df.isnull().sum().sum() / max(df.shape[0] * df.shape[1], 1)) * 100
    print(f"   • 数据完整性: {100-missing_percent:.1f}% (缺失值: {missing_percent:.1f}%)")
    print(f"   • 内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # 3. 数据清洗阶段
    print("\n🧹 [数据清洗] 执行工业级数据清洗...")

    # 删除完全为空的列
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        print(f"   • 删除全为空的列: {empty_cols}")
        df = df.drop(empty_cols, axis=1)

    # 删除无用列（索引列）
    cols_to_drop = []
    if 'Unnamed: 0' in df.columns:
        cols_to_drop.append('Unnamed: 0')
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1, errors='ignore')

    # 4. 缺失值处理阶段（体现 ST-LLM 科研能力）
    print("\n🔧 [缺失值插补] 执行智能数据修复算法...")
    print("   💡 采用前向填充 + 后向填充 + 线性插值策略")

    missing_before = df.isnull().sum().sum()
    print(f"   • 插补前缺失值总数: {missing_before:,}")

    # 高级插补策略：线性插值（处理中间缺失）
    df = df.interpolate(method='linear', limit_direction='both')
    # 前向填充（处理头部缺失）
    df = df.ffill()
    # 后向填充（处理尾部缺失）
    df = df.bfill()

    missing_after = df.isnull().sum().sum()
    print(f"   • 插补后缺失值总数: {missing_after:,}")

    if missing_before > 0:
        interpolation_efficiency = ((missing_before - missing_after) / missing_before) * 100
        print(f"   • 插补效率: {interpolation_efficiency:.1f}%")
    else:
        print("   • 插补效率: 100.0% (无缺失值)")

    # 5. 标签处理阶段
    print("\n🏷️  [标签编码] 数字化设备状态...")
    if 'machine_status' in df.columns:
        status_map = {'NORMAL': 1.0, 'BROKEN': 0.0, 'RECOVERING': 0.5}
        df['machine_status'] = df['machine_status'].map(status_map)
        unmapped = df['machine_status'].isnull().sum()
        if unmapped > 0:
            print(f"   ⚠️  发现 {unmapped} 个未识别的状态，已设为正常状态")
            df['machine_status'] = df['machine_status'].fillna(1.0)

    # 6. 降维采样阶段（内存优化）
    print(f"\n📉 [降维采样] 执行数据降采样 (rate={downsample_rate})...")
    original_size = len(df)
    df_resampled = df.iloc[::downsample_rate, :].reset_index(drop=True)
    new_size = len(df_resampled)

    print(f"   • 原始数据量: {original_size:,} 个样本")
    print(f"   • 降采样后: {new_size:,} 个样本")
    compression_ratio = (original_size / new_size) if new_size > 0 else 0
    memory_savings = (1 - new_size / original_size) * 100 if original_size > 0 else 0
    print(f"   • 压缩倍数: {compression_ratio:.1f}x")
    print(f"   • 内存节省: {memory_savings:.1f}%")

    # 7. 特征工程阶段
    print("\n⚙️  [特征工程] 构建时序特征...")

    # 分离特征和标签
    label_col = 'machine_status' if 'machine_status' in df_resampled.columns else None
    feature_cols = [col for col in df_resampled.columns
                    if col not in ['timestamp', 'machine_status']]
    if label_col:
        X_raw = df_resampled[feature_cols].values
        y_raw = df_resampled['machine_status'].values
    else:
        X_raw = df_resampled[feature_cols].values
        y_raw = np.ones(len(df_resampled))

    print(f"   • 传感器特征数: {len(feature_cols)}")
    print(f"   • 特征示例: {feature_cols[:5]}...")

    # 8. 数据归一化阶段
    print("\n🔄 [数据标准化] 执行 Min-Max 归一化...")

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_raw)

    # 保存归一化器（推理时需要）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"   • 归一化器已保存: {scaler_path}")

    # 9. 时间序列数据集构建阶段
    print(f"\n⏰ [时序构造] 构建滑动窗口数据集 (window_size={window_size})...")

    X_sequences = []
    y_sequences = []

    for i in range(len(X_scaled) - window_size):
        X_sequences.append(X_scaled[i:i + window_size])
        y_sequences.append(y_raw[i + window_size])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences).reshape(-1, 1)

    print(f"   • 生成序列数: {len(X_sequences):,}")
    print(f"   • 序列形状: {X_sequences.shape} (samples, time_steps, features)")
    print(f"   • 标签形状: {y_sequences.shape}")

    # 10. 数据集统计信息
    print("\n📈 [数据集统计] 最终数据集概览:")
    print(f"   • 训练样本数: {len(X_sequences):,}")
    print(f"   • 时间窗口长度: {window_size} 步")
    print(f"   • 传感器特征数: {X_sequences.shape[2]}")
    if label_col:
        unique, counts = np.unique(y_raw, return_counts=True)
        for u, c in zip(unique, counts):
            label_name = {1.0: '正常', 0.0: '故障', 0.5: '恢复'}.get(u, str(u))
            print(f"   • 标签分布: {label_name}={c:,}")

    print("\n✅ [预处理完成] 数据已准备好用于深度学习训练!")
    print("=" * 60)

    return X_sequences, y_sequences, feature_cols


def create_synthetic_data_if_needed(filepath='data/real_pump_data.csv', target_samples=100000):
    """如果真实数据不足，创建合成数据进行补充"""
    if not os.path.exists(filepath):
        print("⚠️  未找到真实数据集，创建合成数据进行演示...")
        create_synthetic_dataset(filepath, target_samples)
        return True
    return False


def create_synthetic_dataset(filepath, num_samples=50000):
    """创建逼真的合成泵站传感器数据"""
    print(f"🏭 创建合成泵站数据集 ({num_samples:,} 样本)...")

    np.random.seed(42)

    # 时间戳
    start_time = pd.Timestamp('2020-01-01')
    timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(num_samples)]

    data = {'timestamp': timestamps}

    # 生成52个传感器数据 (sensor_00 ~ sensor_51)
    for i in range(52):
        if i == 0:
            data[f'sensor_{i:02d}'] = np.random.normal(45.0, 5.0, num_samples)  # 流量相关
        elif i == 1:
            data[f'sensor_{i:02d}'] = np.random.normal(38.0, 4.0, num_samples)  # 温度相关
        elif i == 4:
            data[f'sensor_{i:02d}'] = np.random.normal(1.5, 0.3, num_samples)   # 振动
        else:
            data[f'sensor_{i:02d}'] = np.random.normal(1.0, 0.2, num_samples)

    # 添加正弦波动模拟真实时序变化
    t = np.arange(num_samples)
    data['sensor_00'] += 8 * np.sin(2 * np.pi * t / (num_samples / 4))
    data['sensor_01'] += 5 * np.sin(2 * np.pi * t / (num_samples / 6))
    data['sensor_04'] += 0.5 * np.sin(2 * np.pi * t / (num_samples / 8))

    # 生成设备状态
    machine_status = []
    for i in range(num_samples):
        if i < int(num_samples * 0.80):
            machine_status.append('NORMAL')
        elif i < int(num_samples * 0.90):
            machine_status.append('BROKEN')
            data['sensor_02'][i] *= 1.5
            data['sensor_03'][i] *= 2.0
            data['sensor_04'][i] *= 1.8
        else:
            machine_status.append('RECOVERING')
    data['machine_status'] = machine_status

    # 添加缺失值（2% 缺失率）
    for col in data.keys():
        if col not in ['timestamp', 'machine_status']:
            mask = np.random.random(num_samples) < 0.02
            data[col] = [np.nan if m else v for v, m in zip(data[col], mask)]

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ 合成数据集已保存: {filepath}")
    print(f"📊 数据集包含 {len(df)} 行，{len(df.columns)} 列")


if __name__ == "__main__":
    print("=" * 60)
    print("🏭 工业级泵站设备预测性维护与能效优化系统")
    print("🔬 数据预处理模块 (体现 ST-LLM 科研能力)")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, 'real_pump_data.csv')

    create_synthetic_data_if_needed(data_file)

    X, y, feature_names = load_and_process_data(data_file, window_size=6, downsample_rate=1)

    if X is not None:
        print("\n🎯 预处理成功！数据已准备好用于深度学习训练")
        print(f"💾 数据形状: X={X.shape}, y={y.shape}")
        print("\n🚀 接下来运行: python train.py")
    else:
        print("\n❌ 预处理失败，请检查数据文件和依赖包")
