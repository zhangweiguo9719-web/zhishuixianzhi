#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业级泵站数字孪生平台 - 快速测试脚本

验证所有组件是否正常工作
"""

import os
import sys
import pandas as pd
import numpy as np

def test_imports():
    """测试关键依赖包导入"""
    print("🔍 测试依赖包导入...")

    try:
        import streamlit
        print("✅ Streamlit")
    except ImportError:
        print("❌ Streamlit")
        return False

    try:
        import torch
        print("✅ PyTorch")
    except ImportError:
        print("❌ PyTorch")
        return False

    try:
        import pandas
        print("✅ Pandas")
    except ImportError:
        print("❌ Pandas")
        return False

    try:
        import numpy
        print("✅ NumPy")
    except ImportError:
        print("❌ NumPy")
        return False

    try:
        import matplotlib
        print("✅ Matplotlib")
    except ImportError:
        print("❌ Matplotlib")
        return False

    try:
        import sklearn
        print("✅ Scikit-learn")
    except ImportError:
        print("❌ Scikit-learn")
        return False

    return True

def test_data():
    """测试数据文件"""
    print("\n📊 测试数据文件...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "data", "real_pump_data.csv")
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False

    try:
        df = pd.read_csv(data_file)
        print(f"✅ 数据文件加载成功: {len(df)} 行 × {len(df.columns)} 列")

        # 检查必要列
        required_cols = ['machine_status']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ 缺少必要列: {col}")
                return False

        print("✅ 数据结构检查通过")
        return True
    except Exception as e:
        print(f"❌ 数据文件读取失败: {e}")
        return False

def test_models():
    """测试模型文件"""
    print("\n🤖 测试模型文件...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_file = os.path.join(script_dir, "models", "scaler.pkl")
    model_file = os.path.join(script_dir, "models", "pump_lstm_final.pth")

    # 检查归一化器
    if not os.path.exists(scaler_file):
        print(f"⚠️ 归一化器不存在: {scaler_file} (将自动生成)")
    else:
        try:
            import joblib
            scaler = joblib.load(scaler_file)
            print("✅ 数据归一化器存在")
        except Exception as e:
            print(f"❌ 归一化器加载失败: {e}")
            return False

    # 检查模型
    if not os.path.exists(model_file):
        print(f"⚠️ 模型文件不存在: {model_file} (将自动训练)")
    else:
        try:
            import torch
            checkpoint = torch.load(model_file, map_location='cpu')
            print("✅ LSTM模型文件存在")
        except Exception as e:
            print(f"❌ 模型文件加载失败: {e}")
            return False

    return True

def main():
    print("=" * 60)
    print("🧪 工业级泵站数字孪生平台 - 系统测试")
    print("=" * 60)

    all_passed = True

    # 1. 测试Python版本
    version = sys.version_info
    print(f"🐍 Python版本: {version.major}.{version.minor}.{version.micro}")
    if not (version.major >= 3 and version.minor >= 9):
        print("❌ 需要Python 3.9+")
        all_passed = False
    else:
        print("✅ Python版本符合要求")

    # 2. 测试依赖包
    if not test_imports():
        all_passed = False

    # 3. 测试数据文件
    if not test_data():
        all_passed = False

    # 4. 测试模型文件
    if not test_models():
        all_passed = False

    print("\n" + "=" * 60)

    if all_passed:
        print("🎉 系统测试通过！可以正常运行")
        print("\n🚀 启动命令:")
        print("1️⃣ 双击 '启动系统.bat' (一键启动)")
        print("2️⃣ 或依次运行:")
        print("   • 双击 '运行预处理.bat'")
        print("   • 双击 '运行训练.bat'")
        print("   • 双击 '启动应用.bat'")
    else:
        print("⚠️  系统测试发现问题")
        print("\n🔧 修复建议:")
        print("1. 确保所有依赖包已安装")
        print("2. 检查数据文件是否存在")
        print("3. 如有问题，重新运行安装脚本")

    print("\n💡 技术亮点:")
    print("• 基于真实工业数据集 (22万+条)")
    print("• PyTorch LSTM深度学习模型")
    print("• ST-LLM启发的智能预处理")
    print("• 实时预测性维护系统")

    print("=" * 60)
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()









