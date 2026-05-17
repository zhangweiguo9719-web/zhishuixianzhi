#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏭 工业级泵站数字孪生平台 - 运行指南

自动检测系统状态并提供运行指导
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"🐍 Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 9:
        print("✅ Python版本符合要求")
        return True
    else:
        print("❌ 需要Python 3.9+")
        return False

def check_dependencies():
    """检查关键依赖包"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'torch', 'matplotlib', 'plotly'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)

    return len(missing_packages) == 0, missing_packages

def check_files():
    """检查项目文件完整性"""
    required_files = [
        'app.py',
        'preprocessing.py',
        'train.py',
        'requirements.txt',
        'data/real_pump_data.csv'
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - 存在")
        else:
            print(f"❌ {file_path} - 缺失")
            missing_files.append(file_path)

    return len(missing_files) == 0, missing_files

def run_command(cmd):
    """运行系统命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 70)
    print("🏭 工业级泵站数字孪生平台 - 系统检查与运行指南")
    print("=" * 70)

    all_good = True

    # 1. 检查Python版本
    print("\n🔍 [1/4] 检查Python环境...")
    if not check_python_version():
        all_good = False

    # 2. 测试依赖包
    print("\n📦 [2/4] 检查依赖包...")
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        all_good = False

    # 3. 检查文件完整性
    print("\n📁 [3/4] 检查项目文件...")
    files_ok, missing_files = check_files()
    if not files_ok:
        all_good = False

    # 4. 提供运行指导
    print("\n🚀 [4/4] 运行指导")
    print("-" * 50)

    if all_good:
        print("🎉 系统检查通过！可以正常运行")
        print("\n推荐运行方式：")
        print("1️⃣ 双击 '启动系统.bat' 文件（一键启动）")
        print("2️⃣ 或手动运行以下命令：")
        print("   python preprocessing.py    # 数据预处理")
        print("   python train.py           # 训练模型")
        print("   python -m streamlit run app.py  # 启动界面")
    else:
        print("⚠️  系统检查发现问题，需要修复：")
        print()

        if missing_deps:
            print("📦 缺失依赖包，运行以下命令安装：")
            print(f"pip install {' '.join(missing_deps)}")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            print()

        if missing_files:
            print("📁 缺失项目文件：")
            for file in missing_files:
                print(f"   • {file}")
            print("请确保所有文件都在项目目录中")
            print()

        print("🔧 修复完成后，重新运行此脚本检查")

    print("\n" + "=" * 70)
    print("💡 使用提示：")
    print("• 首次运行需要下载数据和训练模型，可能需要5-10分钟")
    print("• 训练完成后，模型会保存在 models/ 目录")
    print("• 如遇到内存不足，可以调整 train.py 中的 batch_size 参数")
    print("• 系统会在浏览器中自动打开可视化界面")
    print("=" * 70)

    # 等待用户输入
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()









