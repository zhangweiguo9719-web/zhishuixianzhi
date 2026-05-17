#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业级泵站数字孪生平台 - 自动安装依赖包

自动检测并安装缺失的Python依赖包
"""

import subprocess
import sys
import os

def run_command(cmd):
    """运行系统命令并返回结果"""
    try:
        print(f"🔄 执行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 命令执行成功")
            return True, result.stdout
        else:
            print(f"❌ 命令执行失败 (错误码: {result.returncode})")
            print(f"错误信息: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ 命令执行异常: {e}")
        return False, str(e)

def check_and_install_package(package_name, pip_name=None):
    """检查并安装单个包"""
    if pip_name is None:
        pip_name = package_name

    try:
        __import__(package_name)
        print(f"✅ {package_name} - 已安装")
        return True
    except ImportError:
        print(f"📦 {package_name} - 需要安装")
        success, output = run_command(f"pip install {pip_name}")
        if success:
            # 验证安装
            try:
                __import__(package_name)
                print(f"✅ {package_name} - 安装成功")
                return True
            except ImportError:
                print(f"❌ {package_name} - 安装失败")
                return False
        return success

def main():
    print("=" * 60)
    print("📦 工业级泵站数字孪生平台 - 依赖包安装器")
    print("=" * 60)

    # 检查Python版本
    version = sys.version_info
    print(f"🐍 当前Python版本: {version.major}.{version.minor}.{version.micro}")
    if not (version.major >= 3 and version.minor >= 9):
        print("❌ 需要Python 3.9或更高版本")
        return False

    print("\n🔍 开始安装依赖包...")
    print("-" * 40)

    # 核心科学计算包
    core_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly'),
        ('scikit-learn', 'scikit-learn'),
        ('joblib', 'joblib'),
        ('tqdm', 'tqdm'),
    ]

    # PyTorch深度学习框架
    pytorch_packages = [
        ('torch', 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'),
    ]

    # 可视化和工具包
    ui_packages = [
        ('streamlit', 'streamlit'),
    ]

    all_success = True

    # 安装核心包
    print("📊 安装核心科学计算包...")
    for package, pip_name in core_packages:
        if not check_and_install_package(package, pip_name):
            all_success = False

    print("\n🧠 安装PyTorch深度学习框架...")
    for package, pip_name in pytorch_packages:
        if not check_and_install_package(package, pip_name):
            all_success = False

    print("\n🎛️ 安装可视化界面包...")
    for package, pip_name in ui_packages:
        if not check_and_install_package(package, pip_name):
            all_success = False

    print("\n" + "=" * 60)

    if all_success:
        print("🎉 所有依赖包安装完成！")
        print("\n🚀 下一步操作:")
        print("1. 运行: python 运行指南.py  (验证安装)")
        print("2. 运行: python preprocessing.py  (数据预处理)")
        print("3. 运行: python train.py  (深度学习训练)")
        print("4. 运行: python -m streamlit run app.py  (启动可视化界面)")
        print("\n💡 提示: 或者直接双击 '启动系统.bat' 一键启动")
    else:
        print("⚠️  部分依赖包安装失败")
        print("\n🔧 故障排除:")
        print("1. 检查网络连接")
        print("2. 尝试使用国内镜像:")
        print("   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名")
        print("3. 或使用conda安装:")
        print("   conda install 包名")
        print("4. 重新运行此脚本")

    print("=" * 60)
    input("\n按回车键退出...")

    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)









