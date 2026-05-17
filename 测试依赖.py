#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智水先知 - 依赖测试脚本
检查所有必要的Python包是否正确安装
"""

import sys
import platform

def test_import(module_name, display_name=None):
    """测试导入模块"""
    if display_name is None:
        display_name = module_name

    try:
        __import__(module_name)
        print(f"✅ {display_name} - 已安装")
        return True
    except ImportError as e:
        print(f"❌ {display_name} - 未安装 ({e})")
        return False

def main():
    print("=" * 50)
    print("🌊 智水先知 - 系统依赖检查")
    print("=" * 50)
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print()

    # 必需的依赖包
    required_packages = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('torch', 'PyTorch'),
        ('plotly', 'Plotly'),
    ]

    print("检查核心依赖包:")
    print("-" * 30)

    all_good = True
    for module, display in required_packages:
        if not test_import(module, display):
            all_good = False

    print()
    print("-" * 50)

    if all_good:
        print("🎉 所有依赖包都已正确安装！")
        print()
        print("现在可以运行系统了:")
        print("1. 双击 '启动系统.bat' 文件")
        print("2. 或者在命令行运行: python -m streamlit run app.py")
        print()
        print("系统将在浏览器中自动打开 ✨")
    else:
        print("⚠️  部分依赖包未安装")
        print()
        print("请运行以下命令安装缺失的包:")
        print("pip install streamlit pandas numpy scikit-learn matplotlib")
        print()
        print("或者运行: pip install -r requirements.txt")

    print("-" * 50)
    input("按回车键退出...")

if __name__ == "__main__":
    main()









