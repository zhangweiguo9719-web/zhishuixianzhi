#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装和配置中文字体支持
解决matplotlib和plotly中文显示方框问题
"""

import os
import sys
import matplotlib
import matplotlib.font_manager as fm

def check_chinese_fonts():
    """检查系统中可用的中文字体"""
    print("=" * 60)
    print("🔍 检查系统中文字体支持")
    print("=" * 60)
    
    # 获取所有可用字体
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    # 常见中文字体名称
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',           # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'STHeiti',          # 华文黑体
        'STSong',           # 华文宋体
        'Arial Unicode MS', # Arial Unicode MS
    ]
    
    print("\n📋 系统中可用的中文字体:")
    found_fonts = []
    for font in chinese_fonts:
        if font in font_list:
            print(f"  ✅ {font}")
            found_fonts.append(font)
        else:
            print(f"  ❌ {font} (未找到)")
    
    if found_fonts:
        print(f"\n✅ 找到 {len(found_fonts)} 个中文字体")
        print(f"📝 推荐使用: {found_fonts[0]}")
        return found_fonts[0]
    else:
        print("\n⚠️  未找到中文字体，将使用默认字体")
        return None

def configure_matplotlib_font(font_name=None):
    """配置matplotlib使用中文字体"""
    print("\n" + "=" * 60)
    print("⚙️  配置matplotlib中文字体")
    print("=" * 60)
    
    if font_name:
        matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'sans-serif']
        print(f"✅ 已设置matplotlib字体为: {font_name}")
    else:
        # 尝试Windows常见字体路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',      # 黑体
            'C:/Windows/Fonts/msyh.ttc',       # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',     # 宋体
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    fm.fontManager.addfont(font_path)
                    font_name = os.path.basename(font_path).split('.')[0]
                    matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'sans-serif']
                    print(f"✅ 已从 {font_path} 加载字体")
                    break
                except Exception as e:
                    print(f"⚠️  加载字体失败: {e}")
        
        if not matplotlib.rcParams['font.sans-serif'][0]:
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
            print("⚠️  使用默认字体")
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("✅ 已禁用负号Unicode显示")
    
    return matplotlib.rcParams['font.sans-serif'][0]

def test_chinese_display():
    """测试中文显示"""
    print("\n" + "=" * 60)
    print("🧪 测试中文显示")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        ax.plot(x, y, label='正弦波')
        ax.set_title('中文字体测试 - 工业级泵站监控系统', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        test_file = 'test_chinese_font.png'
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_file):
            print(f"✅ 测试图表已保存: {test_file}")
            print("📊 请打开图片检查中文是否正确显示")
            return True
        else:
            print("❌ 测试图表保存失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("=" * 60)
    print("🎨 工业级泵站系统 - 中文字体配置工具")
    print("=" * 60)
    
    # 1. 检查字体
    font_name = check_chinese_fonts()
    
    # 2. 配置matplotlib
    configured_font = configure_matplotlib_font(font_name)
    
    # 3. 测试显示
    test_success = test_chinese_display()
    
    print("\n" + "=" * 60)
    if test_success:
        print("🎉 中文字体配置完成！")
        print(f"📝 当前使用字体: {configured_font}")
        print("\n💡 提示:")
        print("1. 如果中文仍显示为方框，请重启Streamlit应用")
        print("2. 确保系统已安装中文字体（Windows通常自带）")
        print("3. 如果问题持续，请检查系统字体设置")
    else:
        print("⚠️  配置完成，但测试失败")
        print("💡 建议:")
        print("1. 检查系统是否安装了中文字体")
        print("2. 尝试重启Python环境")
        print("3. 查看错误信息并手动配置")
    
    print("=" * 60)
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()









