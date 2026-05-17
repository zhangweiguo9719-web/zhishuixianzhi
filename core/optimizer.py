# core/optimizer.py
# 系统匹配寻优引擎：根据泵-管网匹配特性给出最优运行建议
import numpy as np
from core.physics import PumpPhysicsModel


class SystemMatcher:
    """
    泵-管网系统匹配诊断与优化建议引擎

    诊断逻辑:
    - 根据当前流量 Q 判断偏工况程度
    - 结合效率给出维护/优化建议
    - 推荐最优运行频率
    """

    def __init__(self):
        self.physics = PumpPhysicsModel()

    def analyze(self, real_flow: float, real_head: float) -> dict:
        """
        分析泵-管网系统匹配状态

        参数:
            real_flow: 当前流量 (m³/h)
            real_head: 当前扬程 (m)

        返回:
            dict: {
                'optimal_freq',     # 推荐频率 (Hz)
                'optimal_flow',     # 推荐流量 (m³/h)
                'diagnosis',        # 诊断结论
                'resistance_S',     # 管网阻力系数
                'color',            # 状态颜色 (HTML色值)
                'efficiency_loss',  # 效率损失百分比
                'maintenance_tips', # 维护建议
                'severity'          # 严重程度: normal/warning/critical
            }
        """
        H_static = 25.0
        # 管网阻力系数 S: H = H_static + S * Q²  =>  S = (H - H_static) / Q²
        if real_flow > 1:
            S_curr = (max(real_head - H_static, 0)) / (real_flow ** 2)
        else:
            S_curr = 0.0001

        # 根据流量区间判断偏工况程度
        # 最佳流量区间 (BEP 范围): 额定流量 * (0.7 ~ 1.1)
        Q_bep = self.physics.RATED_FLOW
        Q_low = Q_bep * 0.70
        Q_high = Q_bep * 1.10
        Q_warn_low = Q_bep * 0.50
        Q_warn_high = Q_bep * 1.30

        if real_flow < Q_warn_low:
            # 严重偏工况（大马拉小车）
            severity = "critical"
            opt_freq = 38.5
            opt_Q = Q_bep * 0.85
            diagnosis = "⚠️ 严重偏工况（大马拉小车）"
            color = "#d32f2f"
            eff_loss = max(0, (1 - real_flow / Q_bep) * 40)
            tips = "立即降低频率至38Hz，减少阀门开度或增开并联泵。建议检查实际用水需求，避免长期低负荷运行。"
        elif real_flow < Q_low:
            # 轻度过低
            severity = "warning"
            opt_freq = 42.0
            opt_Q = Q_bep * 0.85
            diagnosis = "⚠️ 轻度过载运行，效率偏低"
            color = "#f57c00"
            eff_loss = max(0, (1 - real_flow / Q_bep) * 20)
            tips = "建议适当降低频率至42Hz，若夜间流量持续偏低可考虑蓄水策略。"
        elif real_flow > Q_warn_high:
            # 过载风险
            severity = "critical"
            opt_freq = 44.0
            opt_Q = Q_bep * 0.95
            diagnosis = "⚠️ 接近汽蚀风险区（过载）"
            color = "#f57c00"
            eff_loss = max(0, (real_flow / Q_bep - 1) * 25)
            tips = "关注入口压力，防止汽蚀。建议检查进口滤网是否堵塞，适当关小出口阀调节流量。"
        elif real_flow > Q_high:
            # 轻度过高
            severity = "warning"
            opt_freq = 46.0
            opt_Q = Q_bep * 1.0
            diagnosis = "⚠️ 略超高效区上限"
            color = "#faad14"
            eff_loss = max(0, (real_flow / Q_bep - 1) * 10)
            tips = "流量略超BEP上限，当前效率尚可，但需关注轴承温升和振动趋势。"
        else:
            # 匹配良好
            severity = "normal"
            opt_freq = round(max(38.0, min(50.0, 50.0 * (real_flow / Q_bep))), 1)
            opt_Q = real_flow
            diagnosis = "✅ 工况匹配良好（BEP追踪中）"
            color = "#2e7d32"
            eff_loss = 0.0
            tips = "当前运行在高效区。继续保持，定期监测振动和轴承温度。"

        # 估算推荐频率下的能效提升
        try:
            wp_current = self.physics.calc_working_point(50.0, H_static, S_curr)
            wp_optimal = self.physics.calc_working_point(opt_freq, H_static, S_curr)
            power_save = max(0, wp_current['power_shaft'] - wp_optimal['power_shaft'])
        except Exception:
            power_save = 0.0

        return {
            'optimal_freq': opt_freq,
            'optimal_flow': round(opt_Q, 1),
            'diagnosis': diagnosis,
            'resistance_S': round(S_curr, 8),
            'color': color,
            'efficiency_loss': round(eff_loss, 1),
            'maintenance_tips': tips,
            'severity': severity,
            'power_save_kw': round(power_save, 2),
        }
