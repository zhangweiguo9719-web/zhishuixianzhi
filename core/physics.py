# core/physics.py
# 泵站物理特性模型：基于泵相似定律 (Affinity Laws) 的水力计算
import numpy as np


class PumpPhysicsModel:
    """
    水泵物理特性模型

    核心原理 — 泵相似定律 (Affinity Laws):
      Q ∝ N (流量与转速成正比)
      H ∝ N² (扬程与转速平方成正比)
      P ∝ N³ (功率与转速立方成正比)

    Q-H 曲线方程: H = H_shutoff - k * Q²
    其中 H_shutoff = H_额 * (f/50)²
    """

    # 额定工况参数 (对应 50Hz)
    RATED_FLOW = 50.0       # m³/h，额定流量
    RATED_HEAD = 32.0       # m，额定扬程
    RATED_SPEED = 2900       # rpm，额定转速
    RATED_POWER = 7.5       # kW，电机额定功率
    MOTOR_EFF = 0.94        # 电机效率
    PF = 0.88               # 功率因数
    RATED_CURRENT = 14.5    # A，额定电流

    def __init__(self, rated_flow=None, rated_head=None, rated_power=None):
        if rated_flow is not None:
            self.RATED_FLOW = rated_flow
        if rated_head is not None:
            self.RATED_HEAD = rated_head
        if rated_power is not None:
            self.RATED_POWER = rated_power

    def get_pump_curve(self, freq_hz: float, n_points: int = 100) -> tuple:
        """
        获取指定频率下的泵 Q-H 特性曲线

        参数:
            freq_hz: 变频器频率 (Hz)
            n_points: 曲线采样点数

        返回:
            (Q_range, H_curve, eta_curve) — 流量数组、扬程数组、效率数组
        """
        ratio = freq_hz / 50.0

        # 额定参数按相似定律缩放
        Q_max = self.RATED_FLOW * ratio * 1.4     # 最大流量 (略超额定)
        Q_range = np.linspace(0, Q_max, n_points)

        # Q-H 曲线: H = H_shutoff - k * Q²
        # 闭阀扬程 (H_shutoff ≈ 1.2 * RATED_HEAD)
        H_shutoff = 1.2 * self.RATED_HEAD * (ratio ** 2)
        # 系数 k 由额定工况点确定: H_rated = H_shutoff - k * Q_rated²
        k_h = (H_shutoff - self.RATED_HEAD) / (self.RATED_FLOW ** 2)
        H_curve = H_shutoff - k_h * (Q_range ** 2)
        H_curve = np.maximum(H_curve, 0)  # 扬程不能为负

        # 效率曲线: 抛物线型，最高效率点位于 BEP (0.85~1.1 * Q_rated)
        Q_bep = self.RATED_FLOW * ratio
        eta_max = 0.88
        # 效率曲线以 BEP 为中心
        sigma = (Q_bep * 0.8) ** 2
        Eta_curve = eta_max * np.exp(-((Q_range - Q_bep) ** 2) / sigma)
        Eta_curve = np.maximum(Eta_curve, 0.1)

        return Q_range, H_curve, Eta_curve

    def get_system_curve(self, static_head: float = 25.0, resistance: float = 0.0001) -> tuple:
        """
        获取管网阻力曲线: H = H_static + S * Q²

        参数:
            static_head: 静扬程 (m)，即高程差 + 压力差
            resistance: 管网阻力系数 S

        返回:
            (Q_range, H_system) — 流量数组、管网扬程数组
        """
        Q_range = np.linspace(0, self.RATED_FLOW * 1.5, 100)
        H_system = static_head + resistance * (Q_range ** 2)
        return Q_range, H_system

    def affinity_laws(self, old_freq: float, new_freq: float,
                      old_flow=None, old_head=None, old_power=None) -> dict:
        """
        相似定律换算: 从旧频率工况推算新频率下的参数

        参数:
            old_freq, new_freq: 旧/新频率 (Hz)
            old_flow, old_head, old_power: 旧频率下的已知参数

        返回:
            dict，包含新频率下的 Q, H, P
        """
        ratio = new_freq / old_freq
        result = {}
        if old_flow is not None:
            result['flow'] = old_flow * ratio
        if old_head is not None:
            result['head'] = old_head * (ratio ** 2)
        if old_power is not None:
            result['power'] = old_power * (ratio ** 3)
        return result

    def calc_working_point(self, freq_hz: float,
                            static_head: float = 25.0,
                            resistance: float = 0.0001) -> dict:
        """
        计算泵在指定频率和管网阻力下的工作（工况）点

        返回:
            dict: {Q, H, eta, p_shaft, current, load_factor}
        """
        ratio = freq_hz / 50.0
        Q_opt = self.RATED_FLOW * ratio
        H_shutoff = 1.2 * self.RATED_HEAD * (ratio ** 2)
        k_h = (H_shutoff - self.RATED_HEAD) / (self.RATED_FLOW ** 2)

        # 迭代求解: pump_H(Q) = system_H(Q)
        # H_shutoff - k_h * Q² = static_head + resistance * Q²
        k_total = k_h + resistance
        if k_total <= 0:
            Q = Q_opt
        else:
            Q = np.sqrt(max(H_shutoff - static_head, 0) / k_total)

        H = H_shutoff - k_h * (Q ** 2)
        H = max(H, 0)

        # 效率
        Q_bep = self.RATED_FLOW * ratio
        eta = 0.88 * np.exp(-((Q - Q_bep) / (Q_bep * 0.8)) ** 2)
        eta = max(eta, 0.4)

        # 轴功率
        p_water = (998 * 9.81 * (Q / 3600) * H) / 1000
        p_shaft = p_water / eta if eta > 0 else p_water

        # 电流
        current = (p_shaft * 1000) / (1.732 * 380 * self.PF * self.MOTOR_EFF)
        load_factor = p_shaft / self.RATED_POWER

        return {
            'flow': float(Q),
            'head': float(H),
            'efficiency': float(eta * 100),
            'power_shaft': float(p_shaft),
            'current': float(current),
            'load_factor': float(load_factor),
        }

    def estimate_bep(self, freq_hz: float) -> dict:
        """估算最佳能效点 (BEP)"""
        ratio = freq_hz / 50.0
        return {
            'flow': self.RATED_FLOW * ratio,
            'head': self.RATED_HEAD * (ratio ** 2),
            'efficiency': 88.0,
            'power': self.RATED_POWER * (ratio ** 3),
        }
