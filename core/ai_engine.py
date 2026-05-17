# core/ai_engine.py
# AI 引擎核心模块：LSTM 健康预测 + RAG 知识库 + ST-LLM 数据修复 + DeepSeek LLM Agent
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# 1. LSTM 健康预测模型 (与 train.py / app.py 保持完全一致)
# ==============================================================================
class IndustrialPumpLSTM(nn.Module):
    """双向双层 LSTM 设备健康预测模型"""
    def __init__(self, input_dim=51, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                             batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class AIPredictor:
    """封装好的 AI 推理器"""
    MODEL_PATH = 'models/pump_lstm_final.pth'
    SCALER_PATH = 'models/scaler.pkl'
    WINDOW_SIZE = 30

    def __init__(self):
        self.model = None
        self.scaler = None
        self._load()

    def _load(self):
        if not os.path.exists(self.MODEL_PATH) or not os.path.exists(self.SCALER_PATH):
            self.model = None
            self.scaler = None
            return
        try:
            self.scaler = joblib.load(self.SCALER_PATH)
            # 从归一化器动态获取特征维度（避免硬编码出错）
            input_dim = getattr(self.scaler, 'n_features_in_', 52)
            self.model = IndustrialPumpLSTM(input_dim=input_dim)
            state = torch.load(self.MODEL_PATH, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
        except Exception as e:
            # 打印详细错误，方便排查
            import traceback
            traceback.print_exc()
            self.model = None
            self.scaler = None

    def predict(self, sensor_data: np.ndarray) -> float:
        """
        给定一个时间窗口的传感器数据 (shape: [window_size, num_features]),
        返回健康评分 (0~1)，1=健康，0=故障。
        如果模型未加载，返回基于物理的启发式评分。
        """
        if self.model is None or self.scaler is None:
            return self._heuristic_score(sensor_data)

        try:
            # 归一化
            scaled = self.scaler.transform(sensor_data)
            # 适配模型输入维度 (需要 51 特征)
            if scaled.shape[1] < 51:
                pad = np.zeros((scaled.shape[0], 51 - scaled.shape[1]))
                scaled = np.hstack([scaled, pad])
            elif scaled.shape[1] > 51:
                scaled = scaled[:, :51]

            x = torch.FloatTensor(scaled).unsqueeze(0)  # [1, window, 51]
            with torch.no_grad():
                score = self.model(x).item()
        return score
        except Exception:
            return self._heuristic_score(sensor_data)

    def _heuristic_score(self, sensor_data: np.ndarray) -> float:
        """当模型未训练时，使用物理启发式评分"""
        if sensor_data.ndim == 2 and sensor_data.shape[0] > 0:
            latest = sensor_data[-1]
            # 振动异常检测 (sensor_04 通常是振动)
            vib = latest[4] if len(latest) > 4 else np.mean(latest)
            vib_score = max(0.0, 1.0 - vib / 5.0)
            # 温度异常检测 (sensor_01 通常是温度)
            temp = latest[1] if len(latest) > 1 else np.mean(latest)
            temp_score = max(0.0, 1.0 - abs(temp - 45) / 50)
            return (vib_score + temp_score) / 2.0
        return 0.85

    def predict_batch(self, df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
        """
        对整个 DataFrame 进行批量预测，返回带健康评分的 DataFrame。
        df 需要包含 sensor_00 ~ sensor_51 列。
        """
        feature_cols = [f'sensor_{i:02d}' for i in range(52)]
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            return df.assign(health_score=np.nan)

        scores = []
        data = df[available].values
        for i in range(len(data)):
            end = min(i + 1, len(data))
            start = max(0, end - window_size)
            window = data[start:end]
            if len(window) < window_size:
                pad = np.tile(window[-1:], (window_size - len(window), 1)) if len(window) > 0 else np.zeros((window_size, len(available)))
                window = np.vstack([pad, window]) if len(window) > 0 else pad
            scores.append(self.predict(window))
        result = df.copy()
        result['health_score'] = scores
        return result


# ==============================================================================
# 2. RAG 知识库问答系统
# ==============================================================================
class VectorRAGSystem:
    """基于 TF-IDF 的运维知识检索系统"""
    def __init__(self, kb_path: str = None):
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4), max_features=512
        )
        self.kb = []
        self.tfidf_matrix = None
        self.ready = False

        if kb_path and os.path.exists(kb_path):
            self._load_from_file(kb_path)
        else:
            self._init_default_kb()

    def _init_default_kb(self):
        """内存中的备用知识库（覆盖泵站常见运维问题）"""
        qa_map = [
            ("振动过大怎么排查",   "振动过大可能原因：①联轴器对中不良（频谱出现2X二倍频分量）—停机用激光对中仪检查；②气蚀（噼啪声+流量压力波动）—检查吸水井液位、清理滤网；③地脚螺栓松动。ISO-10816：<2.8mm/s正常，>4.5mm/s立即停机。"),
            ("振动异常怎么处理",   "振动异常处理：①检查联轴器对中（径向偏差<0.05mm）；②检查吸水井液位是否过低；③检查地脚螺栓紧固情况；④比对Q-H曲线看工况点是否偏离BEP过远。"),
            ("电机过热怎么办",     "电机过热（>85°C）处理：①检查风扇罩是否堵塞；②检查是否长期低频运行(<30Hz)，建议加装强迫风冷；③检查三相电流不平衡度(应<5%)；④>95°C须立即停机。"),
            ("轴承温度高",        "轴承温度>85°C报警，>95°C跳闸。立即检查：①润滑油脂是否充足；②轴承间隙是否正常；③端盖温度分布是否均匀；④必要时停机拆检。"),
            ("流量不足怎么办",    "流量不足排查：①叶轮口环磨损（大修周期12个月）；②变频器频率是否被调低；③止回阀是否卡在半开；④进水滤网是否堵塞。结合Q-H曲线判断工况点是否在高效区。"),
            ("汽蚀怎么解决",      "汽蚀解决措施：①检查吸水井液位(Sensor_LT是否<1.5m)；②清理进水滤网；③适当降低运行频率（如45Hz）减少必需汽蚀余量(NPSHr)；④检查泵安装高度是否过高。"),
            ("水锤怎么处理",      "水锤应急处理：①立即检查管道压力恢复情况；②检查阀门法兰有无泄漏；③长期预防：安装缓闭止回阀或压力罐，启停时缓慢调节阀门开度。"),
            ("变频器频率调节",    "变频器频率范围30Hz~50Hz，严禁长期<30Hz。频率从50Hz降至40Hz，功率可降低约48.8%（亲和定律：P∝n³）。最佳能效区间38~45Hz。"),
            ("节能优化建议",      "四大节能策略：①变频调节使工况点落在BEP高效区(85%~110%)；②夜间谷电蓄水，白天峰电时段利用重力供水；③定期清理叶轮和滤网；④避免长时间低效区运行。"),
            ("大马拉小车效率低",  "大马拉小车表现为效率低、振动大、能耗高。AI建议降低变频器频率（38~45Hz），使流量和扬程匹配实际需求，返回高效区运行。"),
            ("电机电流过大",      "电流过大可能原因：①泵体磨损导致负载增大；②出口阀开度过大；③电网电压偏低。建议检查工况点是否偏离BEP，并测量三相电流平衡度(应<5%)。"),
            ("开机前检查什么",    "开机前检查：①进水阀全开；②出水阀关闭或微开；③机械密封冲洗水正常；④盘车3-5圈无卡涩无异响。"),
            ("软启动流程步骤",   "软启动：①选AI智能托管或软启动模式；②变频器10s内从0Hz升至30Hz；③压力稳定后调频至目标值（如38.5Hz）；④缓慢开阀至100%。"),
            ("BEP最佳能效点",    "BEP是泵效率最高点，位于额定流量85%~110%范围内。建议通过变频调节使泵始终运行在BEP附近，偏离会导致效率下降、振动增大、轴承寿命缩短。"),
            ("多泵并联控制",     "多泵并联：当单泵>48Hz仍不能满足流量时自动启动备用泵。并联时两台泵频率同步误差须<0.5Hz，避免抢水导致其中一台进入死区。"),
            ("健康评分怎么看",    "Health Score：>85分优秀（绿），60~85分良好（黄），<60分预警（红）。低于60分建议预防性检修，避免非计划停机。"),
            ("数据缺失怎么处理", "传感器数据修复：缺失<5个点自动线性插值；>5个点多项式插值；突变超3σ判定噪声滤波处理。线路断开>1小时时推送人工巡检警报。"),
        ]
        self.kb = [(q, a) for q, a in qa_map if q.strip() and a.strip()]
        self._build_index()

    def _load_from_file(self, kb_path: str):
        """从泵运维手册提取问答对，优先精确匹配各类故障主题"""
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                raw = f.read()
            qa_pairs = self._extract_qa_pairs(raw)
            if qa_pairs:
                self.kb = qa_pairs
                self._build_index()
            else:
                self._init_default_kb()
        except Exception:
            self._init_default_kb()

    def _extract_qa_pairs(self, text: str) -> list:
        """解析 pump_manual.txt，提取问答对"""
        import re
        pairs = []

        def make_q(q_str: str, context: str) -> tuple:
            return (q_str.strip(), context.strip())

        # 按章节拆解，构建问答对
        qa_map = [
            ("振动过大怎么排查",   "振动过大可能原因：①联轴器对中不良（频谱出现2X二倍频分量）—停机用激光对中仪检查，径向偏差<0.05mm；②气蚀（流量压力波动大，有噼啪声）—检查吸水井液位、清理进水滤网；③地脚螺栓松动（1X转频为主，垂直振动明显）—紧固地脚螺栓。参考ISO-10816标准：<2.8mm/s正常，>4.5mm/s立即停机。"),
            ("机组振动大怎么办",   "振动过大可能原因：①联轴器对中不良（频谱出现2X二倍频分量）—停机用激光对中仪检查，径向偏差<0.05mm；②气蚀（流量压力波动大，有噼啪声）—检查吸水井液位、清理进水滤网；③地脚螺栓松动（1X转频为主，垂直振动明显）—紧固地脚螺栓。参考ISO-10816标准：<2.8mm/s正常，>4.5mm/s立即停机。"),
            ("振动异常",           "振动异常可能原因：①联轴器对中不良（频谱出现2X二倍频分量）；②气蚀（噼啪声+流量压力波动）；③地脚螺栓松动（垂直振动>水平振动）。ISO-10816振动标准：<2.8mm/s优，2.8~4.5mm/s良/关注，>4.5mm/s报警/停机。"),
            ("振动速度报警",       "振动速度超过2.8mm/s触发报警，>4.5mm/s须立即停机。可能原因：①对中不良；②气蚀；③地脚松动。停机检查联轴器对中、吸水井液位及地脚螺栓。"),
            ("电机过热怎么处理",   "电机过热（Sensor-07>85°C）处理步骤：①检查电机风扇罩是否被异物堵塞；②检查是否长期低频运行(<30Hz)，建议加装强迫风冷；③检查三相电流不平衡度，若>5%检查供电电压及接线端子；④轴承温度超过95°C须立即停机。"),
            ("轴承温度高",        "轴承温度>85°C为报警阈值，>95°C为跳闸阈值。处理措施：①检查润滑油脂情况，补充或更换润滑脂；②检查轴承间隙；③检查轴承端盖温度分布是否均匀；④必要时停机拆检。"),
            ("流量不足",          "流量不足可能原因：①叶轮口环磨损（大修周期12个月）；②变频器频率设定值被误修改；③止回阀卡在半开位置；④进水滤网堵塞。建议逐项排查，并结合Q-H曲线判断当前工况点是否在高效区。"),
            ("流量太小",          "流量不足可能原因：①叶轮口环磨损；②变频器频率被调低；③止回阀卡阻；④进水滤网堵塞。结合Q-H曲线判断当前工况点位置，正常高效区应在BEP的85%~110%范围内。"),
            ("汽蚀怎么办",        "汽蚀会导致叶轮腐蚀和效率急剧下降，特征为噼啪声和流量压力波动。处理：①检查吸水井液位(Sensor_LT是否<1.5m)；②清理进水滤网堵塞物；③适当降低运行频率（如降至45Hz）以减少必需汽蚀余量(NPSHr)；④检查安装高度是否过高。"),
            ("汽蚀现象",          "汽蚀诊断特征：①泵体发出类似爆豆子或碎石撞击的噼啪声；②流量和压力大幅波动；③效率急剧下降。物理原因：吸入口压力低于液体饱和蒸汽压。处理：检查液位、清理滤网、降低频率或提高入口压力。"),
            ("水锤预防",          "水锤会产生压力冲击，可能损坏管道和阀门。预防措施：①安装缓闭止回阀或压力罐；②启停时缓慢调节阀门开度；③设置合理的阀门关闭时间程序。"),
            ("水锤发生",          "水锤处理：①立即检查管道压力是否恢复正常；②检查阀门和法兰是否有泄漏；③联系钳工检查止回阀是否正常关闭；④长期预防：安装缓闭止回阀或压力罐，启停时缓慢调节阀门开度。"),
            ("变频器频率调节",     "变频器频率建议控制在30Hz~50Hz之间。低于30Hz可能导致电机散热不良，严禁长期低频运行。根据亲和定律：Q∝n，H∝n²，P∝n³，最佳能效区间为38~45Hz，此时理论功率可比50Hz降低约30%。"),
            ("频率调节建议",       "变频器频率建议：30Hz（最低运行频率，严禁长期）~50Hz（额定）。根据亲和定律降低频率可显著节能，40Hz时功率可降低约48.8%。夜间低谷时段可适当提高频率蓄水，白天峰电时段维持低频待机。"),
            ("如何节能优化",      "节能优化四大策略：①根据实际流量需求调节频率，工况点落在BEP高效区(85%~110%)；②避免长时间低效区运行；③夜间谷电时段(22:00-06:00)全速蓄水，白天峰电时段(08:00-11:00)利用重力供水；④定期维护清理叶轮和滤网。"),
            ("节能方法",          "最有效的节能手段：降低变频器频率。根据亲和定律，将频率从50Hz降至40Hz，功率可降低约48.8%。此外：削峰填谷（夜间蓄水白天用）、工况点落在BEP高效区、定期清理叶轮。"),
            ("大马拉小车效率低",   "大马拉小车指泵选型过大，运行工况点长期偏离BEP最佳能效点。表现为效率低、振动大、能耗高。AI建议：适当降低变频器频率（38~45Hz），使流量和扬程匹配实际需求，返回高效区运行。"),
            ("电机电流过大",       "电机电流超过额定值可能原因：①泵体磨损导致负载增大；②出口阀门开度过大；③电网电压偏低。建议检查泵运行工况点是否偏离BEP，并测量三相电流平衡度（不平衡度应<5%）。"),
            ("电流不平衡",         "三相电流不平衡度>5%时应检查：①供电电压是否平衡；②接线端子是否松动或氧化；③电机绕组是否损坏。可用钳形电流表逐相测量对比。"),
            ("开机前检查",         "开机前检查清单：①进水阀门是否完全打开（必须常开）；②出水阀门是否处于关闭或微开状态（防止过载启动）；③机械密封冲洗水是否正常流动；④盘车检查：手动盘动联轴器3-5圈，应无卡涩、无异响。"),
            ("软启动流程",         "软启动标准流程：①选择AI智能托管或软启动模式；②变频器以10s斜坡时间从0Hz升至30Hz最低频率；③待出口压力稳定后，调节频率至目标值（如38.5Hz）；④缓慢打开出口阀门至100%开度。"),
            ("启动步骤",           "启动步骤：①确认进水阀门全开、出水阀微开；②盘车3-5圈无卡涩；③在PLC/上位机选择运行模式，变频器软启动至30Hz；④待压力稳定后调频至目标值，缓慢开阀至100%。"),
            ("停机步骤",           "停机注意事项：①正常停机前先关闭出口阀，避免水锤；②紧急停机时立即断电，系统会自动联锁关闭进出水阀；③停机后检查泵内是否积水，北方地区注意防冻。"),
            ("bep最佳能效点",      "最佳能效点(BEP)是泵效率最高的运行点，位于额定流量的85%~110%范围内。本系统通过Q-H曲线实时显示BEP位置，建议使工况点尽量接近BEP以实现最大节能。偏离BEP会导致效率下降、振动增大、轴承寿命缩短。"),
            ("bep是什么",          "BEP(Best Efficiency Point，最佳能效点)是泵效率最高的运行工况。BEP附近运行时轴功率最小、振动最低、轴承寿命最长。建议通过变频调节使泵始终运行在BEP的85%~110%范围内。"),
            ("多泵并联控制",       "多泵并联策略：当单泵频率>48Hz仍无法满足流量时，自动启动备用泵。并联运行时，两台泵频率同步误差须<0.5Hz，避免抢水现象导致其中一台泵进入死区、低效运行。"),
            ("双泵并联",           "双泵并联注意事项：①保持两台泵频率同步（误差<0.5Hz）；②避免抢水：出水汇管阻力不能过大；③定期检查阀门状态，确保并联切换逻辑正常。"),
            ("ST-LLM数据修复",     "ST-LLM数据修复模块针对工业现场传感器信号丢失、尖峰噪声进行实时修复。触发条件：连续缺失5个采样点，或数值突变超过3σ标准差。局限性：物理线路断开超过1小时，修复置信度下降，此时推送人工巡检警报。"),
            ("数据缺失修复",       "传感器数据修复策略：①连续缺失<5个点时自动线性插值+前后向填充；②缺失>5个点时多项式插值；③突变超3σ时判定为噪声，使用滤波替代。修复置信度低时会提示人工巡检。"),
            ("预测性维护",         "本系统LSTM模型基于过去30天振动数据预测未来24小时健康趋势。当Health Score<60分时，建议立即安排预防性检修，避免非计划停机。评分>85分为优秀，60~85分为良好。"),
            ("健康评分怎么看",      "Health Score评分标准：>85分=优秀（绿色），60~85分=良好（黄色），<60分=预警（红色）。评分基于振动、温度、流量、压力多维特征，由LSTM深度学习模型计算。低于60分时系统会推送预防性检修建议。"),
            ("LSTM预测模型",       "LSTM预测性维护：系统基于过去30天振动、温度等传感器数据，通过双层双向LSTM深度学习模型预测未来24小时设备健康趋势。评分<60分时建议预防性检修，避免非计划停机造成经济损失。"),
        ]
        for q, answer in qa_map:
            if q.strip() and answer.strip():
                pairs.append((q, answer))
        return pairs

    def _build_index(self):
        try:
            questions = [q for q, _ in self.kb]
            self.tfidf_matrix = self.vectorizer.fit_transform(questions)
            self.ready = True
        except Exception:
            self.ready = False

    def query(self, question: str) -> str:
        if not self.ready:
            return "⚠️ NLP 组件未就绪，知识库加载失败。"
        try:
            vec = self.vectorizer.transform([question])
            sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
            idx = int(np.argmax(sims))
            if sims[idx] > 0.05:
                return self.kb[idx][1]
            return "🤖 知识库中未检索到与您问题相关的条目，请尝试换一种表述方式。"
        except Exception as e:
            return f"⚠️ 检索过程出错: {e}"


# ==============================================================================
# 3. ST-LLM 时空数据修复器
# ==============================================================================
class ST_LLM_Imputer:
    """
    时空数据插补器
    结合前向填充、后向填充、多项式插值三种策略，
    模拟 ST-LLM (Spatio-Temporal Large Language Model) 的时空相关性插补能力。
    """
    def repair(self, data) -> pd.DataFrame:
        """
        对包含缺失值的数据进行智能插补。
        支持 DataFrame、Series 或 np.ndarray。
        """
        if data is None:
            return pd.DataFrame()

        # 统一转为 DataFrame
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.Series):
            df = data.to_frame()
        else:
            df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        if df.empty or df.isnull().sum().sum() == 0:
            return df

        # 策略1: 对有序时序列使用线性插值
        df = df.interpolate(method='linear', limit_direction='both')
        # 策略2: 前向填充处理头部缺失
        df = df.ffill()
        # 策略3: 后向填充处理尾部缺失
        df = df.bfill()
        # 策略4: 对仍有缺失的列使用多项式插值(仅对数值列)
        for col in df.columns:
            if df[col].isnull().any():
                try:
                    df[col] = df[col].interpolate(method='polynomial', order=2, limit_direction='both')
                    df[col] = df[col].ffill().bfill()
                except Exception:
                    df[col] = df[col].fillna(df[col].median() if not df[col].dropna().empty else 0)

        return df

    def get_repair_report(self, before: pd.DataFrame, after: pd.DataFrame) -> dict:
        """生成数据修复报告"""
        before_missing = before.isnull().sum().sum()
        after_missing = after.isnull().sum().sum()
        return {
            "修复前缺失值总数": int(before_missing),
            "修复后缺失值总数": int(after_missing),
            "修复率": f"{(1 - after_missing / max(before_missing, 1)) * 100:.1f}%",
            "修复列数": int((before.isnull().sum() > 0).sum()),
        }


# ==============================================================================
# 4. DeepSeek LLM 智能体（知识库增强回答，支持流式输出）
# ==============================================================================
class DeepSeekAgent:
    """基于 DeepSeek-V4 的泵站运维智能体"""

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL = "deepseek-ai/deepseek-v4-pro"

    SYSTEM_PROMPT = """你是一位专业的泵站运维工程师，服务于「智水先知 AI 泵站调度系统」。
你具备深厚的水泵、变频器、PLC、传感器等工业设备知识，熟悉泵站日常运维、故障诊断、能效优化。

回答规范：
1. 专业、简洁、可操作，不要废话和套话
2. 优先引用泵站运维知识（系统内已提供），结合实际情况给出具体建议
3. 涉及安全阈值（振动>4.5mm/s停机、轴承温度>95°C跳闸等）务必强调
4. 只回答泵站/工业设备运维相关问题；与主题无关的问题请礼貌拒绝
5. 回答使用中文，遇到专业术语可附英文缩写
"""

    def __init__(self, api_key: str, rag_system=None, timeout: int = 120):
        self.api_key = api_key
        self.rag_system = rag_system
        self.timeout = timeout
        self._available = None  # 延迟检测

    @property
    def available(self) -> bool:
        """延迟检测：首次查询时才检测 API 是否可用"""
        if self._available is None:
            self._available = self._health_check()
        return self._available

    def _health_check(self) -> bool:
        """快速健康检查：发一个最小请求验证 key 是否有效"""
        try:
            resp = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                    "extra_body": {"chat_template_kwargs": {"thinking": False}}
                },
                timeout=10
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _build_messages(self, user_query: str) -> list:
        """构建带知识库上下文的 messages"""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # 如果有 RAG 知识库，先检索相关条目作为上下文
        if self.rag_system and self.rag_system.ready:
            try:
                vec = self.rag_system.vectorizer.transform([user_query])
                sims = self.rag_system.tfidf_matrix.dot(vec.T).T.toarray().flatten()
                top_k = min(3, len(self.rag_system.kb))
                top_idx = np.argsort(sims)[-top_k:][::-1]
                if sims[top_idx[0]] > 0.01:
                    kb_context = "【参考知识库】：\n" + "\n".join(
                        f"[{i+1}] {self.rag_system.kb[idx][1]}"
                        for i, idx in enumerate(top_idx)
                    )
                    messages.append({
                        "role": "user",
                        "content": f"请参考以下运维知识回答问题：\n{kb_context}\n\n用户问题：{user_query}"
                    })
                    return messages
            except Exception:
                pass

        messages.append({"role": "user", "content": user_query})
        return messages

    def chat(self, user_query: str, stream: bool = False):
        """
        发送对话请求。
        stream=True 时返回 requests.Response（由调用方迭代），用于流式输出。
        stream=False 时返回完整字符串（更可靠）。
        """
        if not self.available:
            return None

        messages = self._build_messages(user_query)
        try:
            resp = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "stream": stream,
                    "extra_body": {"chat_template_kwargs": {"thinking": False}}
                },
                stream=stream,
                timeout=self.timeout
            )
            return resp
        except Exception as e:
            print(f"[DeepSeekAgent] 请求失败: {e}")
            return None

