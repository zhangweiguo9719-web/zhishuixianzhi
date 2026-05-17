"""
=============================================================================
脚本名称: train.py (工业级训练流程)
功能: 读取 real_pump_data.csv -> 训练 LSTM -> 生成 .pth 和 .pkl
=============================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import os
import sys

# --- 配置参数 ---
WINDOW_SIZE = 30       # 时间窗口
BATCH_SIZE = 512       # 批次大小
EPOCHS = 5             # 训练轮数 (演示用5轮，实际可更多)
LR = 0.001             # 学习率

# --- 1. 模型定义 (必须与 App 里的推理逻辑一致) ---
class IndustrialPumpLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(IndustrialPumpLSTM, self).__init__()
        # 双向 LSTM 提取时序特征
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        # 全连接层输出故障概率
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, window, features)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出: out[:, -1, :]
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def train_pipeline():
    print("\n" + "="*50)
    print("🚀 启动 AI 模型训练管线 (Training Pipeline)")
    print("="*50)

    # 1. 检查数据
    data_path = 'data/real_pump_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到 {data_path}。请先运行 forge_big_data.py")
        return

    print(f"📂 [1/5] 读取数据集: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   -> 数据量: {len(df):,} 行")

    # 2. 数据预处理
    print("⚙️ [2/5] 特征工程与归一化...")
    # 标签映射: NORMAL=1 (健康), BROKEN=0 (故障), RECOVERING=0.5 (恢复中)
    status_map = {'NORMAL': 1.0, 'BROKEN': 0.0, 'RECOVERING': 0.5}
    df['label'] = df['machine_status'].map(status_map)
    
    # 选取传感器特征 (排除 timestamp, machine_status, label)
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'machine_status', 'label']]
    print(f"   -> 特征数量: {len(feature_cols)} 个 (Sensor_00...Sensor_51)")
    
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[feature_cols].values)
    
    # 保存归一化器 (非常重要！App 需要用它来缩放实时数据)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("   -> 归一化器已保存: models/scaler.pkl")

    # 3. 制作时间序列
    print(f"✂️ [3/5] 构建时间序列窗口 (Window={WINDOW_SIZE})...")
    # 为了快速演示，我们使用步长采样，避免数据量过大撑爆内存
    stride = 5 
    X, y = [], []
    for i in range(0, len(data_scaled) - WINDOW_SIZE, stride):
        X.append(data_scaled[i : i+WINDOW_SIZE])
        y.append(df['label'].values[i+WINDOW_SIZE])
    
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).unsqueeze(1)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"   -> 训练样本数: {len(X):,}")

    # 4. 初始化模型
    print("🧠 [4/5] 初始化 LSTM 神经网络...")
    model = IndustrialPumpLSTM(input_dim=len(feature_cols))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5. 训练循环
    print("🔥 [5/5] 开始训练...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {(1-avg_loss)*100:.1f}%")

    # 6. 保存模型权重
    torch.save(model.state_dict(), 'models/pump_lstm_final.pth')
    print("\n" + "="*50)
    print("✅ 训练完成！大脑已就绪。")
    print("💾 模型保存至: models/pump_lstm_final.pth")
    print("👉 下一步: python -m streamlit run app.py")
    print("="*50)

if __name__ == "__main__":
    train_pipeline()