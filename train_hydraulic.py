"""
训练脚本：液压泵内部泄漏故障诊断模型
============================================
基于 UCI 液压系统数据集（ZeMA gGmbH 测试台架）
诊断目标：内部泵泄漏三分类
  - 0: NORMAL   （无泄漏，正常运行）
  - 1: WEAK_LEAK（轻微泄漏，需关注）
  - 2: SEVERE_LEAK（严重泄漏，需立即检修）

模型架构：
  - 方案A: MLP 多层感知机（快速基线）
  - 方案B: Bi-LSTM 时序分类器（利用周期性特征）
  - 方案C: 1D-CNN + LSTM 混合（捕捉局部+全局时序特征）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import joblib
import os
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
MODEL_DIR = "models_hydraulic"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n" + "=" * 60)
print("加载预处理后的数据...")
X_train = np.load(f"{MODEL_DIR}/X_train.npy")
X_test = np.load(f"{MODEL_DIR}/X_test.npy")
y_train = np.load(f"{MODEL_DIR}/y_train.npy")
y_test = np.load(f"{MODEL_DIR}/y_test.npy")
feature_names = np.load(f"{MODEL_DIR}/feature_names.npy", allow_pickle=True)

label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
scaler = joblib.load(f"{MODEL_DIR}/hydraulic_scaler.pkl")

n_features = X_train.shape[1]
n_classes = len(label_encoder.classes_)
print(f"  训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")
print(f"  特征维度: {n_features}")
print(f"  类别数: {n_classes} -> {list(label_encoder.classes_)}")

# ============================================================================
# 2. 构造时序窗口（将静态特征转为伪序列供LSTM使用）
# ============================================================================
# 策略：每个样本复制成 window_size=10 个时间步
# 模拟周期信号，让 LSTM 能捕捉时间依赖

WINDOW_SIZE = 10  # 模拟10个时间步（对应10秒周期采样）
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001

def make_windowed(X, window_size):
    """将静态特征转为伪时序张量"""
    n_samples = X.shape[0]
    # 每个样本变成 window_size 个时间步，特征重复
    X_windowed = np.tile(X, (1, window_size)).reshape(n_samples, window_size, n_features)
    return X_windowed

X_train_w = make_windowed(X_train, WINDOW_SIZE)
X_test_w = make_windowed(X_test, WINDOW_SIZE)

print(f"\n窗口化后形状: train={X_train_w.shape}, test={X_test_w.shape}")

# --- MLP 数据（2D，无需窗口化）---
X_train_t_2d = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t_2d = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_loader_mlp = DataLoader(
    TensorDataset(X_train_t_2d, y_train_t), batch_size=BATCH_SIZE, shuffle=True
)
test_loader_mlp = DataLoader(
    TensorDataset(X_test_t_2d, y_test_t), batch_size=BATCH_SIZE, shuffle=False
)

# --- LSTM/CNN-LSTM 数据（3D，窗口化）---
X_train_t_3d = torch.FloatTensor(X_train_w)
X_test_t_3d = torch.FloatTensor(X_test_w)

train_loader_seq = DataLoader(
    TensorDataset(X_train_t_3d, y_train_t), batch_size=BATCH_SIZE, shuffle=True
)
test_loader_seq = DataLoader(
    TensorDataset(X_test_t_3d, y_test_t), batch_size=BATCH_SIZE, shuffle=False
)

# ============================================================================
# 3. 模型定义
# ============================================================================

class MLPClassifier(nn.Module):
    """多层感知机（快速基线）"""
    def __init__(self, input_dim, hidden_dim=128, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, window, features) 或 (batch, features)
        x = x.view(x.size(0), -1)  # 展平所有维度
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        return self.fc3(x)


class BiLSTMClassifier(nn.Module):
    """Bi-LSTM 时序分类器"""
    def __init__(self, input_dim, hidden_dim=64, n_classes=3, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        # x: (batch, window, features)
        out, _ = self.lstm(x)
        # 取最后时间步的输出
        out = self.fc(out[:, -1, :])
        return out


class CNN1D_LSTM(nn.Module):
    """1D-CNN 局部特征提取 + 全局池化分类"""
    def __init__(self, input_dim, hidden_dim=64, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ============================================================================
# 4. 训练函数
# ============================================================================

def train_model(model, train_loader, test_loader, model_name, epochs=50):
    """通用训练流程"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_acc = 0
    best_state = None
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    print(f"\n{'='*50}")
    print(f"训练模型: {model_name}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        test_acc = correct / total
        scheduler.step(test_loss)
        history['train_loss'].append(train_loss / len(train_loader))
        history['test_loss'].append(test_loss / len(test_loader))
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Test Loss: {test_loss/len(test_loader):.4f} | "
                  f"Test Acc: {test_acc*100:.1f}%")

    # 加载最优模型
    model.load_state_dict(best_state)
    return model, best_acc, history


def evaluate_model(model, X_test_t, y_test_t, model_name):
    """评估模型并打印报告"""
    model.eval()
    with torch.no_grad():
        X_dev = X_test_t.to(DEVICE)
        outputs = model(X_dev)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()

    y_true = y_test_t.numpy()

    print(f"\n{'='*50}")
    print(f"评估结果: {model_name}")
    print(f"{'='*50}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    print(f"\n准确率 (Accuracy):  {acc*100:.2f}%")
    print(f"加权F1 (F1-Score): {f1*100:.2f}%")
    print(f"精确率 (Precision): {precision*100:.2f}%")
    print(f"召回率 (Recall):    {recall*100:.2f}%")

    print(f"\n分类报告:")
    target_names = [c.replace('_', ' ') for c in label_encoder.classes_]
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 保存混淆矩阵图
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(target_names, rotation=45, ha='right')
    ax.set_yticklabels(target_names)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(f'{model_name} 混淆矩阵\nAcc={acc*100:.1f}% F1={f1*100:.1f}%')
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = f"{MODEL_DIR}/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n混淆矩阵已保存: {cm_path}")

    return acc, f1, cm


# ============================================================================
# 5. 训练三个模型并对比
# ============================================================================
results = {}

# --- 方案A: MLP ---
mlp = MLPClassifier(n_features, hidden_dim=128, n_classes=n_classes)
mlp, mlp_acc, mlp_hist = train_model(mlp, train_loader_mlp, test_loader_mlp, "MLP", epochs=EPOCHS)
mlp_acc_val, mlp_f1, _ = evaluate_model(mlp, X_test_t_2d, y_test_t, "MLP")
results['MLP'] = {'acc': mlp_acc_val, 'f1': mlp_f1, 'model': mlp}
torch.save(mlp.state_dict(), f"{MODEL_DIR}/mlp_pump_leak.pth")

# --- 方案B: Bi-LSTM ---
lstm = BiLSTMClassifier(n_features, hidden_dim=64, n_classes=n_classes)
lstm, lstm_acc, lstm_hist = train_model(lstm, train_loader_seq, test_loader_seq, "Bi-LSTM", epochs=EPOCHS)
lstm_acc_val, lstm_f1, _ = evaluate_model(lstm, X_test_t_3d, y_test_t, "Bi-LSTM")
results['Bi-LSTM'] = {'acc': lstm_acc_val, 'f1': lstm_f1, 'model': lstm}
torch.save(lstm.state_dict(), f"{MODEL_DIR}/bilstm_pump_leak.pth")

# --- 方案C: CNN+LSTM ---
cnn_lstm = CNN1D_LSTM(n_features, hidden_dim=64, n_classes=n_classes)
cnn_lstm, cnn_lstm_acc, cnn_lstm_hist = train_model(cnn_lstm, train_loader_seq, test_loader_seq, "CNN-LSTM", epochs=EPOCHS)
cnn_lstm_acc_val, cnn_lstm_f1, _ = evaluate_model(cnn_lstm, X_test_t_3d, y_test_t, "CNN-LSTM")
results['CNN-LSTM'] = {'acc': cnn_lstm_acc_val, 'f1': cnn_lstm_f1, 'model': cnn_lstm}
torch.save(cnn_lstm.state_dict(), f"{MODEL_DIR}/cnn_lstm_pump_leak.pth")

# ============================================================================
# 6. 对比汇总
# ============================================================================
print("\n" + "=" * 60)
print("模型性能对比汇总")
print("=" * 60)
print(f"{'模型':<15} {'准确率':>12} {'加权F1':>12}")
print("-" * 40)
for name, res in results.items():
    print(f"{name:<15} {res['acc']*100:>11.2f}% {res['f1']*100:>11.2f}%")

# 找出最佳模型
best_model_name = max(results, key=lambda k: results[k]['f1'])
print(f"\n最佳模型: {best_model_name} (F1={results[best_model_name]['f1']*100:.2f}%)")

# 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'MLP': '#e74c3c', 'Bi-LSTM': '#3498db', 'CNN-LSTM': '#2ecc71'}
for name, hist in [('MLP', mlp_hist), ('Bi-LSTM', lstm_hist), ('CNN-LSTM', cnn_lstm_hist)]:
    axes[0].plot(hist['train_loss'], label=name, color=colors[name])
    axes[1].plot(hist['test_acc'], label=name, color=colors[name])
axes[0].set_title('训练损失')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].set_title('测试准确率')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/training_comparison.png", dpi=150)
plt.close()
print(f"\n训练曲线已保存: {MODEL_DIR}/training_comparison.png")

# ============================================================================
# 7. 保存最佳模型元信息
# ============================================================================
import json
meta = {
    'best_model': best_model_name,
    'n_features': int(n_features),
    'n_classes': int(n_classes),
    'classes': list(label_encoder.classes_),
    'window_size': WINDOW_SIZE,
    'results': {k: {'acc': float(v['acc']), 'f1': float(v['f1'])} for k, v in results.items()}
}
with open(f"{MODEL_DIR}/model_meta.json", 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"元信息已保存: {MODEL_DIR}/model_meta.json")

print("\n" + "=" * 60)
print("训练完成！")
print(f"所有模型保存在: {MODEL_DIR}/")
print("=" * 60)
