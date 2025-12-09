import os
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, matthews_corrcoef, log_loss
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

# -------------------------
# 1) 配置数据路径和设备
# -------------------------
# 请根据您的实际路径修改 DATA_FOLDER
DATA_FOLDER = "/share/liuyuqing/causal_net/data/CMIN-Dataset-main/CMIN-US/price/raw"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------------------
# 2) 配置 (Configs)
# -------------------------
class Configs:
    def __init__(self):
        # --- 数据与结构参数 ---
        self.seq_len = 60  # 输入序列长度
        self.pred_len = 1  # 预测长度

        # --- 模型维度 ---
        self.d_model = 15  # 惯性特征维度 (来自 PDM)
        self.d_text = 768  # FinBERT 嵌入维度
        self.d_hidden = 128  # ShockNet 内部隐藏维度/查询维度
        self.memory_slots = 512  # 记忆库大小
        self.dropout = 0.1
        self.d_ff = 64

        # --- PDM 模块参数 (仅作占位符) ---
        self.channel_independence = 0
        self.decomp_method = "dft_decomp"
        self.top_k = 5
        self.moving_avg = 3
        self.down_sampling_layers = 2
        self.down_sampling_window = 2

        # --- 训练超参数 (新增) ---
        self.weight_decay = 1e-5  # L2 正则化，对抗过拟合

        # --- 因果约束损失权重 (改为动态配置)---
        self.lambda_base = 0.5  # 基线损失权重 (保持不变)
        self.margin = 1.0  # Triplet Loss 的 Margin，使约束更严格 (原 0.5)
        self.lambda_ortho_start = 0.1  # 反事实约束损失 L_CF 权重
        self.lambda_ortho_end = 1.0
        self.lambda_triplet_start = 0.1  # 因果三元组损失 L_Triplet 权重
        self.lambda_triplet_end = 1.0



# 3) PDM 分解类 (作为模型架构的组成部分，保留)
class PDM:
    def __init__(self):
        pass

    def decompose(self, x: pd.DataFrame):
        # 实际特征工程已在 Dataset 中完成，这里仅作为占位符
        pass


# 4) 数据集类 (Dataset) - ⚠️ 已修改以支持因果目标计算和全局排序
class Dataset(Dataset):
    def __init__(self, folder, past_window=60):
        self.folder = folder
        # 确保只读取 csv 文件
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
        self.past_window = past_window
        self.data = []
        self.load_all_stocks()

    def load_all_stocks(self):
        print("开始加载数据 (多尺度特征 + 窗口标准化 + 因果目标计算)...")
        for path in self.files:
            try:
                df = pd.read_csv(path)
                df.columns = [c.lower() for c in df.columns]

                required_cols = ["date", "open", "high", "low", "close", "adj close", "volume"]
                if not set(required_cols).issubset(df.columns):
                    continue

                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date")

                # --- 1. 基础标签计算 ---
                adj_close = df["adj close"].ffill().values

                # 计算未来收益率 (Raw Return) -> 用于 L_Triplet 的锚点
                # ret_future[t] 代表 t 时刻对 t+1 的收益率
                ret_future = (adj_close[1:] - adj_close[:-1]) / (adj_close[:-1] + 1e-9)

                # 计算二分类标签 (Y_Actual) -> 用于 L_Pred
                labels = np.where(ret_future > 0, 1, 0)

                # --- 2. 因果目标计算 ---
                # 计算长期 EMA (例如 60 天) 代表纯净惯性 -> 用于 L_Base
                ret_series = pd.Series(ret_future)
                ema_ret = ret_series.ewm(span=60, adjust=False).mean().values

                # 将 EMA 收益率转换为 Logits 形式的近似值
                y_base_logits = np.tanh(ema_ret * 100) * 5.0

                # --- 3. 特征工程 ---
                base_features = df[["open", "high", "low", "close", "volume"]].ffill()
                s1 = base_features.rolling(window=3, min_periods=1).mean()
                s2 = base_features.rolling(window=7, min_periods=1).mean()
                s3 = base_features.rolling(window=30, min_periods=1).mean()
                combined_features = np.concatenate([s1.values, s2.values, s3.values], axis=1)

                combined_features = combined_features[:-1]  # 截断最后一行以匹配 ret_future 长度

                # --- 4. 生成样本 ---
                total_len = len(combined_features)
                if total_len <= self.past_window:
                    continue

                for t in range(self.past_window, total_len):
                    window_raw = combined_features[t - self.past_window: t]

                    # 窗口内 Z-Score 标准化
                    mean = np.mean(window_raw, axis=0, keepdims=True)
                    std = np.std(window_raw, axis=0, keepdims=True) + 1e-5
                    window_norm = (window_raw - mean) / std

                    # 构建 Tensor
                    x_tensor = torch.tensor(window_norm, dtype=torch.float32)
                    y_actual = torch.tensor(labels[t - 1], dtype=torch.float32)  # 真实 0/1
                    y_base = torch.tensor(y_base_logits[t - 1], dtype=torch.float32)  # 惯性基线 Logits
                    y_shock = torch.tensor(ret_future[t - 1], dtype=torch.float32)  # 真实收益数值

                    if 'date' in df.columns:
                        # 这里的 t 对应 label 的时间
                        target_date = df.iloc[t]['date']
                        ticker = os.path.basename(path).replace('.csv', '')

                        # 存储扩充后的数据元组
                        self.data.append((x_tensor, y_actual, y_base, y_shock, target_date, ticker))

            except Exception as e:
                # print(f"Error processing {path}: {e}")
                continue
        print(f"加载完成，样本总数: {len(self.data)}")

        # 关键修正 A：按时间戳对所有股票数据进行全局排序
        # x[4] 是 target_date (pd.Timestamp)
        self.data.sort(key=lambda x: x[4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回所有数据，供 DataLoader 使用
        return self.data[idx]


# 5) Chronological split & DataLoader - 修正后的版本
def create_loaders(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=64):
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    idxs = np.arange(total)

    # 关键修正 B：数据已在 Dataset 中全局按日期排序，此处按顺序切分索引
    train_idx = idxs[:train_end]
    val_idx = idxs[train_end:val_end]
    test_idx = idxs[val_end:]

    # 提取所有样本的 X 和 Y_actual
    X_all = torch.stack([dataset.data[i][0] for i in idxs])
    y_all = torch.tensor([dataset.data[i][1] for i in idxs], dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_all[train_idx], y_all[train_idx]),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_all[val_idx], y_all[val_idx]),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_all[test_idx], y_all[test_idx]),
                             batch_size=batch_size, shuffle=False)

    print(f"数据集加载完成：总样本数={total}, 训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={len(test_idx)}")

    #关键修正 C：打印划分的日期范围以进行最终验证
    train_end_date = dataset.data[train_end - 1][4].strftime('%Y-%m-%d')
    val_end_date = dataset.data[val_end - 1][4].strftime('%Y-%m-%d')
    print(f"严格时间划分：训练集结束日期={train_end_date}, 验证集结束日期={val_end_date}")

    return train_loader, val_loader, test_loader


# 6) Evaluation function (保持不变)
def evaluate_metrics(all_probs, all_targets):
    """计算多维评估指标"""
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # 1. 计算动态阈值
    dynamic_threshold = np.median(all_probs)
    preds = (all_probs > dynamic_threshold).astype(int)

    # 2. 基础分类指标
    tp = ((preds == 1) & (all_targets == 1)).sum()
    tn = ((preds == 0) & (all_targets == 0)).sum()
    fp = ((preds == 1) & (all_targets == 0)).sum()
    fn = ((preds == 0) & (all_targets == 1)).sum()

    total = len(all_targets)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # 3. 高级指标
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5  # 如果只有一个类别，AUC无法计算

    try:
        mcc = matthews_corrcoef(all_targets, preds)
    except ValueError:
        mcc = 0

    # 4. Log Loss (衡量概率校准度)
    try:
        loss_val = log_loss(all_targets, all_probs)
    except:
        loss_val = -1

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
        "log_loss": loss_val,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": dynamic_threshold
    }


# 7) 辅助函数 (保持不变)
def save_model(model, path):
    """保存模型权重"""
    torch.save(model.state_dict(), path)