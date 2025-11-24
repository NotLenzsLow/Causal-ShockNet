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
DATA_FOLDER = "D:\\pythonbank\\datatest"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------------------
# 2) 配置 (Configs)
# -------------------------
class Configs:
    def __init__(self):
        self.seq_len = 60
        self.pred_len = 1
        self.d_model = 15  # 5原始特征 * 3个移动平均尺度 = 15
        self.d_ff = 64
        self.dropout = 0.1
        self.channel_independence = 0
        self.decomp_method = "dft_decomp"
        self.top_k = 5
        self.moving_avg = 3
        self.down_sampling_layers = 2
        self.down_sampling_window = 2


# 3) PDM 分解类 (作为模型架构的组成部分，保留)
class PDM:
    def __init__(self):
        pass

    def decompose(self, x: pd.DataFrame):
        # 实际特征工程已在 Dataset 中完成，这里仅作为占位符
        pass


# 4) 数据集类 (Dataset)
class Dataset(Dataset):
    def __init__(self, folder, past_window=60):
        self.folder = folder
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
        self.past_window = past_window
        self.data = []
        self.load_all_stocks()

    def load_all_stocks(self):
        print("开始加载数据 (多尺度特征 + 窗口标准化)...")
        for path in self.files:
            try:
                df = pd.read_csv(path)
                df.columns = [c.lower() for c in df.columns]

                required_cols = ["date", "open", "high", "low", "close", "adj close", "volume"]
                if not set(required_cols).issubset(df.columns):
                    continue

                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date")

                # 1. 计算标签
                adj_close = df["adj close"].ffill().values
                ret_future = (adj_close[1:] - adj_close[:-1]) / (adj_close[:-1] + 1e-9)
                labels = np.where(ret_future > 0, 1, 0)

                # 2. 特征工程：多尺度移动平均 (15维特征)
                base_features = df[["open", "high", "low", "close", "volume"]].ffill()
                s1 = base_features.rolling(window=3, min_periods=1).mean()
                s2 = base_features.rolling(window=7, min_periods=1).mean()
                s3 = base_features.rolling(window=30, min_periods=1).mean()
                combined_features = np.concatenate([s1.values, s2.values, s3.values], axis=1)

                combined_features = combined_features[:-1]  # 截断最后一行

                # 3. 生成样本 (含 窗口内标准化)
                total_len = len(combined_features)
                if total_len <= self.past_window:
                    continue

                for t in range(self.past_window, total_len):
                    window_raw = combined_features[t - self.past_window: t]

                    # 窗口内 Z-Score 标准化
                    mean = np.mean(window_raw, axis=0, keepdims=True)
                    std = np.std(window_raw, axis=0, keepdims=True) + 1e-5
                    window_norm = (window_raw - mean) / std

                    x_tensor = torch.tensor(window_norm, dtype=torch.float32)
                    y_val = torch.tensor(labels[t - 1], dtype=torch.float32)

                    # 存储 [特征, 标签, 预测日期, 股票代码]
                    # 我们需要将 date/ticker 存储起来，用于后续信号导出

                    # 检查 date 是否存在且有效
                    if 'date' in df.columns and t < len(df):
                        target_date = df.iloc[t]['date']
                        ticker = os.path.basename(path).replace('.csv', '')

                        self.data.append((x_tensor, y_val, target_date, ticker))
                    else:
                        # 如果没有日期信息，只存储特征和标签
                        self.data.append((x_tensor, y_val, None, None))

            except Exception as e:
                # print(f"Error processing {path}: {e}")
                continue
        print(f"加载完成，样本总数: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回 (特征, 标签)
        return self.data[idx][0], self.data[idx][1]


# 5) Chronological split & DataLoader
def create_loaders(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=64):
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    idxs = np.arange(total)

    # 按照索引顺序切分
    train_idx = idxs[:train_end]
    val_idx = idxs[train_end:val_end]
    test_idx = idxs[val_end:]

    # 将数据打包成 TensorDataset
    # 注意：我们这里只提取特征和标签，日期信息在数据集对象中
    X_all = torch.stack([dataset.data[i][0] for i in idxs])
    y_all = torch.tensor([dataset.data[i][1] for i in idxs], dtype=torch.float32).unsqueeze(-1)

    train_ds = TensorDataset(X_all[train_idx], y_all[train_idx])
    val_ds = TensorDataset(X_all[val_idx], y_all[val_idx])
    test_ds = TensorDataset(X_all[test_idx], y_all[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"数据集加载完成：总样本数={total}, 训练集={len(train_ds)}, 验证集={len(val_ds)}, 测试集={len(test_ds)}")
    return train_loader, val_loader, test_loader


# -------------------------
# 6) Evaluation function
# -------------------------
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

    # 3. 高级指标 (需要 sklearn)
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


# 7) 辅助函数
def save_model(model, path):
    """保存模型权重"""
    torch.save(model.state_dict(), path)