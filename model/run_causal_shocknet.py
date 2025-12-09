import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from model.ShockNet import CausalShockNet
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score


# --- 0. Configs 类 (最终优化版参数) ---
class Configs:
    def __init__(self):
        # 模型结构参数
        self.seq_len = 60
        self.pred_len = 1
        self.d_model = 15
        self.d_text = 768
        self.d_hidden = 128
        self.memory_slots = 512
        self.dropout = 0.1
        self.d_ff = 64

        # PDM 参数
        self.channel_independence = 0
        self.decomp_method = "dft_decomp"
        self.top_k = 5
        self.moving_avg = 3
        self.down_sampling_layers = 2
        self.down_sampling_window = 2

        # 训练参数
        self.weight_decay = 1e-4
        self.lambda_base = 0.5

        # 🔥 核心参数 (配合 LayerNorm 使用)
        self.lambda_ortho_start = 0.1
        self.lambda_ortho_end = 1.0

        # Margin 设为 0.5 (因为 LayerNorm 限制了模长，距离变小了)
        self.margin = 0.5

        self.lambda_triplet_start = 0.1
        self.lambda_triplet_end = 1.0


# --- 辅助函数：Batch Hard Triplet Loss ---
def batch_hard_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                            margin: float = 1.0) -> torch.Tensor:
    dist_ap = torch.sum((anchor - positive) ** 2, dim=1)
    dist_an = torch.sum((anchor - negative) ** 2, dim=1)
    loss = dist_ap - dist_an + margin
    loss = torch.max(loss, torch.zeros_like(loss))
    if loss.sum() == 0:
        return torch.zeros(1, device=anchor.device)
    return loss.mean()


# --- 1. 数据集类 (含 Test 模式) ---
class FinancialShockDataset(Dataset):
    def __init__(self, configs, price_data_path: str, event_emb_path: str, mode: str = 'train'):
        self.configs = configs

        # 仅在第一次初始化时打印路径，避免刷屏
        if mode == 'train':
            print(f"Loading Price Data from: {price_data_path}")
            print(f"Loading Event Embeddings from: {event_emb_path}")

        try:
            self.price_df = pd.read_pickle(price_data_path)
        except FileNotFoundError:
            self.price_df = pd.DataFrame()

        try:
            event_df = pd.read_pickle(event_emb_path)
            if 'date' in event_df.columns:
                event_df = event_df.set_index(['date', 'ticker'])
            self.event_map = event_df['event_embedding'].to_dict()
        except FileNotFoundError:
            self.event_map = {}

        if not self.price_df.empty:
            self.price_df = self.price_df.set_index(['date', 'ticker'])
            self.data_indices = [
                (date, ticker) for date, ticker in self.price_df.index
                if (date, ticker) in self.event_map
            ]
        else:
            self.data_indices = []

        # 🔥 严格的时间划分 (70% / 15% / 15%)
        # 必须与 TrendNet 保持一致
        total_len = len(self.data_indices)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)

        if mode == 'train':
            self.data_indices = self.data_indices[:train_end]
        elif mode == 'val':
            self.data_indices = self.data_indices[train_end:val_end]
        elif mode == 'test':  # 🔥 新增测试集分支
            self.data_indices = self.data_indices[val_end:]

        print(f"数据集初始化完成 ({mode})。样本数: {len(self.data_indices)}")

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        date, ticker = self.data_indices[idx]
        X_history = self.price_df.loc[(date, ticker), 'X_history']
        Y_Actual = self.price_df.loc[(date, ticker), 'Y_Actual']
        Y_Base_Logits = self.price_df.loc[(date, ticker), 'Y_Base_Logits']
        Y_Shock_Value = self.price_df.loc[(date, ticker), 'Y_Shock_Value']
        event_emb = self.event_map[(date, ticker)]

        Y_Shock_Value_tensor = torch.tensor([Y_Shock_Value], dtype=torch.float32)

        return torch.tensor(X_history, dtype=torch.float32), \
            torch.tensor(event_emb, dtype=torch.float32), \
            torch.tensor(Y_Actual, dtype=torch.float32), \
            torch.tensor(Y_Base_Logits, dtype=torch.float32), \
            Y_Shock_Value_tensor


# --- 2. 数据加载函数 (返回 Train/Val/Test) ---
def get_financial_shock_loaders(configs, batch_size: int = 64,
                                price_data_path: str = 'price_label_data.pkl',
                                event_emb_path: str = 'cmin_US_event_embeddings_processed.pkl'):
    train_dataset = FinancialShockDataset(configs, price_data_path, event_emb_path, mode='train')
    val_dataset = FinancialShockDataset(configs, price_data_path, event_emb_path, mode='val')
    # 🔥 新增测试集 Dataset
    test_dataset = FinancialShockDataset(configs, price_data_path, event_emb_path, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# --- 3. 通用验证/测试函数 ---
def validate_shocknet(model, dataloader, configs, device, writer=None, epoch=None, mode='Val'):
    model.eval()
    loss_sum, metrics = 0, {'pred': 0, 'base': 0, 'ortho': 0, 'triplet': 0}
    all_y_true, all_y_pred = [], []
    all_y_probs = []  # 用于计算 AUC

    criterion_pred = nn.BCEWithLogitsLoss()
    criterion_base = nn.MSELoss()

    # 如果是测试模式，使用最终权重；如果是验证模式，也可固定
    lambda_ortho = configs.lambda_ortho_end
    lambda_triplet = configs.lambda_triplet_end

    with torch.no_grad():
        for X, event_emb, Y_Actual, Y_Base_Logits, Y_Shock_Value in dataloader:
            X, event_emb = X.to(device), event_emb.to(device)
            Y_Actual, Y_Base_Logits, Y_Shock_Value = Y_Actual.to(device).float(), Y_Base_Logits.to(
                device).float(), Y_Shock_Value.to(device).float()
            if Y_Shock_Value.dim() == 1: Y_Shock_Value = Y_Shock_Value.unsqueeze(1)

            # 前向传播
            (final_pred_prob, final_pred_logits, trend_pred, shock_f, shock_cf,
             feat_f, feat_cf, query_f, query_cf) = model(X, event_emb)

            # 收集预测结果
            y_pred_binary = (final_pred_prob.view(-1) > 0.5).long()
            all_y_true.extend(Y_Actual.view(-1).cpu().numpy())
            all_y_pred.extend(y_pred_binary.cpu().numpy())
            all_y_probs.extend(final_pred_prob.view(-1).cpu().numpy())

            # 计算 Loss (用于早停监控)
            L_pred = criterion_pred(final_pred_logits.view(-1), Y_Actual.view(-1))
            L_base = criterion_base(trend_pred.view(-1), Y_Base_Logits.view(-1))

            # L_ortho (feat 已经过 LayerNorm)
            L_ortho = torch.mean(torch.abs(torch.sum(feat_f * feat_cf, dim=1)))

            # L_triplet (修复路径引用 + 归一化)
            anchor_emb = model.target_embedding_net(Y_Shock_Value)

            anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
            feat_f_norm = F.normalize(feat_f, p=2, dim=1)
            feat_cf_norm = F.normalize(feat_cf, p=2, dim=1)

            L_triplet = batch_hard_triplet_loss(anchor_norm, feat_f_norm, feat_cf_norm, margin=configs.margin)

            L_Total = L_pred + configs.lambda_base * L_base + lambda_ortho * L_ortho + lambda_triplet * L_triplet

            loss_sum += L_Total.item()
            metrics['pred'] += L_pred.item()
            metrics['base'] += L_base.item()
            metrics['ortho'] += L_ortho.item()
            metrics['triplet'] += L_triplet.item()

    num_batches = len(dataloader)
    all_y_true_np = np.array(all_y_true)
    all_y_pred_np = np.array(all_y_pred)
    all_y_probs_np = np.array(all_y_probs)

    mcc, f1, auc, acc = 0.0, 0.0, 0.5, 0.0
    if len(all_y_true_np) > 0 and len(np.unique(all_y_true_np)) > 1:
        mcc = matthews_corrcoef(all_y_true_np, all_y_pred_np)
        f1 = f1_score(all_y_true_np, all_y_pred_np, zero_division=0)
        acc = accuracy_score(all_y_true_np, all_y_pred_np)
        try:
            auc = roc_auc_score(all_y_true_np, all_y_probs_np)
        except:
            auc = 0.5

    avg_loss = loss_sum / num_batches

    if writer and epoch is not None:
        writer.add_scalar(f'Epoch_Loss/{mode}_Total', avg_loss, epoch)
        writer.add_scalar(f'Metrics/{mode}_MCC', mcc, epoch)
        writer.add_scalar(f'Metrics/{mode}_F1', f1, epoch)

    return avg_loss, mcc, f1, auc, acc


# --- 4. 核心训练函数 (含早停与保存) ---
def train_causal_shocknet(model, train_loader, val_loader, configs, epochs=50, learning_rate=1e-3,
                          pos_weight_value=0.908):
    log_dir = f'runs/ShockNet_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将写入: {log_dir}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)

    positive_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(DEVICE)
    criterion_pred = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    criterion_base = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=configs.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    # 🔥 最佳模型跟踪
    best_val_mcc = -1.0
    best_model_path = 'best_causal_shocknet.pth'

    print(f"\n=== 开始 Causal-ShockNet 联合训练 (设备: {DEVICE}) ===")

    for epoch in range(epochs):
        t = epoch / epochs
        lambda_ortho = configs.lambda_ortho_start + t * (configs.lambda_ortho_end - configs.lambda_ortho_start)
        lambda_triplet = configs.lambda_triplet_start + t * (configs.lambda_triplet_end - configs.lambda_triplet_start)

        model.train()
        loss_tracker = {'total': 0, 'pred': 0, 'base': 0, 'ortho': 0, 'triplet': 0}
        L_pred_val, L_ortho_val, L_triplet_val = 0, 0, 0  # 初始化显示变量

        for i, (X, event_emb, Y_Actual, Y_Base_Logits, Y_Shock_Value) in enumerate(train_loader):
            X, event_emb = X.to(DEVICE), event_emb.to(DEVICE)
            Y_Actual, Y_Base_Logits, Y_Shock_Value = Y_Actual.to(DEVICE).float(), Y_Base_Logits.to(
                DEVICE).float(), Y_Shock_Value.to(DEVICE).float()
            if Y_Shock_Value.dim() == 1: Y_Shock_Value = Y_Shock_Value.unsqueeze(1)

            optimizer.zero_grad()
            (final_pred_prob, final_pred_logits, trend_pred, shock_f, shock_cf,
             feat_f, feat_cf, query_f, query_cf) = model(X, event_emb)

            # Debug: 监控特征模长 (验证 LayerNorm 是否生效)
            if i == 0 and epoch % 1 == 0:
                norm_f = torch.norm(feat_f, p=2, dim=1).mean().item()
                norm_cf = torch.norm(feat_cf, p=2, dim=1).mean().item()
                print(
                    f"\n[DEBUG Epoch {epoch}] Feat_F Norm: {norm_f:.4f} | Feat_CF Norm: {norm_cf:.4f} (Should be ~3.8)")

            L_pred_i = criterion_pred(final_pred_logits.view(-1), Y_Actual.view(-1))
            L_base_i = criterion_base(trend_pred.view(-1), Y_Base_Logits.view(-1))

            # 正交损失
            L_ortho_i = torch.mean(torch.abs(torch.sum(feat_f * feat_cf, dim=1)))

            # Triplet Loss
            anchor_emb = model.target_embedding_net(Y_Shock_Value)
            anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
            feat_f_norm = F.normalize(feat_f, p=2, dim=1)
            feat_cf_norm = F.normalize(feat_cf, p=2, dim=1)

            L_triplet_i = batch_hard_triplet_loss(anchor_norm, feat_f_norm, feat_cf_norm, margin=configs.margin)

            L_Total = L_pred_i + configs.lambda_base * L_base_i + lambda_ortho * L_ortho_i + lambda_triplet * L_triplet_i

            L_Total.backward()
            optimizer.step()

            loss_tracker['total'] += L_Total.item()
            loss_tracker['pred'] += L_pred_i.item()
            loss_tracker['ortho'] += L_ortho_i.item()
            loss_tracker['triplet'] += L_triplet_i.item()

            # 用于日志显示
            L_pred_val = L_pred_i.item()
            L_ortho_val = L_ortho_i.item()
            L_triplet_val = L_triplet_i.item()

        avg_train_loss = loss_tracker['total'] / len(train_loader)
        writer.add_scalar('Epoch_Loss/Train_Total', avg_train_loss, epoch)

        # --- 验证阶段 ---
        if val_loader:
            avg_val_loss, val_mcc, val_f1, _, _ = validate_shocknet(model, val_loader, configs, DEVICE, writer, epoch,
                                                                    mode='Val')

            log_msg = f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | MCC={val_mcc:.4f} | F1={val_f1:.4f}"

            # 🔥 保存最佳模型逻辑
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                torch.save(model.state_dict(), best_model_path)
                log_msg += " 🏆 [New Best]"

            scheduler.step(val_mcc)
        else:
            log_msg = f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}"

        print(log_msg)
        print(
            f"    Details: L_Pred={L_pred_val:.4f} | L_Ortho={L_ortho_val:.4f} (w={lambda_ortho:.1f}) | L_Triplet={L_triplet_val:.4f} (w={lambda_triplet:.1f})")

    print(f"\n训练流程完成。最佳验证集 MCC: {best_val_mcc:.4f}")
    writer.close()
    return best_model_path


# --- 5. 🔥🔥🔥 新增：测试集评估函数 🔥🔥🔥 ---
def test_best_model(configs, best_model_path, test_loader):
    print(f"\n=== 正在评估 Causal-ShockNet 在测试集上的表现 ===")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = CausalShockNet(configs).to(DEVICE)

    # 加载权重
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
        print(f"✅ 已加载最佳模型权重: {best_model_path}")
    else:
        print("❌ 错误：找不到模型文件，无法测试。")
        return

    # 在测试集上运行
    avg_test_loss, test_mcc, test_f1, test_auc, test_acc = validate_shocknet(
        model, test_loader, configs, DEVICE, writer=None, mode='Test'
    )

    print("-" * 40)
    print(f"📊 Causal-ShockNet 最终测试集成绩:")
    print(f"   MCC : {test_mcc:.4f}")
    print(f"   F1  : {test_f1:.4f}")
    print(f"   AUC : {test_auc:.4f}")
    print(f"   ACC : {test_acc:.4f}")
    print("-" * 40)


# --- 6. 主程序入口 ---
if __name__ == "__main__":
    configs = Configs()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("正在加载和对齐训练数据 (Train/Val/Test)...")
    PRICE_DATA_PATH = '/share/liuyuqing/causal_net/data/cmin_US_price_label_data.pkl'
    EVENT_EMB_PATH = '/share/liuyuqing/causal_net/cmin_US_event_embeddings_processed.pkl'

    # 获取 Train, Val, Test 三个 loader
    train_loader, val_loader, test_loader = get_financial_shock_loaders(
        configs,
        batch_size=64,
        price_data_path=PRICE_DATA_PATH,
        event_emb_path=EVENT_EMB_PATH
    )

    print("正在初始化 CausalShockNet 模型...")
    model = CausalShockNet(configs).to(DEVICE)

    # 1. 训练并自动保存最佳模型
    best_model_path = train_causal_shocknet(
        model,
        train_loader,
        val_loader,
        configs,
        epochs=50,  # 建议设为 50，通常在 15-20 epoch 达到最佳
        learning_rate=1e-4
    )

    # 2. 立即在测试集上评估最佳模型
    test_best_model(configs, best_model_path, test_loader)

    print("\n✅ 所有任务结束。")