import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from data.trend_data_processor import (
    Configs,
    Dataset,
    create_loaders,  # 确保这个函数在 processor 里能返回三个 loader
    evaluate_metrics,
    DEVICE,
    DATA_FOLDER
)
from model.BaseTrend_model import TrendForecaster
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------
# 1. 导出用于 ShockNet 的数据集 (pkl格式)
# ---------------------------------------------------------
def export_dataset_for_shocknet(dataset, output_path="data/cmin_US_price_label_data.pkl"):
    """
    将 Dataset 中的数据导出为 Pickle 文件，供 ShockNet 读取。
    """
    print(f"\n=== 正在导出用于 ShockNet 的价格与标签数据 ===")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_list = []

    try:
        print("正在转换数据格式 (可能需要几秒钟)...")
        # 遍历数据集的内部数据列表
        for item in dataset.data:
            # 期望 item 为 (x, y_act, y_base, y_shock, date, ticker)
            x, y_act, y_base, y_shock, date, ticker = item

            data_list.append({
                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                'ticker': ticker,
                'X_history': x.numpy(),
                'Y_Actual': y_act.item(),
                'Y_Base_Logits': y_base.item(),
                'Y_Shock_Value': y_shock.item()
            })

        df = pd.DataFrame(data_list)
        df.to_pickle(output_path)
        print(f"导出成功! 文件已保存至: {output_path}")
        print(f"总样本数: {len(df)}")

    except Exception as e:
        print(f"导出过程中发生错误: {e}")


# ---------------------------------------------------------
# 2. 训练循环
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, configs, epochs=150, learning_rate=2e-4):
    log_dir = f'runs/TrendNet_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将写入: {log_dir}")

    DEVICE = next(model.parameters()).device
    positive_weight = torch.tensor(0.908, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_mcc = -1.0  # 改为监控 MCC，因为它是我们最关心的指标
    best_threshold_value = 0.5
    BEST_MODEL_PATH = 'best_trendnet.pth'

    print(f"\n=== 开始 TrendNet 训练 (设备: {DEVICE}) ===")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0

        # --- 训练迭代 ---
        for i, (X, y) in enumerate(train_loader):
            X = X.to(DEVICE)
            y = y.to(DEVICE).float()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        # --- 验证评估 ---
        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE).float()
                logits = model(X)
                pred_prob = torch.sigmoid(logits).view(-1)
                all_probs.extend(pred_prob.cpu().numpy())
                all_targets.extend(y.view(-1).cpu().numpy())

        val_metrics = evaluate_metrics(np.array(all_probs), np.array(all_targets))
        current_mcc = val_metrics['mcc']
        current_f1 = val_metrics['f1']

        writer.add_scalar('Metrics/Val_F1', current_f1, epoch)
        writer.add_scalar('Metrics/Val_MCC', current_mcc, epoch)

        log_msg = f"Epoch {epoch + 1}: TrainLoss={avg_train_loss:.4f} | Val MCC={current_mcc:.4f} | F1={current_f1:.4f}"

        # 调度器步进
        scheduler.step(current_mcc)

        # 保存最佳模型
        if current_mcc > best_val_mcc:
            best_val_mcc = current_mcc
            best_threshold_value = val_metrics['threshold']
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            log_msg += " 🏆 [Saved Best]"

        print(log_msg)

    print(f"训练结束。最佳验证集 MCC: {best_val_mcc:.4f}")
    writer.close()
    return model, best_threshold_value


# ---------------------------------------------------------
# 3. 信号生成函数 (生成 CSV)
# ---------------------------------------------------------
def generate_trend_scores(dataset, model_path="best_trendnet.pth", best_threshold=0.50):
    configs = Configs()
    model = TrendForecaster(configs, prediction_horizon=1).to(DEVICE)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"加载权重成功，准备生成全量分数...")
    else:
        print(f"错误：找不到模型文件 {model_path}。")
        return

    model.eval()

    # 提取所有数据进行推理
    X_all = torch.stack([dataset.data[i][0] for i in range(len(dataset.data))])
    all_loader = DataLoader(TensorDataset(X_all), batch_size=256, shuffle=False)
    all_probs = []

    with torch.no_grad():
        for batch in all_loader:
            X = batch[0].to(DEVICE)
            logits = model(X)
            pred_prob = torch.sigmoid(logits).view(-1)
            all_probs.extend(pred_prob.cpu().numpy())

    # 提取元数据
    metadata = dataset.data
    predicted_signal = (np.array(all_probs) > best_threshold).astype(int)

    df_scores = pd.DataFrame({
        'Target_Date': [d[4] for d in metadata],
        'Ticker': [d[5] for d in metadata],
        'P_Trend': all_probs,
        'Signal': predicted_signal
    })

    df_scores.to_csv("trend_base_scores.csv", index=False)
    print(f"趋势信号已保存至 trend_base_scores.csv。")


# ---------------------------------------------------------
# 🔥🔥【新增】4. 测试集评估函数 🔥🔥
# ---------------------------------------------------------
def test_trend_model(dataset, batch_size=64):
    print(f"\n=== 正在评估 TrendNet (Baseline) 在测试集上的表现 ===")

    # 重新获取 test_loader (利用 create_loaders 的分割逻辑)
    _, _, test_loader = create_loaders(dataset, batch_size=batch_size)

    configs = Configs()
    model = TrendForecaster(configs, prediction_horizon=1).to(DEVICE)

    load_path = 'best_trendnet.pth'
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
        print(f"已加载最佳模型: {load_path}")
    else:
        print("❌ 错误：未找到模型文件，无法测试。")
        return

    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE).float()

            logits = model(X)
            pred_prob = torch.sigmoid(logits).view(-1)

            all_probs.extend(pred_prob.cpu().numpy())
            all_targets.extend(y.view(-1).cpu().numpy())

    # 计算最终指标
    metrics = evaluate_metrics(np.array(all_probs), np.array(all_targets))

    print("-" * 40)
    print(f"📊 TrendNet (Baseline) 测试集最终成绩:")
    print(f"   MCC : {metrics['mcc']:.4f}")
    print(f"   F1  : {metrics['f1']:.4f}")
    print(f"   AUC : {metrics['auc']:.4f}")
    print(f"   ACC : {metrics['accuracy']:.4f}")
    print("-" * 40)


# ---------------------------------------------------------
# 5. 主程序入口
# ---------------------------------------------------------
def run_trend_phase():
    configs = Configs()

    # 1. 初始化数据
    dataset = Dataset(DATA_FOLDER, past_window=configs.seq_len)

    # 2. 导出数据给 ShockNet
    export_dataset_for_shocknet(dataset, output_path="data/cmin_US_price_label_data.pkl")

    # 3. 创建 DataLoader
    train_loader, val_loader, _ = create_loaders(dataset, batch_size=64)

    # 4. 初始化模型
    model = TrendForecaster(configs, prediction_horizon=1).to(DEVICE)

    # 5. 训练
    # 训练后会保存 best_trendnet.pth
    trained_model, best_threshold_value = train_model(
        model, train_loader, val_loader, configs, epochs=150, learning_rate=2e-4
    )

    # 6. 生成全量分数 CSV (可选，用于分析)
    generate_trend_scores(dataset, model_path='best_trendnet.pth', best_threshold=best_threshold_value)

    # 🔥🔥🔥 7. 立即在测试集上跑分 🔥🔥🔥
    test_trend_model(dataset)


if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    run_trend_phase()