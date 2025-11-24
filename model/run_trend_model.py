import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset  # 导入 TensorDataset
from data.trend_data_processor import (
    Configs,
    Dataset,
    create_loaders,
    evaluate_metrics,
    save_model,
    DEVICE,
    DATA_FOLDER
)

from model.BaseTrend_model import TrendForecaster
# 1) Training loop (保持不变)
def train_model(model, train_loader, val_loader, configs, epochs=50, learning_rate=1e-3):
    DEVICE = next(model.parameters()).device
    positive_weight = torch.tensor(0.908, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = -1

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0

        # --- 训练循环 ---
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

        # --- 评估循环：收集所有结果 ---
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

        # !!! 关键修改: 将评估指标的计算移到循环外部 !!!

        # 计算评估指标 (现在使用整个验证集的数据)
        val_metrics = evaluate_metrics(np.array(all_probs), np.array(all_targets))

        # 打印日志
        print(f"应用动态阈值: {val_metrics['threshold']:.4f}")
        print(f"Epoch {epoch + 1}: TrainLoss={avg_train_loss:.6f} | "
              f"ValAcc={val_metrics['accuracy']:.4f} | "
              f"ValPrec={val_metrics['precision']:.4f} | "
              f"ValRecall={val_metrics['recall']:.4f} | "
              f"ValF1={val_metrics['f1']:.4f} | "
              f"AUC={val_metrics['auc']:.4f} | "
              f"MCC={val_metrics['mcc']:.4f}")


        # 保存最优模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_threshold_value = val_metrics['threshold']  #记录当前最佳阈值
            save_model(model, "best_direction_model.pth")
            print(f" 保存最优模型（F1={best_f1:.4f}）")

    return model, best_threshold_value #返回模型和最佳阈值


# 2) 信号生成函数 (优化：接收已加载的数据集)
def generate_trend_scores(full_dataset, model_path="best_direction_model.pth", best_threshold=0.50):
    configs = Configs()
    # 1. 初始化模型并加载权重
    model = TrendForecaster(configs, prediction_horizon=1).to(DEVICE)
    if os.path.exists(model_path):
        # 增加 weights_only=True，消除警告并提升安全性
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"\n=== 模型权重 {model_path} 加载成功，开始生成趋势分数 ===")
    else:
        print(f"\n错误：找不到模型文件 {model_path}。请先运行训练。")
        return

    model.eval()

    # 准备 DataLoader，不需要 shuffle，用于全量推断
    # 从传入的 full_dataset 中提取 Tensor
    X_all = torch.stack([full_dataset.data[i][0] for i in range(len(full_dataset.data))])
    y_all = torch.tensor([full_dataset.data[i][1] for i in range(len(full_dataset.data))],
                         dtype=torch.float32).unsqueeze(-1)

    all_loader = DataLoader(TensorDataset(X_all, y_all), batch_size=256, shuffle=False)

    all_probs = []

    # 2. 批量推断
    with torch.no_grad():
        for X, _ in all_loader:
            X = X.to(DEVICE)
            # model(X) 现在返回的是 Logits
            logits = model(X)
            # 显式计算概率
            pred_prob = torch.sigmoid(logits).view(-1)
            all_probs.extend(pred_prob.cpu().numpy())

    # 3. 提取元数据 (日期和股票代码)
    metadata = full_dataset.data

    #使用传入的最佳阈值生成二元信号
    all_probs_np = np.array(all_probs)
    predicted_signal = (all_probs_np > best_threshold).astype(int)

    df_scores = pd.DataFrame({
        'Target_Date': [d[2] for d in metadata],
        'Ticker': [d[3] for d in metadata],
        'P_Trend': all_probs,
        'Signal': predicted_signal
    })

    # 4. 保存结果
    output_path = "trend_base_scores.csv"
    df_scores.to_csv(output_path, index=False)
    print(f"\n趋势信号生成完成！已保存到 {output_path}。")
    print(f"使用的最佳动态阈值为: {best_threshold:.4f}")
    print(f"总共 {len(df_scores)} 个信号已导出。")
    return df_scores


# 3) Run (修改：将 dataset 传给 generate_trend_scores)
if __name__ == "__main__":
    configs = Configs()

    # 1. 初始化数据 (只加载一次)
    dataset = Dataset(DATA_FOLDER, past_window=configs.seq_len)
    train_loader, val_loader, test_loader = create_loaders(dataset, batch_size=64)

    # 2. 初始化模型
    model = TrendForecaster(configs, prediction_horizon=1).to(DEVICE)

    # 3. 运行训练
    print(f"\n=== 开始训练 (设备: {DEVICE}) ===")
    trained_model, best_threshold_value = train_model(model, train_loader, val_loader, configs, epochs=100, learning_rate=1e-3)

    # 4. 训练结束后，生成基础趋势分数
    # 将加载好的 dataset 对象作为参数传入
    generate_trend_scores(dataset, best_threshold=best_threshold_value)