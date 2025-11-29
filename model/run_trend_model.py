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


# ---------------------------------------------------------
# 1. 新增功能: 导出用于 ShockNet 的数据集 (pkl格式)
# ---------------------------------------------------------
def export_dataset_for_shocknet(dataset, output_path="data/cmin_price_label_data.pkl"):
    """
    将 Dataset 中的数据（特征、真实标签、因果基线、真实冲击、日期、Ticker）导出为 Pickle 文件，
    供 Part 2 (ShockNet) 的 FinancialShockDataset 读取。
    """
    print(f"\n=== 正在导出用于 ShockNet 的价格与标签数据 ===")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data_list = []

    try:
        # 遍历数据集
        # ⚠️ 注意: 这里依赖于您已经在 trend_data_processor.py 中修改了 Dataset 类，
        # 使其返回 (x, y_actual, y_base, y_shock, date, ticker) 6个元素
        print("正在转换数据格式 (可能需要几秒钟)...")
        for item in dataset.data:
            # 解包数据
            x, y_act, y_base, y_shock, date, ticker = item

            data_list.append({
                'date': str(date.date()) if hasattr(date, 'date') else str(date),  # 格式化日期
                'ticker': ticker,
                'X_history': x.numpy(),  # 转为 numpy 保存以减小体积和通用性
                'Y_Actual': y_act.item(),  # 真实涨跌 (0/1)
                'Y_Base_Logits': y_base.item(),  # 惯性基线 Logits
                'Y_Shock_Value': y_shock.item()  # 真实冲击值 (Raw Return)
            })

        df = pd.DataFrame(data_list)

        # 保存为 Pickle
        df.to_pickle(output_path)
        print(f"✅ 导出成功! 文件已保存至: {output_path}")
        print(f"   总样本数: {len(df)}")
        print(f"   包含列名: {df.columns.tolist()}")

    except ValueError as e:
        print(f"❌ 导出失败: 数据集解包错误。")
        print(f"   原因可能是 trend_data_processor.py 中的 Dataset 类尚未更新为返回 6 个元素。")
        print(f"   错误详情: {e}")
    except Exception as e:
        print(f"❌ 导出过程中发生未知错误: {e}")
# ---------------------------------------------------------
# 2. 训练循环 (保持不变)
# ---------------------------------------------------------
def train_model(model, train_loader, val_loader, configs, epochs=50, learning_rate=1e-3):
    DEVICE = next(model.parameters()).device
    positive_weight = torch.tensor(0.908, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = -1
    best_threshold_value = 0.5  # 初始化

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0

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

        # --- 评估 ---
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

        print(
            f"Epoch {epoch + 1}: TrainLoss={avg_train_loss:.6f} | ValF1={val_metrics['f1']:.4f} | AUC={val_metrics['auc']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_threshold_value = val_metrics['threshold']
            save_model(model, "best_direction_model.pth")
            print(f" 保存最优模型（F1={best_f1:.4f}）")

    return model, best_threshold_value


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

    # 修改数据提取逻辑以适配 6 元素元组
    X_all = torch.stack([full_dataset.data[i][0] for i in range(len(full_dataset.data))])
    # 只需要 X 用于预测，不需要 Y

    # 创建只包含 X 的 DataLoader
    all_loader = DataLoader(TensorDataset(X_all), batch_size=256, shuffle=False)
    all_probs = []

    with torch.no_grad():
        for batch in all_loader:
            X = batch[0].to(DEVICE)
            logits = model(X)
            pred_prob = torch.sigmoid(logits).view(-1)
            all_probs.extend(pred_prob.cpu().numpy())

    # 提取元数据 (日期和Ticker在第4和第5个位置)
    # data item: (x, y_act, y_base, y_shock, date, ticker)
    metadata = full_dataset.data

    predicted_signal = (np.array(all_probs) > best_threshold).astype(int)

    df_scores = pd.DataFrame({
        'Target_Date': [d[4] for d in metadata],  # 注意索引变化：date 是 index 4
        'Ticker': [d[5] for d in metadata],  # 注意索引变化：ticker 是 index 5
        'P_Trend': all_probs,
        'Signal': predicted_signal
    })

    df_scores.to_csv("trend_base_scores.csv", index=False)
    print(f"趋势信号生成完成！")
    return df_scores


# 3) Run (修改：将 dataset 传给 generate_trend_scores)
if __name__ == "__main__":
    configs = Configs()

    # 1. 初始化数据 (加载 CSV，计算特征和标签)
    # ⚠️ 此时 Dataset 应该已经包含了 Y_Base 和 Y_Shock 的计算逻辑
    dataset = Dataset(DATA_FOLDER, past_window=configs.seq_len)

    # --- 【新增】步骤 1.5: 导出用于 ShockNet 的数据 ---
    # 利用刚刚加载好的 dataset，直接导出，无需再次读取 CSV
    export_dataset_for_shocknet(dataset, output_path="data/cmin_US_price_label_data.pkl")
    # ---------------------------------------------------

    # 2. 创建 DataLoader 用于 TrendNet 训练
    # ⚠️ 确保 trend_data_processor.py 中的 create_loaders 函数
    # 只提取 dataset.data[i][0] (X) 和 dataset.data[i][1] (y_actual)
    train_loader, val_loader, test_loader = create_loaders(dataset, batch_size=64)

    # 3. 初始化模型
    model = TrendForecaster(configs, prediction_horizon=1).to(DEVICE)

    # 4. 运行训练
    print(f"\n=== 开始 TrendNet 训练 (设备: {DEVICE}) ===")
    trained_model, best_threshold_value = train_model(model, train_loader, val_loader, configs, epochs=100,
                                                      learning_rate=1e-3)

    # 5. 生成基础趋势分数
    generate_trend_scores(dataset, best_threshold=best_threshold_value)