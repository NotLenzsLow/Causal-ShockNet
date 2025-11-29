import torch
import torch.nn as nn
from layers.PDM import PastDecomposableMixing

class TrendForecaster(nn.Module):
    def __init__(self, configs, prediction_horizon=1):
        super(TrendForecaster, self).__init__()
        self.configs = configs
        self.prediction_horizon = prediction_horizon
        # 1. 核心特征提取器
        self.feature_extractor = PastDecomposableMixing(configs)
        # 2. 下采样层 (用于创建多尺度输入)
        self.down_sampling_list = nn.ModuleList()
        for i in range(configs.down_sampling_layers):
            self.down_sampling_list.append(
                nn.AvgPool1d(
                    kernel_size=configs.down_sampling_window,
                    stride=configs.down_sampling_window
                )
            )
        # 3. 趋势预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(configs.d_model, 1),  # 输入d_model维度特征
        )
    def forward(self, x_input, return_feature=False):
        # x_input 的形状: [Batch, SeqLen, d_model]
        #1：创建多尺度输入 x_list ---
        x_list = [x_input]
        x_temp = x_input.permute(0, 2, 1)  # 变为 [Batch, d_model, SeqLen]
        for down_layer in self.down_sampling_list:
            x_temp = down_layer(x_temp)
            x_list.append(x_temp.permute(0, 2, 1))  # 变回 [Batch, T_new, d_model]
        # 通过PDM提取特征
        out_list, out_trend_list = self.feature_extractor(x_list)
        # 3：使用全连接层预测 ---
        # out_trend_list[0] 是原始分辨率的趋势
        trend_features = out_trend_list[0]
        # 取最后一个时间点的特征用于预测
        # [Batch, SeqLen, d_model] -> [Batch, d_model]
        last_timestep_features = trend_features[:, -1, :]
        # 仅计算 Logits (原始分数)
        prediction_logits = self.prediction_head(last_timestep_features)

        if return_feature:
            # 供 CausalShockNet 使用：返回 Logits 和惯性特征
            return prediction_logits, last_timestep_features
        else:
            # 供 run_trend_model 的 BCEWithLogitsLoss 使用：返回 Logits
            return prediction_logits
