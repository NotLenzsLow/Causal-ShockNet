import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BaseTrend_model import TrendForecaster


class NeuralMemoryBank(nn.Module):
    """历史事件模式库"""

    def __init__(self, memory_slots, feature_dim):
        super(NeuralMemoryBank, self).__init__()
        self.memory_slots = memory_slots
        self.feature_dim = feature_dim

        # 记忆键值对
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, feature_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_slots, feature_dim))

        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)

    def forward(self, query_vector):
        # 1. 计算相似度 [Batch, Slots]
        scores = torch.matmul(query_vector, self.memory_keys.t())
        attn_weights = F.softmax(scores, dim=-1)

        # 2. 加权融合 [Batch, Dim]
        memory_output = torch.matmul(attn_weights, self.memory_values)
        return memory_output, attn_weights


class ShockNet(nn.Module):
    def __init__(self, configs):
        super(ShockNet, self).__init__()

        self.d_trend = configs.d_model
        self.d_text = configs.d_text
        self.d_hidden = configs.d_hidden  # 128
        self.memory_slots = configs.memory_slots

        # --- 1. 动态加权聚合模块 ---
        self.trend_proj = nn.Linear(self.d_trend, self.d_text)
        self.gating_layer = nn.Sequential(
            nn.Linear(self.d_text * 2, self.d_text),
            nn.ReLU(),
            nn.Linear(self.d_text, self.d_text),
            nn.Sigmoid()
        )

        # --- 2. 局部查询构造层 ---
        self.query_encoder = nn.Sequential(
            nn.Linear(self.d_text + self.d_trend, self.d_hidden),
            nn.LayerNorm(self.d_hidden),
            nn.ReLU()
        )

        # --- 3. 历史事件模式库 ---
        self.memory_bank = NeuralMemoryBank(self.memory_slots, self.d_hidden)

        # --- 4. 融合与推理网络 ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.d_hidden * 2, self.d_hidden),
            nn.LayerNorm(self.d_hidden),
            nn.ReLU()
        )

        # 5. 冲击目标嵌入网络 E(.)
        # 📢 修正：输出维度必须是 d_hidden (128) 以便计算 Triplet Loss
        self.target_embedding_net = nn.Sequential(
            nn.Linear(1, 64),  # 输入是标量 (1)
            nn.ReLU(),
            nn.Linear(64, self.d_hidden)  # 输出 128
        )

        # 6. 推理头
        self.inference_head = nn.Sequential(
            nn.Linear(self.d_hidden, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()  # 输出冲击值 [-1, 1]
        )

    def forward(self, trend_features, event_embeddings, intervention_type='factual'):
        """
        Returns: shock_pred, fused_feat (特征嵌入), query_vector
        """

        # 1.1 动态加权聚合
        trend_projected = self.trend_proj(trend_features)
        combined_for_gate = torch.cat([trend_projected, event_embeddings], dim=-1)
        gate = self.gating_layer(combined_for_gate)

        # 1.2 反事实干预逻辑
        if intervention_type == 'counterfactual':
            # 反事实：强制将事件特征置零，模拟无事件发生
            # 这里的 shock_features 实际上只有趋势信息投射过来的影子
            shock_features = torch.zeros_like(event_embeddings)
        else:
            # 事实：正常计算，包含事件信息
            shock_features = event_embeddings * gate

        # 2. 局部查询构造
        concat_features = torch.cat([shock_features, trend_features], dim=-1)
        query_vector = self.query_encoder(concat_features)

        # 3. 记忆检索 (激活)
        memory_feat, _ = self.memory_bank(query_vector)

        # 4. 融合与推理
        final_feat_input = torch.cat([query_vector, memory_feat], dim=-1)

        # fused_feat 是用于计算 Triplet/Ortho Loss 的核心向量
        fused_feat = self.fusion_layer(final_feat_input)

        shock_pred = self.inference_head(fused_feat)

        return shock_pred, fused_feat, query_vector


# --- CausalShockNet 最终集成模型 ---
class CausalShockNet(nn.Module):
    def __init__(self, configs):
        super(CausalShockNet, self).__init__()
        self.configs = configs

        # 1. 基础趋势模型 (用于提取 pure inertia)
        # 假设您已经有训练好的 TrendForecaster，或者在这里重新初始化
        self.trend_model = TrendForecaster(configs)

        # 2. 事件编码器 (将 768 维 BERT 向量映射到 d_model 维度)
        self.event_encoder = nn.Sequential(
            nn.Linear(configs.d_text, configs.d_hidden),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_hidden, configs.d_model)
        )

        # 3. 融合门控机制 (Gating / Mixer)
        # 用于将事件冲击注入到趋势特征中
        self.fusion_gate = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.Sigmoid()
        )

        # 🔥🔥🔥【核心修改点 1】新增 LayerNorm 层 🔥🔥🔥
        # 强制特征归一化，解决模长失衡问题
        # 这里的 d_model 应该是 15
        self.feature_norm = nn.LayerNorm(configs.d_model)

        # 4. 最终分类预测头 (根据融合后的特征预测涨跌)
        self.classifier = nn.Linear(configs.d_model, 1)

        # 5. 辅助网络: 用于 Triplet Loss 的 Anchor Embedding (将标量 Shock Value 映射为向量)
        # 这样才能计算 Triplet Loss: Dist(Anchor, Positive) vs Dist(Anchor, Negative)
        self.target_embedding_net = nn.Sequential(
            nn.Linear(1, configs.d_hidden),
            nn.ReLU(),
            nn.Linear(configs.d_hidden, configs.d_model)
            # 注意：Anchor 输出最好也过一下 LayerNorm，或者在 Loss 外面归一化
        )

    def forward(self, x_history, event_emb):
        """
        x_history: [Batch, Seq_Len, Features]
        event_emb: [Batch, d_text]
        """

        # --- A. 反事实分支 (Counterfactual Branch) ---
        # 也就是：如果没有发生事件，市场原本的趋势 (Pure Trend)
        # 从趋势模型中获取特征。注意：BaseTrend_model 需要支持 return_feature=True
        trend_logits, feat_cf_raw = self.trend_model(x_history, return_feature=True)

        # --- B. 事实分支 (Factual Branch) ---
        # 也就是：真实发生的世界 (Trend + Shock)

        # 1. 编码事件
        e_emb = self.event_encoder(event_emb)  # [Batch, d_model]

        # 2. 融合机制 (这里是一个简单的加法或门控示例，具体视您原逻辑而定)
        # 拼接 趋势特征 和 事件特征
        combined = torch.cat([feat_cf_raw, e_emb], dim=1)
        gate = self.fusion_gate(combined)

        # 事实特征 = 趋势 + 门控 * 事件冲击
        # (残差连接思想，保留底色，加入冲击)
        feat_f_raw = feat_cf_raw + gate * e_emb

        # 🔥🔥🔥【核心修改点 2】强制应用 LayerNorm 🔥🔥🔥
        # 这就是破局的关键！强制拉平两个向量的模长。
        feat_cf = self.feature_norm(feat_cf_raw)
        feat_f = self.feature_norm(feat_f_raw)

        # --- C. 预测输出 ---
        # 使用归一化后的“事实特征”进行最终预测
        final_pred_logits = self.classifier(feat_f)
        final_pred_prob = torch.sigmoid(final_pred_logits)

        # 返回内容说明:
        # final_pred_prob: 用于计算 MCC, F1
        # final_pred_logits: 用于计算 BCE Loss (L_Pred)
        # trend_logits: 用于计算 L_Base (基线损失)
        # None, None: 占位符 (如果有 shock_f, shock_cf 单独输出可放这里)
        # feat_f: 归一化后的事实特征 (用于 L_Ortho, L_Triplet) -> Norm ~ 3.8
        # feat_cf: 归一化后的反事实特征 (用于 L_Ortho, L_Triplet) -> Norm ~ 3.8
        # None, None: 占位符

        return (final_pred_prob, final_pred_logits, trend_logits,
                None, None,
                feat_f, feat_cf,
                None, None)