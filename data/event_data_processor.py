import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# --- 1. 全局 FinBERT 模型和设备设置 ---
FINBERT_MODEL_NAME = '/share/liuyuqing/causal_net/data/finbert_hpc_files'

# 假设 DEVICE 变量已在其他地方定义，这里为了完整性添加一个定义
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

try:
    # 强制本地文件加载
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME, local_files_only=True)
    model = AutoModel.from_pretrained(FINBERT_MODEL_NAME, local_files_only=True).to(DEVICE)
    model.eval()
    print("✅ FinBERT 模型和分词器加载成功（本地模式）。")
except Exception as e:
    print(f"致命错误：FinBERT 模型加载失败: {e}")
    raise

# --- 2. 稳健的数据加载函数（修正分隔符和列名） ---

# ⚠️ 修正列名以匹配 Tab 分隔后的字段顺序和数量 (7个字段)
# 实际数据顺序: date, datetime_col, stock_ticker, company_name, title, summary, link
COLUMN_NAMES = ['date', 'datetime_col', 'stock_ticker', 'company_name', 'title', 'summary', 'link']
NUM_COLUMNS = len(COLUMN_NAMES)
SEPARATORS_TO_TRY = ['\t', '|', ';', ',']  # 确保 \t 优先


def load_all_event_files(data_dir: str) -> pd.DataFrame:
    """
    遍历指定目录下的所有事件 CSV 文件，并强制处理 Tab 分隔符和缺失的头行。
    """
    all_files = []

    print(f"Scanning directory: {data_dir}")

    for root, _, files in os.walk(data_dir):
        for file_name in tqdm(files, desc="Loading raw files"):
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                df = None

                for sep in SEPARATORS_TO_TRY:
                    try:
                        # 强制使用 header=None, names=COLUMN_NAMES
                        df = pd.read_csv(
                            file_path,
                            sep=sep,
                            engine='python',
                            header=None,
                            names=COLUMN_NAMES,
                            on_bad_lines='skip'
                        )

                        # 检查列数是否匹配 (7列) 且数据不为空 (这是成功读取的标志)
                        if df.shape[1] == NUM_COLUMNS and not df.empty:
                            # print(f"  --> Successfully read {file_name} with separator: '{sep}'")
                            break
                        else:
                            df = None
                            continue

                    except Exception:
                        df = None
                        continue

                if df is not None and not df.empty:
                    all_files.append(df)
                # else:
                # print(f"Warning: Could not read file {file_name}")

    if not all_files:
        print("Error: No files were loaded successfully.")
        return pd.DataFrame(columns=COLUMN_NAMES)

    final_raw_df = pd.concat(all_files, ignore_index=True)

    # 📢 关键数据清理和对齐准备
    if not final_raw_df.empty:
        # 1. Ticker 规范化：使用正确的 'stock_ticker' 列，并转为大写
        final_raw_df['ticker'] = final_raw_df['stock_ticker'].astype(str).str.upper()

        # 2. 日期规范化：第 1 列 (date) 本身已经是干净的日期，确保它是字符串
        final_raw_df['date'] = final_raw_df['date'].astype(str)

        # 3. 丢弃不需要的列并进行简单清洗
        final_raw_df = final_raw_df.drop(columns=['datetime_col', 'company_name', 'stock_ticker', 'link'],
                                         errors='ignore')
        final_raw_df = final_raw_df.dropna(subset=['date', 'ticker', 'title', 'summary'])

        print("DEBUG: 事件数据 Ticker 和 Date 格式已清理。")

    print(f"\nSuccessfully loaded and merged {len(all_files)} files.")
    print(f"Total rows in raw event data: {len(final_raw_df)}")

    return final_raw_df


# --- 3. 新闻文本聚合函数 (保持不变) ---

def aggregate_news_text(raw_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 'date' 和 'ticker' 聚合新闻文本。
    """
    print("\n--- 步骤 1/2: 聚合事件文本 ---")

    # 注意：现在 raw_data_df 中包含了正确的 'date' 和 'ticker' 列
    raw_data_df['full_text'] = raw_data_df['title'].astype(str) + ' [SEP] ' + raw_data_df['summary'].fillna('').astype(
        str)

    # 按日期和股票代码分组，并将所有 full_text 连接起来
    aggregated_df = raw_data_df.groupby(['date', 'ticker']).agg(
        aggregated_text=('full_text', lambda x: ' '.join(x.astype(str)))
    ).reset_index()

    # 避免极长文本 (BERT限制512 tokens，这里粗略地限制字符数)
    MAX_CHAR_LENGTH = 1000
    aggregated_df['aggregated_text'] = aggregated_df['aggregated_text'].str.slice(0, MAX_CHAR_LENGTH)

    print(f"聚合完成。得到 {len(aggregated_df)} 个 (日期, 股票) 事件。")
    return aggregated_df


# --- 4. FinBERT 编码函数 (保持不变) ---

def encode_texts_to_embeddings(texts: pd.Series, batch_size: int = 64) -> list:
    """
    使用 FinBERT 模型批量编码文本，并返回 [CLS] Token 的隐藏状态作为嵌入向量。
    """
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="--- 步骤 2/2: FinBERT 编码中"):
        batch_texts = texts.iloc[i:i + batch_size].tolist()

        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)

            # 策略：提取 [CLS] Token 的隐藏状态作为文本嵌入 (索引 0)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)

        except Exception as e:
            # 批处理失败容错处理
            print(f"\n[Warning] Error in batch {i // batch_size}: {e}. Filling with zero vectors.")
            num_failed = len(batch_texts)
            zero_embedding = torch.zeros(model.config.hidden_size).cpu().numpy()
            embeddings.extend([zero_embedding] * num_failed)

    return embeddings


# --- 5. 流程控制主函数 (保持不变) ---

def process_full_event_data(raw_data_df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """
    FinBERT 处理的流程总控：聚合 -> 编码 -> 结果合并。
    """
    if raw_data_df.empty:
        print("输入数据为空，跳过 FinBERT 处理。")
        return pd.DataFrame()

    # 1. 文本聚合
    aggregated_df = aggregate_news_text(raw_data_df)

    # 2. 批量编码
    text_series = aggregated_df['aggregated_text']
    embeddings = encode_texts_to_embeddings(text_series, batch_size=batch_size)

    # 3. 将嵌入向量添加到 DataFrame
    aggregated_df['event_embedding'] = embeddings

    # 4. 清理中间文本列
    final_embedded_df = aggregated_df.drop(columns=['aggregated_text'])

    print(f"\n处理完成。嵌入向量已生成。")
    return final_embedded_df


# --- 6. 主执行块 (保持不变) ---

if __name__ == '__main__':
    # ⚠️ 请确保这个路径是正确的 CMIN-US 数据集路径
    EVENT_DATA_PATH = "/share/liuyuqing/causal_net/data/CMIN-Dataset-main/CMIN-US/news/raw"
    OUTPUT_FILE = '/share/liuyuqing/causal_net/cmin_US_event_embeddings_processed.pkl'  # 确保输出路径正确

    print("--- 1. 启动原始事件数据加载 ---")
    raw_event_data_df = load_all_event_files(EVENT_DATA_PATH)

    # 2. 进行 FinBERT 处理
    if not raw_event_data_df.empty:
        print("\n--- 2. 启动 FinBERT 编码和聚合 ---")

        final_embedded_df = process_full_event_data(raw_event_data_df, batch_size=64)

        # 3. 打印和保存结果
        print("\n--- 3. FinBERT 编码完成。 ---")
        print(f"最终事件记录数 (按天/股票聚合后): {len(final_embedded_df)}")
        print("\nDataFrame 头部:")
        print(final_embedded_df.head())

        if 'event_embedding' in final_embedded_df and not final_embedded_df.empty:
            final_embedded_df['date'] = final_embedded_df['date'].astype(str)

            final_embedded_df.to_pickle(OUTPUT_FILE)
            print(f"\n 成功将嵌入结果保存到: {OUTPUT_FILE}")
    else:
        print("无法加载原始数据，FinBERT 编码流程中止。")