# FinAgent-R1：金融多工具 Agent 系统 — 完整项目实施指南

> **项目定位：** 基于 Search-R1 (COLM 2025, GitHub 3.8k★) 的 Agentic RL 框架，
> 训练一个能够自主使用「知识检索 + 金融计算 + SQL 表格查询」三种工具的金融研报问答 Agent。
> 单卡 RTX 3090/4090 完成全部训练，最终交付可部署的全栈 Demo + 开源仓库。

---

## 目录

- [第一章：项目总览与核心思路](#第一章项目总览与核心思路)
- [第二章：环境搭建](#第二章环境搭建)
- [第三章：数据准备](#第三章数据准备)
- [第四章：三个工具后端实现](#第四章三个工具后端实现)
- [第五章：当前训练主线说明（veRL）](#第五章当前训练主线说明verl)
- [第六章：veRL 多轮 Agent RL 训练（当前主线）](#第六章verl-多轮-agent-rl-训练当前主线)
- [第七章：评测与消融实验](#第七章评测与消融实验)
- [第八章：当前 Demo 与部署状态](#第八章当前-demo-与部署状态)
- [第九章：开源仓库整理与发布](#第九章开源仓库整理与发布)
- [第十章：简历撰写与面试准备](#第十章简历撰写与面试准备)

---

# 第一章：项目总览与核心思路

## 1.1 为什么 Search-R1 就是 Agent？

Search-R1 不是"更好的 RAG"。它的核心是 **Agent-Environment 交互范式**：

```
Agent (LLM)          Environment (工具)
  │                      │
  ├── <think>推理</think>  │
  ├── <search>查询</search>──→ 调用搜索引擎
  │                      ├── <information>结果</information>
  ├── <think>继续推理</think>│
  ├── <answer>最终答案</answer>│
  │                      │
  └──────── reward ←─────┘  (答案是否正确)
```

用 MDP 形式化：
- **State (s_t)**：当前推理上下文（问题 + 已有的思考和工具返回）
- **Action (a_t)**：生成推理文本 / 调用某个工具 / 给出最终答案
- **Environment**：工具集合（搜索、计算、SQL）
- **Observation (o_t)**：工具返回的结果
- **Reward (r)**：最终答案的准确性（Exact Match / 程序执行结果匹配）

## 1.2 你的创新点：从单工具 → 多工具 Agent

Search-R1 原版只有搜索一个工具。你要扩展为三个工具：

| Agent Action 标签 | 工具类型 | 功能 | 数据源 |
|------------------|---------|------|--------|
| `<search>query</search>` | 知识检索 | 从研报段落中检索相关信息 | FinQA 研报文本 + BM25/BGE |
| `<calculate>expr</calculate>` | 金融计算 | 执行财务公式（PE、ROE、CAGR、同比增长等） | Python 沙箱 |
| `<sql>query</sql>` | 表格查询 | 查询结构化财务数据 | FinQA 表格 → SQLite |

**关键：Agent 自主决定用哪个工具，不是硬编码的。**

## 1.3 时间规划

```
总计：10-14 天（每天 3-4 小时）

Day 1      ：环境搭建 + 依赖安装
Day 2      ：数据下载与预处理
Day 3      ：构建三个工具后端
Day 4      ：训练数据转换为 veRL parquet + 工具联调
Day 5-8    ：Search-R1 风格多轮 Agent RL（veRL 主线，可能需要多次调参）
Day 9-10   ：评测 + 消融实验
Day 11-12  ：全栈 Demo 开发
Day 13-14  ：开源整理 + README + 文档
```

## 1.4 硬件需求与显存预算

```
RTX 3090 (24GB) 或 RTX 4090 (24GB)

| 组件                              | 显存占用  |
|----------------------------------|----------|
| Qwen2.5-3B (4-bit QLoRA via Unsloth) | ~4 GB   |
| LoRA adapters (r=16)              | ~0.5 GB  |
| vLLM rollout (共享显存)             | ~6 GB   |
| GRPO 训练开销 + Gradient Checkpointing | ~8 GB   |
| 检索模型 BGE-base-zh (推理)         | ~0.5 GB  |
| 总计                              | ~19 GB   |
```

---

# 第二章：环境搭建

## 2.1 操作系统要求

- Ubuntu 20.04 / 22.04（推荐）
- CUDA 12.1+
- Python 3.10+

## 2.2 创建 Conda 虚拟环境

```bash
# 创建环境
conda create -n finagent python=3.10 -y
conda activate finagent

# 确认 CUDA 版本
nvidia-smi
nvcc --version
```

## 2.3 安装 PyTorch

```bash
# CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2.4 安装核心 ML 依赖

```bash
# Hugging Face 全家桶
pip install transformers==4.45.0
pip install datasets==3.0.0
pip install accelerate==1.0.0
pip install peft==0.13.0       # LoRA
pip install trl==0.12.0        # GRPOTrainer

# 推理加速
pip install vllm==0.6.3

# 实验追踪
pip install wandb
pip install ray==2.39.0
pip install hydra-core==1.3.2
pip install codetiming==1.4.0
pip install pandas==2.2.0 pyarrow==17.0.0 dill==0.3.8

# 验证安装
python -c "from transformers import AutoModelForCausalLM; print('transformers OK')"
python -c "from trl import GRPOTrainer; print('trl OK')"
python -c "from peft import LoraConfig; print('peft OK')"
python -c "import ray, hydra; print('ray/hydra OK')"
```

## 2.5 安装 Unsloth（可选，不是当前主线）

```bash
# Unsloth 单独安装（和上面的 trl 兼容）
pip install unsloth

# 验证
python -c "from unsloth import FastLanguageModel; print('unsloth OK')"
```

当前仓库的正式训练主线已经切到 veRL / Search-R1 风格的多轮 Agent RL。
如果 Unsloth 安装报错，可以跳过；它现在只算可选依赖，不是默认训练入口。

## 2.6 安装检索相关依赖

```bash
# BM25 检索
pip install pyserini

# 需要 Java（Pyserini 依赖 Lucene）
# Ubuntu:
sudo apt-get update && sudo apt-get install -y openjdk-21-jdk
java -version  # 确认 Java 21+

# 向量检索
pip install faiss-gpu          # GPU 版 FAISS
pip install sentence-transformers  # BGE 编码模型

# 验证
python -c "import faiss; print('faiss OK, GPU:', faiss.get_num_gpus())"
python -c "from pyserini.search.lucene import LuceneSearcher; print('pyserini OK')"
```

如果构建 BM25 时出现 `Module jdk.incubator.vector not found`，通常是因为当前 shell
仍在使用 Java 11/17 等较低版本；切换到 JDK 21 后重试。

## 2.7 安装数据库依赖

```bash
# SQLite 已内置于 Python，无需安装
# DuckDB（可选，比 SQLite 更快处理分析查询）
pip install duckdb

# 验证
python -c "import sqlite3; print('sqlite3 OK')"
python -c "import duckdb; print('duckdb OK')"
```

## 2.8 安装 Web 开发依赖

```bash
# 后端 / 检索服务
pip install fastapi==0.115.0
pip install uvicorn[standard]
pip install sse-starlette      # Server-Sent Events

# 前端（需要 Node.js）
# Ubuntu:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version  # 确认 v20+
npm --version
```

## 2.9 引入 Search-R1 / veRL 运行时

当前仓库不是“额外 clone 一份 Search-R1 仅供参考”，而是直接把 Search-R1 vendored 到
`vendor/Search-R1/`，并由 `training/search_r1_compat.py` 在运行时加入 `sys.path`。

```bash
# 当前仓库的真实依赖关系
ls vendor/Search-R1

# 训练入口会自动把 vendor/Search-R1 加入 Python 路径
python -c "from training.search_r1_compat import ensure_search_r1_on_path; print(ensure_search_r1_on_path())"
```

如果你是从零搭项目，建议直接把 Search-R1 作为 vendor 目录纳入仓库，
而不是在仓库外再维护一份独立副本。

## 2.10 项目目录结构

```bash
# 创建完整目录结构
mkdir -p data/{raw,processed,corpus,indexes,tables}
mkdir -p tools
mkdir -p training
mkdir -p eval
mkdir -p demo/{backend,frontend}
mkdir -p configs
mkdir -p checkpoints/{sft,grpo}
mkdir -p docs
mkdir -p scripts

# 创建空文件占位
touch tools/__init__.py
touch training/__init__.py
touch eval/__init__.py
```

最终结构：
```
FinAgent-R1/
├── data/
│   ├── raw/                    # 原始下载的数据
│   ├── processed/              # 处理后的训练/测试数据
│   ├── corpus/                 # 检索语料库
│   ├── indexes/                # BM25 + FAISS 索引
│   └── tables/                 # SQLite 数据库
├── tools/
│   ├── __init__.py
│   ├── search_tool.py          # Tool 1: 知识检索
│   ├── calculator_tool.py      # Tool 2: 金融计算
│   ├── sql_tool.py             # Tool 3: SQL 查询
│   ├── tool_dispatcher.py      # 统一工具调度器
│   └── retrieval_server.py     # 检索 FastAPI 服务
├── training/
│   ├── __init__.py
│   ├── reward_functions.py     # 奖励函数
│   ├── finagent_generation.py  # veRL 多轮 generation manager
│   ├── finagent_verl_main.py   # veRL / GRPO 主训练入口
│   ├── search_r1_compat.py     # vendored Search-R1 兼容层
│   ├── tensor_helper.py        # veRL 张量辅助函数
│   ├── sft_coldstart.py        # 可选：SFT 冷启动训练
│   ├── grpo_train.py           # 可选：纯 TRL GRPO 训练
│   ├── grpo_train_unsloth.py   # 可选：Unsloth + TRL GRPO 训练
│   └── void_turn_filter.py     # 空轮次检测与过滤
├── eval/
│   ├── __init__.py
│   ├── evaluate.py             # 主评测脚本
│   └── ablation.py             # 消融实验
├── demo/
│   ├── backend/
│   │   ├── main.py             # FastAPI 后端
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── package.json
│   │   ├── index.html
│   │   └── src/
│   │       ├── App.jsx         # React 主入口
│   │       ├── api.js          # 后端 API 调用封装
│   │       ├── main.jsx
│   │       ├── styles.css
│   │       └── components/
│   │           ├── ChatInterface.jsx    # 聊天界面
│   │           ├── ReasoningPanel.jsx   # 推理过程展示
│   │           └── ToolTrace.jsx        # 工具调用链路可视化
│   └── gradio_app.py           # Gradio 前端 Demo
├── configs/
│   ├── sft_config.yaml         # SFT 训练配置
│   ├── grpo_config.yaml        # TRL GRPO 训练配置
│   └── verl_ppo_finqa.yaml     # 当前 veRL 主训练配置
├── scripts/
│   ├── download_data.sh        # 数据下载脚本
│   ├── explore_data.py         # 数据探索
│   ├── prepare_training_data.py
│   ├── build_corpus.py
│   ├── build_sql_database.py
│   ├── build_question_table_map.py
│   ├── build_bm25_index.sh
│   ├── build_dense_index.py
│   ├── build_indexes.sh
│   ├── prepare_verl_finqa_data.py
│   ├── train_grpo_verl.sh      # veRL 训练启动脚本
│   ├── test_tools.py
│   ├── start_services.sh       # 启动所有服务
│   ├── generate_sft_data.py    # 可选：SFT 种子数据生成
│   └── test_sft_model.py       # 可选：SFT 模型快速验证
├── vendor/Search-R1/           # vendored Search-R1 / veRL runtime
├── docker-compose.yml          # Docker 编排文件
├── Dockerfile.backend          # FastAPI 后端镜像
├── Dockerfile.retrieval        # 检索服务镜像
├── Dockerfile.demo             # Gradio Demo 镜像
├── README.md
├── requirements.txt
└── .gitignore
```

## 2.11 配置 Weights & Biases

```bash
wandb login
# 输入你的 API key（从 https://wandb.ai/authorize 获取）
# 这会在训练时自动记录 loss、reward 曲线等
```

## 2.12 配置 .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/

# Data & Models（太大，不上传）
data/raw/
data/indexes/
data/tables/
checkpoints/
*.bin
*.safetensors
*.pt

# Env
.env
wandb/

# Node
node_modules/
demo/frontend/build/

# IDE
.vscode/
.idea/
EOF
```

---

# 第三章：数据准备

## 3.1 数据集选择说明

我们使用三个互补的数据集：

| 数据集 | 用途 | 大小 | 特点 |
|--------|------|------|------|
| **FinQA** | 主训练 + 主评测 | 8,281 QA / 2,789 报告 | 每个问题带标注推理程序，可精确计算 reward |
| **ConvFinQA** | 补充训练 | 3,892 多轮对话 | 多步推理链，测试 Agent 多轮交互能力 |
| **TAT-QA** | 扩展评测 | 16,552 QA | 混合文本+表格，测试跨模态推理 |

## 3.2 下载 FinQA 数据集

```bash
# scripts/download_data.sh

#!/bin/bash
set -e

echo "=== 下载 FinQA 数据集 ==="
cd data/raw

# 方法 1: 从 Hugging Face 下载（推荐）
python -c "
from datasets import load_dataset
ds = load_dataset('ibm-research/finqa')
ds.save_to_disk('finqa_hf')
print('FinQA 下载完成')
print(f'  Train: {len(ds[\"train\"])} examples')
print(f'  Validation: {len(ds[\"validation\"])} examples')
print(f'  Test: {len(ds[\"test\"])} examples')
"

# 方法 2: 从 GitHub 下载（备选）
# git clone https://github.com/czyssrs/FinQA.git
# 数据在 FinQA/dataset/ 目录下

echo "=== 下载 ConvFinQA 数据集 ==="
git clone https://github.com/czyssrs/ConvFinQA.git
echo "ConvFinQA 下载完成"

echo "=== 下载 Fino1 推理路径数据（可选，用于 SFT）==="
python -c "
from datasets import load_dataset
ds = load_dataset('TheFinAI/Fino1_Reasoning_Path_FinQA_v2')
ds.save_to_disk('fino1_hf')
print('Fino1 下载完成')
"

echo "=== 所有数据下载完成 ==="
```

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

## 3.3 探索 FinQA 数据结构

```python
# scripts/explore_data.py
"""
理解 FinQA 数据的结构——这一步很重要，
后面所有的工具设计都基于这个数据格式
"""

from datasets import load_from_disk
import json

ds = load_from_disk("data/raw/finqa_hf")

# 查看一个样例
example = ds["train"][0]
print("=" * 60)
print("FinQA 样例结构:")
print("=" * 60)

# 每个样例包含：
print(f"\n1. ID: {example['id']}")

print(f"\n2. pre_text (表格前的文本，前 200 字):")
print(example['pre_text'][:200] + "...")

print(f"\n3. table (财务表格):")
# table 是一个列表的列表（二维数组），第一行是表头
for row in example['table'][:5]:  # 只显示前 5 行
    print(f"   {row}")
if len(example['table']) > 5:
    print(f"   ... (共 {len(example['table'])} 行)")

print(f"\n4. post_text (表格后的文本，前 200 字):")
print(example['post_text'][:200] + "...")

print(f"\n5. question: {example['question']}")

print(f"\n6. answer: {example['answer']}")

print(f"\n7. program (标注的推理程序):")
print(f"   {example['program']}")
# 例如: "subtract(1829, 1731), divide(#0, 1731)" 
# 意思是: (1829-1731)/1731

print(f"\n8. program_re (嵌套格式):")
print(f"   {example['program_re']}")

print(f"\n9. gold_inds (支持事实的索引):")
print(f"   {example['gold_inds']}")

# 统计信息
print("\n" + "=" * 60)
print("数据集统计:")
print(f"  训练集: {len(ds['train'])} 个问答对")
print(f"  验证集: {len(ds['validation'])} 个问答对")
print(f"  测试集: {len(ds['test'])} 个问答对")

# 统计推理程序中的操作类型
from collections import Counter
ops = Counter()
for ex in ds['train']:
    prog = ex['program']
    for op in ['add', 'subtract', 'multiply', 'divide', 'greater', 'exp',
               'table_sum', 'table_average', 'table_max', 'table_min']:
        if op in prog:
            ops[op] += 1

print(f"\n推理操作分布:")
for op, count in ops.most_common():
    print(f"  {op}: {count}")
```

```bash
python scripts/explore_data.py
```

## 3.4 处理 FinQA 数据为训练格式

```python
# scripts/prepare_training_data.py
"""
将 FinQA 数据处理为 Agent RL 训练所需的格式：
- prompt: 包含问题和系统提示
- answer: 标准答案（用于 reward 计算）
- program: 推理程序（用于程序执行准确率评测）
- context: 报告上下文（用于构建检索语料库）
- table: 表格数据（用于构建 SQL 数据库）
"""

import json
import os
from datasets import load_from_disk

ds = load_from_disk("data/raw/finqa_hf")

# ========== Agent System Prompt ==========
AGENT_SYSTEM_PROMPT = """You are a financial analysis agent. You can use the following tools to answer questions about financial reports:

Tools:
1. <search>query</search> - Search the financial report knowledge base for relevant passages
2. <calculate>expression</calculate> - Execute financial calculations (e.g., PE ratio, growth rate, percentage change)
3. <sql>SQL query</sql> - Query structured financial data tables

Rules:
- Use <think>...</think> for reasoning
- Tool results will be returned in <observation>...</observation>
- Give your final answer in <answer>your answer</answer>
- You may call tools multiple times or not at all
- For numerical answers, provide the exact number

Question: {question}"""

# ========== 处理训练数据 ==========
def process_example(example, split):
    """处理单个 FinQA 样例"""

    # 构建上下文（pre_text + table + post_text）
    table_text = ""
    if example['table']:
        headers = example['table'][0] if example['table'] else []
        for row in example['table']:
            table_text += " | ".join(str(cell) for cell in row) + "\n"

    context = f"{example['pre_text']}\n\n[Table]\n{table_text}\n{example['post_text']}"

    return {
        "id": example['id'],
        "prompt": AGENT_SYSTEM_PROMPT.format(question=example['question']),
        "question": example['question'],
        "answer": str(example['answer']),
        "program": example['program'],
        "context": context,
        "table": example['table'],
        "pre_text": example['pre_text'],
        "post_text": example['post_text'],
        "gold_inds": example.get('gold_inds', {}),
    }

# 处理各个 split
os.makedirs("data/processed", exist_ok=True)

for split in ['train', 'validation', 'test']:
    output_file = f"data/processed/{split}.jsonl"
    count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in ds[split]:
            processed = process_example(example, split)
            f.write(json.dumps(processed, ensure_ascii=False) + '\n')
            count += 1

    print(f"{split}: {count} 个样例 → {output_file}")

print("\n训练数据处理完成！")
```

```bash
python scripts/prepare_training_data.py
```

## 3.5 构建检索语料库

```python
# scripts/build_corpus.py
"""
从 FinQA 的报告文本中构建检索语料库
每个报告的 pre_text 和 post_text 作为独立的检索段落
"""

import json
import os
from datasets import load_from_disk

ds = load_from_disk("data/raw/finqa_hf")

os.makedirs("data/corpus", exist_ok=True)

corpus = {}  # 用 dict 去重
passage_id = 0

for split in ['train', 'validation', 'test']:
    for example in ds[split]:
        report_id = example['id'].rsplit('-', 1)[0]  # 提取报告 ID

        # pre_text 作为一个段落
        if example['pre_text'] and len(example['pre_text'].strip()) > 50:
            key = example['pre_text'][:200]  # 用前 200 字符作为去重 key
            if key not in corpus:
                corpus[key] = {
                    "id": f"p_{passage_id}",
                    "contents": example['pre_text'].strip(),
                    "report_id": report_id,
                    "source": "pre_text",
                }
                passage_id += 1

        # post_text 作为一个段落
        if example['post_text'] and len(example['post_text'].strip()) > 50:
            key = example['post_text'][:200]
            if key not in corpus:
                corpus[key] = {
                    "id": f"p_{passage_id}",
                    "contents": example['post_text'].strip(),
                    "report_id": report_id,
                    "source": "post_text",
                }
                passage_id += 1

        # 表格也作为一个段落（文本化）
        if example['table']:
            table_text = ""
            for row in example['table']:
                table_text += " | ".join(str(cell) for cell in row) + "\n"
            if len(table_text.strip()) > 30:
                table_key = table_text[:200]
                if table_key not in corpus:
                    corpus[table_key] = {
                        "id": f"p_{passage_id}",
                        "contents": table_text.strip(),
                        "report_id": report_id,
                        "source": "table",
                    }
                    passage_id += 1

# 写入语料库文件（Pyserini 格式）
output_file = "data/corpus/financial_passages.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in corpus.values():
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"语料库构建完成！")
print(f"  总段落数: {len(corpus)}")
print(f"  输出文件: {output_file}")

# 同时保存 ID → 内容 的映射（供 FAISS 使用）
id_to_content = {item['id']: item['contents'] for item in corpus.values()}
with open("data/corpus/passage_map.json", 'w', encoding='utf-8') as f:
    json.dump(id_to_content, f, ensure_ascii=False)

print(f"  段落映射: data/corpus/passage_map.json")
```

```bash
python scripts/build_corpus.py
```

## 3.6 构建财务表格 SQL 数据库

```python
# scripts/build_sql_database.py
"""
将 FinQA 中的所有财务表格导入 SQLite 数据库
每个报告的表格作为一张独立的表
"""

import json
import sqlite3
import re
import os
from datasets import load_from_disk

ds = load_from_disk("data/raw/finqa_hf")

os.makedirs("data/tables", exist_ok=True)

db_path = "data/tables/financial_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 创建主索引表
cursor.execute("""
    CREATE TABLE IF NOT EXISTS table_index (
        table_name TEXT PRIMARY KEY,
        report_id TEXT,
        question_id TEXT,
        num_rows INTEGER,
        num_cols INTEGER,
        headers TEXT
    )
""")

tables_created = 0
errors = 0

def sanitize_column_name(name):
    """清理列名，使其可以作为 SQL 列名"""
    name = str(name).strip()
    name = re.sub(r'[^\w\s]', '', name)          # 去除特殊字符
    name = re.sub(r'\s+', '_', name)              # 空格替换为下划线
    name = name.strip('_')
    if not name or name[0].isdigit():
        name = 'col_' + name
    return name.lower()[:50]  # 截断过长的列名

def sanitize_value(val):
    """清理单元格值"""
    val = str(val).strip()
    # 尝试提取数字（去除 $, %, 逗号等）
    cleaned = re.sub(r'[$,%()（）]', '', val).strip()
    cleaned = cleaned.replace(',', '')
    try:
        return float(cleaned)
    except ValueError:
        return val

for split in ['train', 'validation', 'test']:
    for example in ds[split]:
        table = example['table']
        if not table or len(table) < 2:  # 至少需要表头 + 1 行数据
            continue

        question_id = example['id']
        report_id = question_id.rsplit('-', 1)[0]
        table_name = f"t_{report_id.replace('-', '_').replace('/', '_')}"

        # 跳过已经创建的表（同一报告的多个问题共享表格）
        cursor.execute("SELECT 1 FROM table_index WHERE table_name = ?", (table_name,))
        if cursor.fetchone():
            continue

        try:
            # 处理表头
            headers = table[0]
            col_names = []
            seen_names = {}
            for h in headers:
                name = sanitize_column_name(h)
                if not name:
                    name = f"col_{len(col_names)}"
                # 处理重名
                if name in seen_names:
                    seen_names[name] += 1
                    name = f"{name}_{seen_names[name]}"
                else:
                    seen_names[name] = 0
                col_names.append(name)

            # 创建表
            col_defs = ", ".join([f'"{c}" TEXT' for c in col_names])
            cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')

            # 插入数据
            placeholders = ", ".join(["?" for _ in col_names])
            for row in table[1:]:
                if len(row) == len(col_names):
                    values = [str(sanitize_value(v)) for v in row]
                    cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', values)

            # 记录到索引表
            cursor.execute("""
                INSERT OR REPLACE INTO table_index VALUES (?, ?, ?, ?, ?, ?)
            """, (
                table_name, report_id, question_id,
                len(table) - 1, len(col_names),
                json.dumps(col_names)
            ))

            tables_created += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: 表 {table_name} 创建失败: {e}")

conn.commit()

# 统计
cursor.execute("SELECT COUNT(*) FROM table_index")
total = cursor.fetchone()[0]

print(f"\nSQL 数据库构建完成！")
print(f"  数据库路径: {db_path}")
print(f"  总表数: {total}")
print(f"  创建成功: {tables_created}")
print(f"  创建失败: {errors}")

# 测试查询
cursor.execute("SELECT table_name, headers FROM table_index LIMIT 3")
for row in cursor.fetchall():
    print(f"\n  示例表: {row[0]}")
    print(f"  列名: {row[1]}")

conn.close()
```

```bash
python scripts/build_sql_database.py
```

## 3.7 创建问题到表格的映射

```python
# scripts/build_question_table_map.py
"""
创建「问题 ID → 表名」的映射
Agent 执行 SQL 查询时需要知道查哪张表
"""

import json
from datasets import load_from_disk

ds = load_from_disk("data/raw/finqa_hf")

question_to_table = {}

for split in ['train', 'validation', 'test']:
    for example in ds[split]:
        question_id = example['id']
        report_id = question_id.rsplit('-', 1)[0]
        table_name = f"t_{report_id.replace('-', '_').replace('/', '_')}"
        question_to_table[question_id] = table_name

with open("data/tables/question_table_map.json", 'w', encoding='utf-8') as f:
    json.dump(question_to_table, f, ensure_ascii=False)

print(f"问题-表格映射完成: {len(question_to_table)} 个映射")
```

```bash
python scripts/build_question_table_map.py
```

---

# 第四章：三个工具后端实现

## 4.1 Tool 1：金融知识检索

### 4.1.1 构建 BM25 索引

```bash
# scripts/build_bm25_index.sh

#!/bin/bash
set -e

echo "=== 构建 BM25 索引 ==="

INDEX_DIR="data/indexes/bm25"
CORPUS_DIR="data/corpus"

mkdir -p $INDEX_DIR

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $CORPUS_DIR \
  --index $INDEX_DIR \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions \
  --storeDocvectors \
  --storeRaw

echo "BM25 索引构建完成！文件: $INDEX_DIR"
```

```bash
chmod +x scripts/build_bm25_index.sh
./scripts/build_bm25_index.sh
```

当前仓库里的 `build_bm25_index.sh` 已经增加了预检查：
- 当前 Python 环境是否安装了 `pyserini`
- 当前 shell 是否正在使用 `Java 21+`
- `data/corpus/financial_passages.jsonl` 是否已经存在

### 4.1.2 构建 BGE 稠密检索索引

```python
# scripts/build_dense_index.py
"""
用 BGE-base-en-v1.5 编码所有段落，构建 FAISS HNSW 索引
注意：FinQA 是英文数据集，用英文版 BGE
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("加载 BGE 编码模型...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

print("读取语料库...")
passages = []
passage_ids = []
with open("data/corpus/financial_passages.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        passages.append(doc["contents"][:512])  # 截断过长的段落
        passage_ids.append(doc["id"])

print(f"总段落数: {len(passages)}")

print("编码段落（这可能需要 5-10 分钟）...")
batch_size = 128
all_embeddings = []
for i in range(0, len(passages), batch_size):
    batch = passages[i:i+batch_size]
    # BGE 推荐在查询前加 "Represent this sentence:" 前缀
    embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    all_embeddings.append(embs)
    if (i // batch_size) % 10 == 0:
        print(f"  进度: {min(i + batch_size, len(passages))}/{len(passages)}")

embeddings = np.vstack(all_embeddings).astype("float32")
dim = embeddings.shape[1]
print(f"嵌入维度: {dim}")

print("构建 FAISS HNSW 索引...")
index = faiss.IndexHNSWFlat(dim, 32)  # M=32, 平衡精度和速度
index.hnsw.efConstruction = 200       # 构建时的搜索深度
index.hnsw.efSearch = 128             # 查询时的搜索深度
index.add(embeddings)

faiss.write_index(index, "data/indexes/dense_hnsw.index")

# 保存 passage_id 的顺序（和 FAISS 索引对应）
with open("data/indexes/faiss_id_map.json", 'w', encoding='utf-8') as f:
    json.dump(passage_ids, f)

print(f"FAISS 索引构建完成！")
print(f"  索引文件: data/indexes/dense_hnsw.index")
print(f"  ID 映射: data/indexes/faiss_id_map.json")
```

```bash
python scripts/build_dense_index.py
```

### 4.1.3 检索工具实现

```python
# tools/search_tool.py
"""
Tool 1: 金融知识检索
支持 BM25 + Dense 混合检索
"""

import json
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class FinancialSearchTool:
    def __init__(self,
                 bm25_index_path="data/indexes/bm25",
                 dense_index_path="data/indexes/dense_hnsw.index",
                 passage_map_path="data/corpus/passage_map.json",
                 faiss_id_map_path="data/indexes/faiss_id_map.json",
                 topk=3):

        # BM25
        self.bm25 = LuceneSearcher(bm25_index_path)
        self.topk = topk

        # Dense
        self.dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.dense_index = faiss.read_index(dense_index_path)

        with open(passage_map_path, 'r', encoding='utf-8') as f:
            self.passage_map = json.load(f)

        with open(faiss_id_map_path, 'r', encoding='utf-8') as f:
            self.faiss_ids = json.load(f)

    def search_bm25(self, query: str) -> list:
        hits = self.bm25.search(query, k=self.topk)
        results = []
        for hit in hits:
            raw = json.loads(hit.raw)
            results.append({
                "id": hit.docid,
                "text": raw["contents"][:500],
                "score": float(hit.score),
                "source": "bm25",
            })
        return results

    def search_dense(self, query: str) -> list:
        q_emb = self.dense_model.encode(
            [f"Represent this sentence: {query}"],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.dense_index.search(q_emb, self.topk)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.faiss_ids):
                continue
            pid = self.faiss_ids[idx]
            text = self.passage_map.get(pid, "")
            results.append({
                "id": pid,
                "text": text[:500],
                "score": float(score),
                "source": "dense",
            })
        return results

    def search_hybrid(self, query: str) -> list:
        """混合检索：BM25 + Dense 结果融合"""
        bm25_results = self.search_bm25(query)
        dense_results = self.search_dense(query)

        # 简单融合：合并去重，按分数排序
        all_results = bm25_results + dense_results
        seen = set()
        deduped = []
        for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
            if r["text"][:100] not in seen:
                seen.add(r["text"][:100])
                deduped.append(r)
        return deduped[:self.topk]

    def execute(self, query: str) -> str:
        """Agent 调用接口：返回格式化的检索结果"""
        results = self.search_hybrid(query)
        if not results:
            return "No relevant financial information found."

        output_parts = []
        for i, r in enumerate(results):
            output_parts.append(f"[{i+1}] {r['text']}")
        return "\n\n".join(output_parts)


# 全局单例（避免重复加载模型）
_search_tool = None

def get_search_tool():
    global _search_tool
    if _search_tool is None:
        _search_tool = FinancialSearchTool()
    return _search_tool

def execute_search(query: str) -> str:
    """外部调用接口"""
    return get_search_tool().execute(query)
```

## 4.2 Tool 2：金融计算器

```python
# tools/calculator_tool.py
"""
Tool 2: 金融计算器
支持常见财务公式和安全的数学表达式执行
"""

import re
import math

# ========== 预定义金融公式 ==========

FINANCIAL_FORMULAS = {
    "pe_ratio":      "Price / Earnings Per Share",
    "pb_ratio":      "Price / Book Value Per Share",
    "roe":           "Net Income / Shareholders' Equity",
    "roa":           "Net Income / Total Assets",
    "current_ratio": "Current Assets / Current Liabilities",
    "debt_to_equity":"Total Debt / Total Equity",
    "gross_margin":  "(Revenue - COGS) / Revenue",
    "net_margin":    "Net Income / Revenue",
    "cagr":          "((End Value / Start Value) ^ (1/Years)) - 1",
    "yoy_growth":    "(Current - Previous) / Previous * 100",
    "pct_change":    "(New - Old) / Old * 100",
}

def execute_calculate(expression: str) -> str:
    """
    执行金融计算

    支持两种模式：
    1. 直接数学表达式: "subtract(1829, 1731), divide(#0, 1731)"
    2. Python 风格表达式: "(1829 - 1731) / 1731"

    Args:
        expression: 计算表达式

    Returns:
        计算结果字符串
    """
    expression = expression.strip()

    # ===== 模式 1: FinQA 风格程序执行 =====
    # FinQA 的标注格式如: "subtract(1829, 1731), divide(#0, 1731)"
    if any(op in expression.lower() for op in ['add(', 'subtract(', 'multiply(',
                                                 'divide(', 'greater(', 'exp(']):
        try:
            return execute_finqa_program(expression)
        except Exception as e:
            pass  # 如果失败，尝试其他模式

    # ===== 模式 2: 百分比变化快捷计算 =====
    pct_match = re.search(
        r'(?:pct_change|yoy_growth|growth|change)[:\s]*'
        r'(\-?[\d,.]+)\s*[,/to→]\s*(\-?[\d,.]+)',
        expression, re.I
    )
    if pct_match:
        old_val = float(pct_match.group(1).replace(',', ''))
        new_val = float(pct_match.group(2).replace(',', ''))
        if old_val != 0:
            change = (new_val - old_val) / abs(old_val) * 100
            return f"Percentage change: ({new_val} - {old_val}) / |{old_val}| × 100 = {change:.2f}%"
        return "Error: Division by zero (old value is 0)"

    # ===== 模式 3: CAGR 计算 =====
    cagr_match = re.search(
        r'cagr[:\s]*(\-?[\d,.]+)\s*[,/]\s*(\-?[\d,.]+)\s*[,/]\s*(\d+)',
        expression, re.I
    )
    if cagr_match:
        start = float(cagr_match.group(1).replace(',', ''))
        end = float(cagr_match.group(2).replace(',', ''))
        years = int(cagr_match.group(3))
        if start > 0 and years > 0:
            cagr = ((end / start) ** (1 / years) - 1) * 100
            return f"CAGR: ({end}/{start})^(1/{years}) - 1 = {cagr:.2f}%"
        return "Error: Start value must be positive and years > 0"

    # ===== 模式 4: 通用安全数学计算 =====
    try:
        safe_expr = expression
        safe_expr = safe_expr.replace('^', '**')
        safe_expr = safe_expr.replace(',', '')  # 移除千位分隔符

        # 安全执行（限制可用函数）
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "exp": math.exp, "pow": pow, "pi": math.pi,
            "sum": sum, "len": len,
        }
        result = eval(safe_expr, {"__builtins__": {}}, allowed_names)

        # 格式化结果
        if isinstance(result, float):
            if abs(result) > 1e6:
                return f"Result: {expression} = {result:,.2f}"
            elif abs(result) < 0.01 and result != 0:
                return f"Result: {expression} = {result:.6f}"
            else:
                return f"Result: {expression} = {result:.4f}"
        return f"Result: {expression} = {result}"

    except Exception as e:
        return f"Calculation error: {str(e)}. Expression: {expression}"


def execute_finqa_program(program: str) -> str:
    """
    执行 FinQA 风格的推理程序
    例如: "subtract(1829, 1731), divide(#0, 1731)"
    """
    steps = [s.strip() for s in program.split(',')]
    results = []

    for step in steps:
        step = step.strip()
        if not step:
            continue

        # 解析操作和参数
        match = re.match(r'(\w+)\((.+)\)', step)
        if not match:
            continue

        op = match.group(1).lower()
        args_str = match.group(2)

        # 解析参数（支持 #0, #1 引用前面步骤的结果）
        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if arg.startswith('#'):
                idx = int(arg[1:])
                if idx < len(results):
                    args.append(results[idx])
                else:
                    return f"Error: Reference #{idx} not found"
            elif arg.lower() in ('const_100', 'const_1000'):
                args.append(float(arg.split('_')[1]))
            else:
                try:
                    args.append(float(arg.replace(',', '').replace('%', '')))
                except ValueError:
                    args.append(arg)

        # 执行操作
        try:
            if op == 'add':
                result = args[0] + args[1]
            elif op == 'subtract':
                result = args[0] - args[1]
            elif op == 'multiply':
                result = args[0] * args[1]
            elif op == 'divide':
                result = args[0] / args[1] if args[1] != 0 else float('inf')
            elif op == 'greater':
                result = 'yes' if args[0] > args[1] else 'no'
            elif op == 'exp':
                result = args[0] ** args[1]
            elif op == 'table_sum':
                result = sum(float(x) for x in args if isinstance(x, (int, float)))
            elif op == 'table_average':
                nums = [float(x) for x in args if isinstance(x, (int, float))]
                result = sum(nums) / len(nums) if nums else 0
            else:
                result = 0
                return f"Unknown operation: {op}"

            results.append(result)
        except Exception as e:
            return f"Error in step '{step}': {e}"

    if results:
        final = results[-1]
        if isinstance(final, float):
            # 格式化：如果是百分比类的结果
            if abs(final) < 10:
                return f"Result: {final:.4f} ({final*100:.2f}%)"
            else:
                return f"Result: {final:,.2f}"
        return f"Result: {final}"

    return "No result computed"
```

## 4.3 Tool 3：SQL 表格查询

```python
# tools/sql_tool.py
"""
Tool 3: SQL 财务表格查询
Agent 可以写 SQL 查询结构化的财务数据
"""

import sqlite3
import json
import re

DB_PATH = "data/tables/financial_data.db"

# 加载问题→表名映射
with open("data/tables/question_table_map.json", 'r', encoding='utf-8') as f:
    QUESTION_TABLE_MAP = json.load(f)

def get_table_schema(table_name: str) -> str:
    """获取表的 schema 信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        if not columns:
            return f"Table '{table_name}' not found."

        schema = f"Table: {table_name}\nColumns:\n"
        for col in columns:
            schema += f"  - {col[1]} ({col[2]})\n"

        # 显示前 3 行样例数据
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
        rows = cursor.fetchall()
        if rows:
            schema += f"\nSample data (first 3 rows):\n"
            for row in rows:
                schema += f"  {row}\n"

        return schema
    except Exception as e:
        return f"Error getting schema: {e}"
    finally:
        conn.close()


def execute_sql(query: str, question_id: str = None) -> str:
    """
    执行 SQL 查询

    Args:
        query: SQL 查询语句
        question_id: 可选，如果提供则自动定位对应的表

    Returns:
        查询结果字符串
    """
    query = query.strip()

    # 安全检查：禁止危险操作
    forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in forbidden:
        if keyword in query.upper():
            return f"Error: {keyword} operations are not allowed. Only SELECT queries are permitted."

    # 如果 Agent 的查询中没有指定表名，尝试自动推断
    if question_id and question_id in QUESTION_TABLE_MAP:
        suggested_table = QUESTION_TABLE_MAP[question_id]
        # 如果查询是 "SHOW TABLES" 类型，返回 schema
        if query.upper().startswith('SHOW') or query.upper().startswith('DESCRIBE'):
            return get_table_schema(suggested_table)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return "Query returned no results."

        # 获取列名
        col_names = [desc[0] for desc in cursor.description]

        # 格式化输出
        output = f"Columns: {', '.join(col_names)}\n"
        output += "-" * 60 + "\n"

        for row in rows[:20]:  # 最多显示 20 行
            formatted_row = " | ".join(str(v) for v in row)
            output += f"{formatted_row}\n"

        if len(rows) > 20:
            output += f"\n... ({len(rows)} total rows, showing first 20)"

        return output

    except Exception as e:
        return f"SQL Error: {str(e)}\nQuery: {query}"
    finally:
        conn.close()


def get_available_tables() -> str:
    """列出所有可用的表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT table_name, num_rows, num_cols, headers FROM table_index LIMIT 10")
    rows = cursor.fetchall()
    conn.close()

    output = "Available tables (first 10):\n"
    for row in rows:
        output += f"  {row[0]} ({row[1]} rows, {row[2]} cols) - {row[3]}\n"
    return output
```

## 4.4 统一工具调度器

```python
# tools/tool_dispatcher.py
"""
Agent 工具统一调度器
负责：
1. 从 Agent 输出中检测工具调用标签
2. 路由到对应工具执行
3. 格式化返回结果
4. 记录工具调用轨迹
"""

import re
from tools.search_tool import execute_search
from tools.calculator_tool import execute_calculate
from tools.sql_tool import execute_sql, get_table_schema

# ========== 工具注册表 ==========
TOOL_REGISTRY = {
    "search": {
        "tag": "search",
        "fn": execute_search,
        "description": "Search the financial report knowledge base",
    },
    "calculate": {
        "tag": "calculate",
        "fn": execute_calculate,
        "description": "Execute financial calculations",
    },
    "sql": {
        "tag": "sql",
        "fn": execute_sql,
        "description": "Query structured financial data tables",
    },
}


def detect_tool_call(text: str):
    """
    从 Agent 输出中检测工具调用

    Returns:
        (tool_name, query, remaining_text) 或 (None, None, None)
    """
    for name, tool in TOOL_REGISTRY.items():
        tag = tool["tag"]
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            query = match.group(1).strip()
            return name, query, text[match.end():]
    return None, None, None


def execute_tool(tool_name: str, query: str, question_id: str = None) -> str:
    """
    执行指定工具

    Args:
        tool_name: 工具名
        query: 工具输入
        question_id: 当前问题 ID（用于 SQL 工具定位表）

    Returns:
        工具执行结果字符串
    """
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{tool_name}'"

    fn = TOOL_REGISTRY[tool_name]["fn"]

    try:
        if tool_name == "sql":
            result = fn(query, question_id=question_id)
        else:
            result = fn(query)

        # 截断过长的结果（防止上下文爆炸）
        if len(result) > 1500:
            result = result[:1500] + "\n... [truncated]"

        return result

    except Exception as e:
        return f"Tool execution error ({tool_name}): {str(e)}"


def format_observation(result: str) -> str:
    """将工具结果格式化为 observation 标签"""
    return f"\n<observation>\n{result}\n</observation>\n"


def multi_turn_agent_rollout(
    generate_fn,  # 生成函数: (prompt) -> text
    prompt: str,
    question_id: str = None,
    max_turns: int = 5,
    max_total_tokens: int = 4096,
):
    """
    多轮 Agent Rollout 控制器

    核心逻辑：
    1. Agent 生成文本
    2. 检测是否有工具调用
    3. 如果有 → 执行工具 → 拼接 observation → 继续生成
    4. 如果没有或已有 <answer> → 结束

    Args:
        generate_fn: 模型生成函数
        prompt: 初始 prompt
        question_id: 当前问题 ID
        max_turns: 最大交互轮数
        max_total_tokens: 最大总 token 数

    Returns:
        dict with keys:
            full_text: 完整生成文本
            tool_trace: 工具调用轨迹列表
            token_mask: token 级别的 mask（True = 外部 token，不参与梯度）
    """
    full_text = ""
    tool_trace = []
    token_mask_segments = []  # (text, is_external) 列表
    current_input = prompt

    for turn in range(max_turns):
        # === Step 1: Agent 生成 ===
        new_text = generate_fn(current_input)
        full_text += new_text
        token_mask_segments.append((new_text, False))  # Agent 自己生成的

        # === Step 2: 检测工具调用 ===
        tool_name, query, _ = detect_tool_call(new_text)

        if tool_name:
            # === Step 3: 执行工具 ===
            result = execute_tool(tool_name, query, question_id=question_id)
            observation = format_observation(result)

            full_text += observation
            token_mask_segments.append((observation, True))  # 外部 token → mask

            tool_trace.append({
                "turn": turn + 1,
                "tool": tool_name,
                "query": query,
                "result": result[:300],  # 只记录前 300 字符用于展示
            })

            # 继续生成
            current_input = prompt + full_text

        else:
            # 没有工具调用 → 检查是否有答案
            if "<answer>" in new_text:
                break
            # 也没有答案 → 可能是纯推理，继续
            # 但如果已经很长了就停止
            if len(full_text) > max_total_tokens * 4:  # 粗略估计
                break

    return {
        "full_text": full_text,
        "tool_trace": tool_trace,
        "token_mask_segments": token_mask_segments,
    }


def extract_answer(text: str) -> str:
    """从 Agent 输出中提取最终答案"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 回退：取最后一行
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""
```

## 4.5 检索 FastAPI 服务（独立进程运行）

```python
# tools/retrieval_server.py
"""
检索服务：作为独立进程运行
供训练时的 rollout 和 Demo 调用
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from tools.search_tool import get_search_tool, search_service_payload

app = FastAPI(title="FinAgent-R1 Retrieval Service", version="1.0")

class BatchQueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = True
    method: str = "hybrid"

class SingleQueryRequest(BaseModel):
    query: str
    topk: int | None = None
    method: str = "hybrid"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/retrieve")
def retrieve(request: BatchQueryRequest):
    return search_service_payload(
        queries=request.queries,
        method=request.method,
        topk=request.topk,
        return_scores=request.return_scores,
    )

@app.post("/search")
def search(request: SingleQueryRequest):
    result = get_search_tool().batch_search(
        queries=[request.query],
        method=request.method,
        topk=request.topk,
        return_scores=True,
    )[0]
    return {
        "results": result,
        "query": request.query,
        "method": request.method,
        "topk": request.topk or get_search_tool().topk,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

## 4.6 测试所有工具

```python
# scripts/test_tools.py
"""
集成测试：验证所有三个工具都能正常工作
"""

print("=" * 60)
print("Tool 1: 金融知识检索")
print("=" * 60)
from tools.search_tool import execute_search
result = execute_search("What was the revenue growth in 2019?")
print(f"Query: 'What was the revenue growth in 2019?'")
print(f"Result (前 200 字):\n{result[:200]}\n")

print("=" * 60)
print("Tool 2: 金融计算器")
print("=" * 60)
from tools.calculator_tool import execute_calculate

# 测试百分比变化
r1 = execute_calculate("pct_change: 1829, 1731")
print(f"pct_change(1829, 1731): {r1}")

# 测试 FinQA 程序
r2 = execute_calculate("subtract(1829, 1731), divide(#0, 1731)")
print(f"subtract(1829,1731), divide(#0,1731): {r2}")

# 测试 CAGR
r3 = execute_calculate("cagr: 100, 150, 3")
print(f"cagr(100, 150, 3 years): {r3}")

# 测试通用表达式
r4 = execute_calculate("(1829 - 1731) / 1731 * 100")
print(f"(1829-1731)/1731*100: {r4}")

print("\n" + "=" * 60)
print("Tool 3: SQL 查询")
print("=" * 60)
from tools.sql_tool import execute_sql, get_available_tables

tables = get_available_tables()
print(f"可用表（前 10 张）:\n{tables}")

# 尝试查询第一张表
import sqlite3
conn = sqlite3.connect("data/tables/financial_data.db")
cursor = conn.cursor()
cursor.execute("SELECT table_name FROM table_index LIMIT 1")
first_table = cursor.fetchone()
conn.close()

if first_table:
    table_name = first_table[0]
    r = execute_sql(f'SELECT * FROM "{table_name}" LIMIT 3')
    print(f"\n查询 {table_name} 的前 3 行:\n{r}")

print("\n" + "=" * 60)
print("Tool Dispatcher 测试")
print("=" * 60)
from tools.tool_dispatcher import detect_tool_call

test_cases = [
    'Let me search for this. <search>revenue growth 2019</search>',
    'I need to calculate: <calculate>(1829 - 1731) / 1731</calculate>',
    '<sql>SELECT * FROM t_abc LIMIT 5</sql>',
    'Just thinking, no tool call here.',
]

for text in test_cases:
    tool, query, _ = detect_tool_call(text)
    print(f"Input: '{text[:60]}...'")
    print(f"  → Tool: {tool}, Query: {query}\n")

print("=== 所有工具测试完成 ===")
```

```bash
python scripts/test_tools.py
```

---

# 第五章：当前训练主线说明（veRL）

> 2026-03 仓库现状更新：
> 1. 正式训练主线已经切到 Search-R1 风格的 veRL 多轮 Agent RL。
> 2. SFT 可选分支已完整实现：`scripts/generate_sft_data.py`、`training/sft_coldstart.py`、
>    `scripts/test_sft_model.py` 均已存在于仓库中，可作为冷启动可选流程使用。
> 3. `configs/sft_config.yaml` 配合上述脚本可构成完整的 SFT pipeline。

## 5.1 为什么主线改为 veRL

当前仓库真正落地的不是“单轮 GRPOTrainer 示例”，而是多轮 Agent-Environment 交互：

- 训练入口：`training/finagent_verl_main.py`
- Search-R1 兼容层：`training/search_r1_compat.py`
- 多轮 rollout / action 执行：`training/finagent_generation.py`
- 启动脚本：`scripts/train_grpo_verl.sh`
- 训练配置：`configs/verl_ppo_finqa.yaml`

这样做的原因是：金融问答里的搜索、SQL、计算通常不是“一步到位”的单轮生成，
而是需要多轮 `<tool> -> <observation> -> <think>` 交互。

## 5.2 当前可执行的数据准备流程

当前仓库用于训练的数据不是 `sft_seed.jsonl`，而是 veRL / Search-R1 风格的 parquet：

```bash
# 1. FinQA 原始数据转 jsonl
python scripts/prepare_training_data.py

# 2. 构建 question_id -> table_name 映射
python scripts/build_question_table_map.py

# 3. 生成 veRL 训练 parquet
python scripts/prepare_verl_finqa_data.py
```

输出目录：

```text
data/verl/finqa/train.parquet
data/verl/finqa/validation.parquet
data/verl/finqa/test.parquet
```

这些 parquet 里已经包含：

- chat prompt（system + user）
- `reward_model.ground_truth.target`
- `reward_model.ground_truth.program`
- `reward_model.ground_truth.question_id`
- `table_name` 和 `extra_info`

因此当前主线不再依赖额外的 SFT seed 脚本。

## 5.3 SFT 分支当前状态

SFT 冷启动作为可选分支已完整实现，相关文件均已存在于仓库中：

- `configs/sft_config.yaml` — SFT 训练配置
- `scripts/generate_sft_data.py` — 生成带工具标签的 SFT 种子数据
- `training/sft_coldstart.py` — 执行 SFT 冷启动训练
- `scripts/test_sft_model.py` — 快速验证 SFT 模型输出

推荐执行顺序：

```bash
python scripts/generate_sft_data.py
python training/sft_coldstart.py
python scripts/test_sft_model.py
```

注意：SFT 不是当前主线训练步骤，但适合作为冷启动或对照实验使用。

---

# 第六章：veRL 多轮 Agent RL 训练（当前主线）

## 6.1 训练入口与运行时

当前仓库的正式训练入口是：

- `training/finagent_verl_main.py` — 主训练脚本
- `scripts/train_grpo_verl.sh` — 启动脚本（设置环境变量后调用上面的 Python 入口）
- `configs/verl_ppo_finqa.yaml` — 训练超参配置

### 核心调用链

```
train_grpo_verl.sh
  → training/finagent_verl_main.py
      → training/search_r1_compat.py     # 把 vendor/Search-R1 加入 sys.path
      → verl.trainer.ppo.ray_trainer      # veRL 原版 PPO/GRPO trainer
          → training/finagent_generation.py  # FinAgent 多轮 generation manager（通过 import 替换注入）
      → training/reward_functions.py      # rule-based reward
```

### Generation Manager 注入机制

FinAgent 需要用自定义的 `LLMGenerationManager` 替换 Search-R1 原版的单工具版本。
注入方式是直接修改 `vendor/Search-R1/verl/trainer/ppo/ray_trainer.py` 的 import 行：

```python
# vendor/Search-R1/verl/trainer/ppo/ray_trainer.py 第 43 行
try:
    from training.finagent_generation import LLMGenerationManager, GenerationConfig
except ImportError:
    from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
```

这样做的原因是：Python 的 `from X import Y` 会把 `Y` 绑定为函数内局部变量，
之后通过 `module.Y = new_Y` 做 monkey-patch 无法影响已经绑定的局部引用。
因此必须在 import 阶段就替换为 FinAgent 版本。

### Ray Worker 进程 PYTHONPATH

`finagent_verl_main.py` 在 `ray.init()` 中通过 `runtime_env.env_vars.PYTHONPATH`
将项目根目录和 `vendor/Search-R1` 暴露给 Ray worker 进程，
确保 worker 能正确 import `training.*` 和 `tools.*` 模块：

```python
ray.init(runtime_env={
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "PYTHONPATH": f"{ROOT_DIR}:{ROOT_DIR / 'vendor' / 'Search-R1'}",
    },
})
```

### 启动方式

```bash
# 终端 1：启动检索服务（veRL 训练时 generation manager 会通过 HTTP 调用检索）
python tools/retrieval_server.py

# 终端 2：启动 veRL 训练
bash scripts/train_grpo_verl.sh
```

默认输出目录由 `configs/verl_ppo_finqa.yaml` 控制，当前是：

```text
checkpoints/grpo_verl
```

## 6.2 当前奖励函数接口

当前仓库不是把奖励拆成三个对外暴露的函数接口，
而是统一走 `compute_finagent_score()`：

- 主奖励：答案是否正确
- 行为 bonus：结构化标签、工具使用、多工具使用
- 惩罚项：工具过度调用、无工具回答、无效 retry

也就是说，当前实际接口是“一个综合 reward 函数 + 一组可配置权重”，
而不是旧版本文档里的：

- `accuracy_reward()`
- `agent_behavior_reward()`
- `tool_efficiency_reward()`

如果你后续要恢复 TRL / Unsloth 版训练脚本，需要额外写一层 adapter。

## 6.3 多轮 rollout 机制

多轮 Agent 训练的核心在 `training/finagent_generation.py` 中的 `LLMGenerationManager`。

### 单轮循环

```
模型 generate → 截断到第一个完整动作 → 解析动作类型 → 执行工具 → 包装 <observation> → 拼回 context
```

### 关键方法

| 方法 | 作用 |
|------|------|
| `run_llm_loop()` | 主循环入口，最多执行 `max_turns` 轮 |
| `_postprocess_responses()` | 截断到第一个完整的 `</search>` / `</calculate>` / `</sql>` / `</answer>` |
| `_parse_action()` | 用正则提取动作类型和内容 |
| `execute_predictions()` | 批量执行动作：search 走 HTTP/本地检索，calculate 走 Python 沙箱，sql 走 SQLite |
| `_update_rolling_state()` | 把工具结果拼回 rolling context，供下一轮 generate 使用 |
| `_generate_with_gpu_padding()` | 多 GPU 时自动 pad batch 到可整除大小 |

### 支持的四类动作

| 动作标签 | 处理方式 | 结束 |
|---------|---------|------|
| `<search>query</search>` | HTTP 调用 `retrieval_server.py` 或本地 `get_search_tool()` | 否 |
| `<calculate>expr</calculate>` | 调用 `execute_calculate(content)` | 否 |
| `<sql>query</sql>` | 调用 `execute_sql(content, question_id=qid)`，自动路由到对应表 | 否 |
| `<answer>result</answer>` | 标记为 done，不再继续生成 | 是 |

### question_id 路由

`finagent_generation.py` 会从 `DataProto.non_tensor_batch` 中提取每个样本的 `question_id`，
在调用 SQL 工具时传入 `execute_sql(content, question_id=qid)`，
由 `sql_tool.py` 内部通过 `question_table_map.json` 自动映射到对应的 SQLite 表。

## 6.4 veRL 配置要点

当前主配置文件是 `configs/verl_ppo_finqa.yaml`，其中最关键的字段包括：

- `data.train_files = data/verl/finqa/train.parquet`
- `data.val_files = data/verl/finqa/validation.parquet`
- `trainer.default_local_dir = checkpoints/grpo_verl`
- `trainer.project_name = finagent_r1`
- `trainer.experiment_name = finqa_3tool_grpo_verl`
- `retriever.url = http://127.0.0.1:5000/retrieve`
- `max_turns = 4`
- `actor_rollout_ref.rollout.n_agent = 4`

常见覆写方式：

```bash
bash scripts/train_grpo_verl.sh \
  trainer.total_training_steps=100 \
  trainer.save_freq=50 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B
```

## 6.5 TRL / Unsloth 兼容训练路线

以下兼容训练脚本已存在于仓库中，可作为轻量替代或对照实验使用：

- `training/grpo_train.py` — 纯 TRL GRPOTrainer 训练入口
- `training/grpo_train_unsloth.py` — Unsloth + TRL GRPO 训练入口
- `training/void_turn_filter.py` — 空轮次检测与过滤工具

这些脚本不是当前主训练路径（主线为 veRL），但适合在单卡上做快速验证、小规模实验或消融对照。

如果你的目标是和当前代码保持一致，请统一使用 veRL 主线。

## 6.6 veRL 训练常见问题与排查

### 检索服务未启动

训练时如果 `retrieval_server.py` 没有运行，generation manager 会在 `batch_search()` 时报
HTTP 连接错误。确保在训练前启动：

```bash
python tools/retrieval_server.py
```

默认监听 `http://127.0.0.1:5000`，对应 `configs/verl_ppo_finqa.yaml` 中的 `retriever.url`。

### Ray worker import 失败

如果 Ray worker 报 `ModuleNotFoundError: No module named 'training'`，
说明 `PYTHONPATH` 没有正确传递。检查：

1. `finagent_verl_main.py` 的 `ray.init()` 是否设置了 `PYTHONPATH` env var
2. 是否从项目根目录启动训练（`bash scripts/train_grpo_verl.sh`）

### 显存不足

单卡 24GB 的推荐配置：

```yaml
# 开启 CPU offload 节省显存
actor_rollout_ref.actor.fsdp_config.param_offload: true
actor_rollout_ref.actor.fsdp_config.grad_offload: true
actor_rollout_ref.actor.fsdp_config.optimizer_offload: true

# 降低 batch size
data.train_batch_size: 32
actor_rollout_ref.actor.ppo_micro_batch_size: 4

# 降低 vLLM 显存占用
actor_rollout_ref.rollout.gpu_memory_utilization: 0.4
```

### Reward 全为 0

可能原因：
- `data/verl/finqa/train.parquet` 中的 `reward_model.ground_truth.target` 字段为空
- 模型生成中没有 `<answer>...</answer>` 标签（检查 `max_turns` 是否太小）
- 数值容差（`finagent_reward.tolerance`）设置过严

调试方法：在 `RewardManager` 的 `num_examine` 参数设为大于 0 来打印样本：

```bash
bash scripts/train_grpo_verl.sh \
  trainer.total_training_steps=5
```

观察打印的 `[reward]` 值和对应的生成文本。

---

# 第七章：评测与消融实验

> 当前仓库已实现基础离线评测，但不是早期设计版的完整指标集合。

## 7.1 主评测脚本

当前实际脚本是 `eval/evaluate.py`，已经实现：

- Execution Accuracy
- Tool Usage Rate
- Multi-Tool Rate
- 各工具类型调用统计
- 结果保存到 `eval/results_*.json`

当前未实现：

- Program Accuracy (PA)

建议运行方式：

```bash
# smoke test：直接评测基础模型
python eval/evaluate.py Qwen/Qwen2.5-3B --max-samples 50

# 如果你已经导出了可加载的训练后模型，也可以替换成自己的路径
python eval/evaluate.py <your_model_path> --max-samples 50
```

注意：当前 `eval/evaluate.py` 直接按 `AutoModelForCausalLM.from_pretrained(model_path)` 加载模型，
不是旧文档里“base model + PEFT adapter”那条加载方式。

## 7.2 消融实验脚本

当前 `eval/ablation.py` 是一个“简化版 experiment runner”，默认实验列表只有：

- Base model
- SFT checkpoint（如果存在）
- GRPO / veRL checkpoint

运行方式：

```bash
python eval/ablation.py
```

需要注意：

- 当前仓库没有完整的 Vanilla RAG baseline 脚本
- 当前仓库也没有 search-only GRPO 的单独实验分支
- 如果你走的是 veRL 主线，建议先把 `EXPERIMENTS` 中的 checkpoint 路径改成你实际导出的模型路径

---

# 第八章：当前 Demo 与部署状态

## 8.1 FastAPI 后端

当前 `demo/backend/main.py` 已实现：

1. `/api/health`
2. `/api/ask`

当前未实现：

1. `/api/ask_stream`

后端通过 `MODEL_PATH` 环境变量加载模型，并调用
`tools/tool_dispatcher.multi_turn_agent_rollout()` 执行多轮工具交互。

## 8.2 启动 Demo

当前仓库推荐这样启动：

```bash
# 启动检索服务 + 后端
bash scripts/start_services.sh

# 另一个终端启动 Gradio
python demo/gradio_app.py
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

问答测试：

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the percentage change in revenue from 2018 to 2019?"}'
```

如果你已经导出了可加载的训练后模型，可以这样指定：

```bash
MODEL_PATH=<your_model_path> python demo/backend/main.py
```

## 8.3 前端状态

当前可运行的前端有两种方式：

1. **Gradio**（轻量）：`python demo/gradio_app.py`
2. **React**（完整）：`cd demo/frontend && npm install && npm run dev`

`demo/frontend/src/` 目录下已包含完整的 React 实现，包括：
- `App.jsx` — 主应用入口
- `api.js` — 后端 API 调用封装
- `components/ChatInterface.jsx` — 聊天界面
- `components/ReasoningPanel.jsx` — 推理过程展示面板
- `components/ToolTrace.jsx` — 工具调用链路可视化

## 8.4 Docker 部署

仓库已包含完整的 Docker 部署文件：

- `docker-compose.yml` — 编排所有服务
- `Dockerfile.backend` — FastAPI 后端镜像
- `Dockerfile.retrieval` — 检索服务镜像
- `Dockerfile.demo` — Gradio Demo 镜像

启动方式：

```bash
docker compose up --build
```

---

# 第九章：开源仓库整理与发布

## 9.1 README.md 模板

```markdown
# FinAgent-R1 🏦🔍🧮

A multi-tool financial analysis agent trained with reinforcement learning,
based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) (COLM 2025).

The agent autonomously decides **when** and **which** tool to use:
- 🔍 `<search>` — Retrieve relevant passages from financial reports
- 🧮 `<calculate>` — Execute financial calculations (PE, ROE, CAGR, etc.)
- 📊 `<sql>` — Query structured financial data tables

## Highlights
- 📈 **XX% EM improvement** over vanilla RAG on FinQA test set
- 🔧 **Multi-tool agent** — 3 tools, autonomous selection via RL
- 💻 **Single GPU training** — Qwen2.5-3B + veRL/GRPO on RTX 3090/4090
- 🚀 **Demo stack** — FastAPI + Gradio

## Quick Start
... (安装、训练、部署步骤)

## Results
| Method | EM (%) | Tool Usage | Multi-Tool |
|--------|--------|------------|------------|
| Base model (no tools) | XX.X | 0% | 0% |
| Vanilla RAG | XX.X | 100% | 0% |
| FinAgent-R1 (search only) | XX.X | XX% | 0% |
| **FinAgent-R1 (3 tools)** | **XX.X** | **XX%** | **XX%** |

## Acknowledgments
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) — Core RL framework
- [AutoTIR](https://github.com/weiyifan1023/AutoTIR) — Multi-tool RL inspiration
- [FinQA](https://github.com/czyssrs/FinQA) — Financial QA dataset
- [veRL](https://github.com/volcengine/verl) — RL training runtime
```

---

# 第十章：简历撰写与面试准备

## 10.1 简历项目描述（中文）

```
FinAgent-R1：基于强化学习的金融多工具 Agent 系统        2025.03 – 至今
个人项目 · 基于 Search-R1（COLM 2025，GitHub 3.8k★）& AutoTIR 框架

• 基于 Search-R1 的 Agentic RL 范式，构建了一个金融领域多工具 Agent：
  LLM 作为 Agent 核心，通过 GRPO 强化学习自主学习何时调用工具、
  调用哪个工具、如何利用工具返回结果进行推理，无需预设工具调用模板。

• 实现了三种 Agent Action：① 搜索金融知识库（BM25+BGE 双通道检索），
  ② 执行金融计算（PE/ROE/CAGR 等），③ SQL 查询财务表格，
  Agent 在推理过程中动态决策调用策略。

• 使用 Qwen2.5-3B + veRL/GRPO 在单卡 RTX 3090 上完成多轮 Agent RL 训练，
  实现了基于 observation 的 state masking 与多轮工具交互 rollout。

• 设计了复合奖励函数（答案准确率 + 行为合规性 + 工具效率），
  在 FinQA 测试集上较 Vanilla RAG 提升约 XX% Execution Accuracy。

• 部署为可演示的 Agent 应用（FastAPI + Gradio），
  支持展示 Agent 工具调用轨迹；React 与 Docker 仍可继续补全。
```

## 10.2 面试必备 Q&A

### Q1: "Search-R1 和你的项目的关系是什么？"

Search-R1 是一个开源的 RL 训练框架，发表在 COLM 2025，
原版只支持搜索引擎一个工具。我在此基础上做了三件事：
1. 迁移到金融领域（换数据集、换检索语料）
2. 扩展为多工具 Agent（加了计算器和 SQL 查询）
3. 适配成 veRL / Search-R1 风格的多轮 rollout 与 reward pipeline
4. 做了可运行的 FastAPI + Gradio Demo

### Q2: "为什么选 FinQA 数据集？"

FinQA 有三个独特优势：
1. 每个问题都有标注的推理程序（program），可以精确计算 reward
2. 同时包含文本和表格，天然需要多工具协同
3. 是金融 QA 领域最广泛使用的 benchmark，结果有可比性

### Q3: "GRPO 和 PPO 的区别？为什么选 GRPO？"

PPO 需要一个额外的 value model（评估每个状态的好坏），
这意味着显存需要额外放一个和 policy 一样大的模型。

GRPO 去掉了 value model，改为：
- 同一个 prompt 生成 N 个候选答案
- 用组内相对排名来估计 advantage
- 加 KL 正则防止偏离太远

好处：显存省一半，单卡可跑。Search-R1 论文实验表明两者效果相近。

### Q4: "Retrieved Token Masking 怎么实现的？"

在多轮 rollout 中，我维护一个 token-level 的 mask 列表。
当工具返回结果被拼接到上下文时，
对应位置的 mask 设为 True。
在计算 policy gradient 时：
```
loss = -log_prob * advantage * (1 - mask)
```
mask=True 的位置 loss 直接为 0，不参与梯度更新。

### Q5: "面试时能现场演示吗？"

可以。启动检索服务、后端和 Gradio Demo，输入一个问题比如
"What was the CAGR of revenue from 2017 to 2019?"，
实时看到 Agent 先搜索研报找到数据，
再调用计算器算 CAGR，最后给出答案。

---

## 检查清单

```
□ Day 1:   conda 环境 + 全部依赖安装
□ Day 2:   下载 FinQA + 预处理 + 构建语料库 + 构建 SQL 数据库
□ Day 3:   BM25 索引 + FAISS 索引 + 三个工具实现 + 集成测试
□ Day 4:   训练数据转换为 veRL parquet + 配置 question-table map
□ Day 5:   编写奖励函数 + 配置 veRL / Search-R1 rollout
□ Day 6-7: veRL 多轮 Agent RL 训练（可能需要多次调参）
□ Day 8:   跑基础评测与已有消融
□ Day 9:   整理实验结果 + 制作表格
□ Day 10:  FastAPI 后端开发
□ Day 11:  Gradio/React 前端开发
□ Day 12:  可选补 Docker / 流式接口 / React 前端
□ Day 13:  README + 文档 + .gitignore
□ Day 14:  GitHub 发布 + 简历更新
```

---

> **最后提醒：** 简历上所有 XX% 的占位符，必须替换为你跑完实验后的真实数字。
> 真实数据即使低于预估也没关系，诚实比夸大重要得多。
> 面试官问的不是"你的数字有多高"，而是"你理不理解为什么是这个数字"。
