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
- [第五章：SFT 冷启动训练](#第五章sft-冷启动训练)
- [第六章：GRPO Agent RL 训练（核心）](#第六章grpo-agent-rl-训练核心)
- [第七章：评测与消融实验](#第七章评测与消融实验)
- [第八章：全栈 Agent Demo 开发](#第八章全栈-agent-demo-开发)
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
Day 4-5    ：SFT 冷启动训练
Day 6-8    ：GRPO Agent RL 训练（核心，可能需要多次调参）
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

# 验证安装
python -c "from transformers import AutoModelForCausalLM; print('transformers OK')"
python -c "from trl import GRPOTrainer; print('trl OK')"
python -c "from peft import LoraConfig; print('peft OK')"
```

## 2.5 安装 Unsloth（推荐，显存效率最高）

```bash
# Unsloth 单独安装（和上面的 trl 兼容）
pip install unsloth

# 验证
python -c "from unsloth import FastLanguageModel; print('unsloth OK')"
```

如果 Unsloth 安装报错，可以跳过，后面用纯 TRL 方案。

## 2.6 安装检索相关依赖

```bash
# BM25 检索
pip install pyserini

# 需要 Java（Pyserini 依赖 Lucene）
# Ubuntu:
sudo apt-get update && sudo apt-get install -y default-jdk
java -version  # 确认 Java 11+

# 向量检索
pip install faiss-gpu          # GPU 版 FAISS
pip install sentence-transformers  # BGE 编码模型

# 验证
python -c "import faiss; print('faiss OK, GPU:', faiss.get_num_gpus())"
python -c "from pyserini.search.lucene import LuceneSearcher; print('pyserini OK')"
```

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
# 后端
pip install fastapi==0.115.0
pip install uvicorn[standard]
pip install sse-starlette      # Server-Sent Events
pip install flask              # 检索服务用

# 前端（需要 Node.js）
# Ubuntu:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version  # 确认 v20+
npm --version
```

## 2.9 克隆 Search-R1（参考用）

```bash
mkdir -p ~/projects
cd ~/projects

# 克隆原版 Search-R1 作为参考
git clone https://github.com/PeterGriffinJin/Search-R1.git
# 你主要参考的文件：
#   search_r1/search/retrieval_server.py  - 检索服务实现
#   verl 训练脚本                         - RL 训练流程

# 创建你自己的项目
mkdir FinAgent-R1
cd FinAgent-R1
git init
```

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
│   └── retrieval_server.py     # 检索 Flask 服务
├── training/
│   ├── __init__.py
│   ├── reward_functions.py     # 奖励函数
│   ├── sft_coldstart.py        # SFT 冷启动
│   ├── grpo_train.py           # GRPO 主训练（TRL 方案）
│   ├── grpo_train_unsloth.py   # GRPO 主训练（Unsloth 方案）
│   ├── void_turn_filter.py     # Void Turn 过滤
│   └── multi_turn_rollout.py   # 多轮 Agent Rollout
├── eval/
│   ├── __init__.py
│   ├── evaluate.py             # 主评测脚本
│   └── ablation.py             # 消融实验
├── demo/
│   ├── backend/
│   │   ├── main.py             # FastAPI 后端
│   │   ├── inference.py        # 推理引擎
│   │   └── requirements.txt
│   └── frontend/
│       ├── package.json
│       └── src/
│           ├── App.jsx
│           ├── components/
│           │   ├── ChatInterface.jsx
│           │   ├── ToolTrace.jsx
│           │   └── ReasoningPanel.jsx
│           └── api.js
├── configs/
│   ├── sft_config.yaml
│   └── grpo_config.yaml
├── scripts/
│   ├── download_data.sh        # 数据下载脚本
│   ├── build_indexes.sh        # 构建索引脚本
│   └── start_services.sh       # 启动所有服务
├── docker-compose.yml
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

## 4.5 检索 Flask 服务（独立进程运行）

```python
# tools/retrieval_server.py
"""
检索服务：作为独立进程运行
供训练时的 rollout 和 Demo 调用
"""

from flask import Flask, request, jsonify
from tools.search_tool import FinancialSearchTool
import time

app = Flask(__name__)
search_tool = None

@app.before_request
def init_search():
    global search_tool
    if search_tool is None:
        print("初始化检索工具...")
        search_tool = FinancialSearchTool()
        print("检索工具初始化完成")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    method = data.get("method", "hybrid")  # bm25 / dense / hybrid

    start = time.time()

    if method == "bm25":
        results = search_tool.search_bm25(query)
    elif method == "dense":
        results = search_tool.search_dense(query)
    else:
        results = search_tool.search_hybrid(query)

    elapsed = time.time() - start

    return jsonify({
        "results": results,
        "query": query,
        "method": method,
        "latency_ms": round(elapsed * 1000, 1),
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("=== 金融知识检索服务 ===")
    print("启动地址: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
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

# 第五章：SFT 冷启动训练

## 5.1 为什么需要 SFT 冷启动

直接从 base model 开始 GRPO 训练，模型不知道要使用 `<search>`, `<calculate>`, `<sql>` 这些标签。
SFT 冷启动的目的是让模型学会"工具调用的格式"，只需要 100-300 条数据，训练 1-3 epoch。

## 5.2 生成 SFT 种子数据

```python
# scripts/generate_sft_data.py
"""
为 SFT 冷启动生成种子数据
两种方式：
  方式 A: 基于规则自动生成（简单快速）
  方式 B: 使用 Fino1 数据集（有 GPT-4o 推理路径）
"""

import json
import random
from datasets import load_from_disk

# ===== 方式 A: 基于规则自动构造 =====

def generate_sft_example_rule_based(example):
    """
    根据 FinQA 的标注 program 自动构造工具调用轨迹
    """
    question = example['question']
    answer = str(example['answer'])
    program = example['program']

    # 判断需要哪些工具
    needs_search = True  # 基本都需要检索上下文
    needs_calculate = any(op in program for op in
                         ['add(', 'subtract(', 'multiply(', 'divide(', 'exp('])
    needs_sql = 'table_' in program  # table_sum, table_average 等需要 SQL

    # 构造搜索查询（从问题中提取关键词）
    search_query = question[:80]

    # 构造计算表达式
    calc_expr = program if needs_calculate else None

    # 构造回答
    parts = []
    parts.append(f"<think>Let me analyze this financial question. I need to find relevant data from the financial report.</think>")

    if needs_search:
        parts.append(f"<search>{search_query}</search>")
        parts.append(f"<observation>[Financial report excerpt with relevant data]</observation>")
        parts.append(f"<think>I found the relevant data. Now let me process it.</think>")

    if needs_sql:
        parts.append(f"<sql>SELECT * FROM financial_table WHERE relevant_column IS NOT NULL</sql>")
        parts.append(f"<observation>[Table data with relevant figures]</observation>")
        parts.append(f"<think>I have the data from the table. Let me calculate the answer.</think>")

    if needs_calculate:
        parts.append(f"<calculate>{program}</calculate>")
        parts.append(f"<observation>Result: {answer}</observation>")
        parts.append(f"<think>The calculation gives us the answer.</think>")

    parts.append(f"<answer>{answer}</answer>")

    response = "\n".join(parts)

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a financial analysis agent with access to search, calculate, and SQL tools."
            },
            {
                "role": "user",
                "content": f"Answer this financial question:\n{question}"
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
    }


# 加载数据
ds = load_from_disk("data/raw/finqa_hf")

sft_data = []
indices = list(range(len(ds['train'])))
random.shuffle(indices)

for idx in indices[:300]:  # 取 300 条
    example = ds['train'][idx]
    sft_example = generate_sft_example_rule_based(example)
    sft_data.append(sft_example)

# 保存
with open("data/processed/sft_seed.jsonl", 'w', encoding='utf-8') as f:
    for item in sft_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"SFT 种子数据生成完成: {len(sft_data)} 条")
print(f"保存至: data/processed/sft_seed.jsonl")

# 检查一个样例
print("\n--- 样例 ---")
print(json.dumps(sft_data[0], indent=2, ensure_ascii=False)[:1000])
```

```bash
python scripts/generate_sft_data.py
```

## 5.3 SFT 训练脚本

```python
# training/sft_coldstart.py
"""
SFT 冷启动训练
让模型学会使用 <think>, <search>, <calculate>, <sql>, <answer> 标签

预计训练时间：单卡 3090，约 30-60 分钟
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# ===== 1. 加载模型 =====
MODEL_NAME = "Qwen/Qwen2.5-3B"

print(f"加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# ===== 2. 配置 LoRA =====
lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    target_modules=[         # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== 3. 加载数据 =====
dataset = load_dataset("json", data_files="data/processed/sft_seed.jsonl", split="train")
print(f"SFT 数据集大小: {len(dataset)}")

# ===== 4. 训练配置 =====
training_args = TrainingArguments(
    output_dir="checkpoints/sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="finagent-sft-coldstart",
    max_grad_norm=1.0,
)

# ===== 5. 训练 =====
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
)

print("开始 SFT 冷启动训练...")
trainer.train()

# ===== 6. 保存 =====
trainer.save_model("checkpoints/sft/final")
tokenizer.save_pretrained("checkpoints/sft/final")
print("SFT 冷启动训练完成！模型保存至 checkpoints/sft/final")
```

```bash
# 启动训练
python training/sft_coldstart.py
```

## 5.4 验证 SFT 效果

```python
# scripts/test_sft_model.py
"""
加载 SFT 模型，测试是否学会了工具调用格式
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载基础模型 + LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, "checkpoints/sft/final")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/sft/final")

# 测试
test_questions = [
    "What was the percentage change in total revenue from 2018 to 2019?",
    "Calculate the gross margin ratio given revenue of $5.2 billion and COGS of $3.1 billion.",
    "What is the debt-to-equity ratio based on the financial statements?",
]

for q in test_questions:
    prompt = f"""You are a financial analysis agent with access to search, calculate, and SQL tools.

Answer this financial question:
{q}"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {q}")
    print(f"A: {response[:500]}")
    print("-" * 60)

    # 检查是否使用了工具标签
    has_think = "<think>" in response
    has_search = "<search>" in response
    has_calc = "<calculate>" in response
    has_sql = "<sql>" in response
    has_answer = "<answer>" in response
    print(f"  ✓ <think>: {has_think}")
    print(f"  ✓ <search>: {has_search}")
    print(f"  ✓ <calculate>: {has_calc}")
    print(f"  ✓ <sql>: {has_sql}")
    print(f"  ✓ <answer>: {has_answer}")
```

```bash
python scripts/test_sft_model.py
```

如果模型能生成包含 `<think>`, `<search>`, `<calculate>`, `<answer>` 标签的回答，
说明 SFT 冷启动成功。如果标签使用率低于 50%，增加 SFT 数据到 500 条或多训练几个 epoch。

---

# 第六章：GRPO Agent RL 训练（核心）

## 6.1 奖励函数

```python
# training/reward_functions.py
"""
FinAgent-R1 复合奖励函数

奖励由三个维度组成：
1. 答案准确率（主奖励，权重最高）
2. Agent 行为质量（格式合规 + 工具使用合理性）
3. 工具效率（避免过度/不足调用）
"""

import re
import math

# ========== 辅助函数 ==========

def extract_answer(text: str) -> str:
    """从 Agent 输出中提取 <answer>...</answer> 的内容"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 回退：取最后一行
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""

def normalize_answer(s: str) -> str:
    """标准化答案用于比较"""
    s = str(s).strip().lower()
    # 去除常见格式符号
    s = s.replace('$', '').replace(',', '').replace('%', '').strip()
    # 尝试转为数字比较
    try:
        return str(round(float(s), 4))
    except ValueError:
        return s

def answers_match(pred: str, gold: str, tolerance: float = 0.01) -> bool:
    """
    判断预测答案和标准答案是否匹配
    支持数值型的近似匹配（tolerance 容差）
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # 字符串精确匹配
    if pred_norm == gold_norm:
        return True

    # 数值近似匹配
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        if gold_num == 0:
            return abs(pred_num) < tolerance
        return abs(pred_num - gold_num) / abs(gold_num) < tolerance
    except ValueError:
        pass

    # 包含匹配（答案在预测中）
    if gold_norm in pred_norm:
        return True

    return False


# ========== 奖励函数 1: 答案准确率 ==========

def accuracy_reward(completions: list, answer: list = None, **kwargs) -> list:
    """
    主奖励：答案是否正确
    正确: +1.0
    错误: +0.0
    """
    if answer is None:
        return [0.0] * len(completions)

    rewards = []
    for i, completion in enumerate(completions):
        pred = extract_answer(completion)
        gold = answer[i] if isinstance(answer, list) else answer

        if answers_match(pred, gold):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# ========== 奖励函数 2: Agent 行为质量 ==========

def agent_behavior_reward(completions: list, **kwargs) -> list:
    """
    Agent 行为质量奖励：
    - 有推理过程 (<think>): +0.1
    - 使用了至少一个工具: +0.15
    - 使用了多种工具: +0.1（额外加分）
    - 有结构化答案 (<answer>): +0.05
    - 过度调用工具 (>4次): -0.1
    """
    rewards = []
    for comp in completions:
        score = 0.0

        # 推理过程
        if "<think>" in comp and "</think>" in comp:
            score += 0.1

        # 工具使用统计
        search_count = comp.count("<search>")
        calc_count = comp.count("<calculate>")
        sql_count = comp.count("<sql>")
        total_calls = search_count + calc_count + sql_count

        # 使用了至少一个工具
        if total_calls >= 1:
            score += 0.15

        # 使用了多种工具（这是你的创新点！）
        tool_types_used = sum([search_count > 0, calc_count > 0, sql_count > 0])
        if tool_types_used >= 2:
            score += 0.1

        # 过度调用惩罚
        if total_calls > 4:
            score -= 0.1

        # 有结构化答案
        if "<answer>" in comp and "</answer>" in comp:
            score += 0.05

        rewards.append(score)
    return rewards


# ========== 奖励函数 3: 工具效率 ==========

def tool_efficiency_reward(completions: list, **kwargs) -> list:
    """
    工具效率奖励：
    - 搜索了但没有基于结果推理: -0.05
    - 没使用任何工具就给出答案: -0.05（对于需要数据的问题）
    """
    rewards = []
    for comp in completions:
        score = 0.0

        # 检查是否有 observation 后跟 think
        observations = re.findall(r"<observation>.*?</observation>", comp, re.DOTALL)
        thinks_after_obs = re.findall(
            r"</observation>.*?<think>(.*?)</think>", comp, re.DOTALL
        )

        if observations and not thinks_after_obs:
            # 有工具返回但没有后续推理
            score -= 0.05

        rewards.append(score)
    return rewards
```

## 6.2 Void Turn 过滤

```python
# training/void_turn_filter.py
"""
Void Turn Filtering (借鉴 SimpleTIR, 2025)

在多轮 Agent 训练中，"void turn" 是指：
某一轮 Agent 既没有产生有效的工具调用，也没有给出答案。

这些无效轮次会导致：
1. 低概率 token 的梯度特别大
2. 多轮累积后梯度爆炸

解决方法：检测到含有 void turn 的整条轨迹，从 GRPO batch 中移除。
"""

import re

def has_void_turn(trajectory: str) -> bool:
    """
    检查一条 Agent 轨迹是否包含无效轮次

    将轨迹按 </observation> 分割成多轮，
    每一轮要么有工具调用，要么有最终答案，
    否则就是 void turn。

    Args:
        trajectory: Agent 的完整输出文本

    Returns:
        True 如果包含 void turn
    """
    # 按 observation 分割（每个 observation 标志一轮结束）
    turns = re.split(r'</observation>', trajectory)

    for i, turn in enumerate(turns[:-1]):  # 最后一段可能只有答案
        has_tool_call = any(
            tag in turn for tag in ['<search>', '<calculate>', '<sql>']
        )
        has_answer = '<answer>' in turn
        has_think = '<think>' in turn

        # 既没有工具调用也没有答案 → void turn
        # 注意：纯 <think> 推理是允许的（Agent 在思考后再决定用什么工具）
        if not has_tool_call and not has_answer and not has_think:
            return True

    return False


def filter_void_trajectories(trajectories: list, rewards: list) -> tuple:
    """
    过滤包含 void turn 的轨迹

    Args:
        trajectories: Agent 轨迹列表
        rewards: 对应的奖励列表

    Returns:
        (filtered_trajectories, filtered_rewards)
    """
    filtered_trajs = []
    filtered_rewards = []
    void_count = 0

    for traj, reward in zip(trajectories, rewards):
        if has_void_turn(traj):
            void_count += 1
            continue
        filtered_trajs.append(traj)
        filtered_rewards.append(reward)

    if void_count > 0:
        print(f"  [Void Turn Filter] 过滤了 {void_count}/{len(trajectories)} "
              f"条轨迹 ({void_count/len(trajectories)*100:.1f}%)")

    return filtered_trajs, filtered_rewards
```

## 6.3 GRPO 主训练脚本（Unsloth 方案）

```python
# training/grpo_train_unsloth.py
"""
GRPO + LoRA + Unsloth 方案
显存效率最高，单卡 RTX 3090 可运行

注意：这个脚本展示的是简化版（单轮生成+reward）。
完整的多轮搜索交互需要自定义 rollout 循环，
详见 6.4 节的进阶方案。
"""

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import json
from training.reward_functions import (
    accuracy_reward, agent_behavior_reward, tool_efficiency_reward
)

# ===== 1. 加载模型 (Unsloth 4-bit 量化) =====
MODEL_NAME = "Qwen/Qwen2.5-3B"
# 如果有 SFT checkpoint，改为:
# MODEL_NAME = "checkpoints/sft/final"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,  # 自动检测
)

# 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ===== 2. 加载训练数据 =====
def load_training_data():
    """加载并格式化训练数据"""
    items = []
    with open("data/processed/train.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            items.append({
                "prompt": item["prompt"],
                "answer": item["answer"],
            })
    return items

from datasets import Dataset
raw_data = load_training_data()
dataset = Dataset.from_list(raw_data)
print(f"训练数据: {len(dataset)} 条")

# ===== 3. 训练配置 =====
training_args = GRPOConfig(
    output_dir="checkpoints/grpo",

    # 训练超参数
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # GRPO 特有参数
    num_generations=4,          # 每个 prompt 生成 4 个候选答案
    max_prompt_length=1024,     # prompt 最大长度
    max_completion_length=1536, # 生成最大长度

    # 显存优化
    bf16=True,
    gradient_checkpointing=True,

    # 日志
    logging_steps=5,
    save_steps=100,
    save_total_limit=5,
    report_to="wandb",
    run_name="finagent-grpo",

    # 梯度裁剪
    max_grad_norm=1.0,
)

# ===== 4. 启动训练 =====
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_funcs=[
        accuracy_reward,        # 权重最高：答案准确率
        agent_behavior_reward,  # Agent 行为质量
        tool_efficiency_reward, # 工具效率
    ],
)

print("=" * 60)
print("开始 GRPO Agent RL 训练")
print("=" * 60)
print(f"  模型: {MODEL_NAME}")
print(f"  训练数据: {len(dataset)} 条")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Generations per prompt: {training_args.num_generations}")
print(f"  Learning rate: {training_args.learning_rate}")
print("=" * 60)

trainer.train()

# ===== 5. 保存最终模型 =====
trainer.save_model("checkpoints/grpo/final")
tokenizer.save_pretrained("checkpoints/grpo/final")
print("\nGRPO 训练完成！模型保存至 checkpoints/grpo/final")
```

```bash
# 启动检索服务（另一个终端）
python tools/retrieval_server.py &

# 启动训练
python training/grpo_train_unsloth.py
```

## 6.4 TRL GRPOTrainer 方案（备选）

如果 Unsloth 安装有问题，用纯 TRL：

```python
# training/grpo_train.py
"""
纯 TRL GRPOTrainer 方案（不依赖 Unsloth）
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import json
import torch
from training.reward_functions import (
    accuracy_reward, agent_behavior_reward, tool_efficiency_reward
)

# 加载模型
MODEL_NAME = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 数据
raw_data = []
with open("data/processed/train.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        raw_data.append({"prompt": item["prompt"], "answer": item["answer"]})
dataset = Dataset.from_list(raw_data)

# LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
)

# 训练配置
training_args = GRPOConfig(
    output_dir="checkpoints/grpo",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=1536,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_steps=100,
    report_to="wandb",
    run_name="finagent-grpo-trl",
)

trainer = GRPOTrainer(
    model=model, args=training_args, train_dataset=dataset,
    tokenizer=tokenizer, peft_config=lora_config,
    reward_funcs=[accuracy_reward, agent_behavior_reward, tool_efficiency_reward],
)

trainer.train()
trainer.save_model("checkpoints/grpo/final")
```

## 6.5 训练监控与调参指南

```
训练过程中关注以下 W&B 指标：

1. reward/mean — 应该随训练逐步上升
   · 如果一直不涨：检查奖励函数是否有 bug
   · 如果突然下降：可能 collapse，降低 learning rate

2. reward/accuracy_reward — 答案准确率
   · 目标: 从 ~0.1 逐步上升到 ~0.4-0.6

3. loss — 应该整体下降
   · 如果 loss 变成 NaN：梯度爆炸，降低 lr 或增加 gradient clip

4. generation_length — 生成长度
   · 如果持续增长且不收敛：模型可能在胡说，检查格式奖励

调参建议（按优先级）：
· Learning rate: 先试 5e-6，如果不收敛试 1e-5，如果不稳定试 2e-6
· num_generations: 4 是好的起点，显存够可以试 8
· KL coefficient: 如果模型生成太保守，降低 KL；如果太发散，增加 KL
· 训练步数: 至少 200-500 steps，最好跑完整个 epoch
```

---

# 第七章：评测与消融实验

## 7.1 主评测脚本

```python
# eval/evaluate.py
"""
在 FinQA 测试集上评测模型

评测指标：
1. Execution Accuracy (EA): 答案数值是否正确
2. Program Accuracy (PA): 推理程序是否正确
3. Tool Usage Rate: Agent 使用工具的频率
4. Multi-Tool Rate: 使用多种工具的频率
"""

import json
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tools.tool_dispatcher import multi_turn_agent_rollout, extract_answer
from training.reward_functions import normalize_answer, answers_match

def load_model(model_path, base_model="Qwen/Qwen2.5-3B"):
    """加载训练好的模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    return model, tokenizer

def generate_fn_factory(model, tokenizer):
    """创建生成函数"""
    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=3072).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,  # 评测用低温度
                do_sample=True,
                top_p=0.95,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                              skip_special_tokens=True)
    return generate

def evaluate(model_path, test_file="data/processed/test.jsonl",
             max_samples=None):
    """运行评测"""
    print(f"加载模型: {model_path}")
    model, tokenizer = load_model(model_path)
    gen_fn = generate_fn_factory(model, tokenizer)

    # 加载测试数据
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"评测样本数: {len(test_data)}")

    # 评测指标
    metrics = defaultdict(int)
    results = []

    for i, item in enumerate(test_data):
        # 运行 Agent
        output = multi_turn_agent_rollout(
            generate_fn=gen_fn,
            prompt=item["prompt"],
            question_id=item.get("id"),
            max_turns=5,
        )

        pred = extract_answer(output["full_text"])
        gold = item["answer"]

        # 计算指标
        is_correct = answers_match(pred, gold)
        metrics["total"] += 1
        if is_correct:
            metrics["correct"] += 1

        # 工具使用统计
        tool_trace = output["tool_trace"]
        if tool_trace:
            metrics["used_tools"] += 1
            tools_used = set(t["tool"] for t in tool_trace)
            if len(tools_used) >= 2:
                metrics["multi_tool"] += 1
            for tool in tools_used:
                metrics[f"tool_{tool}"] += 1

        results.append({
            "id": item.get("id"),
            "question": item["question"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "tools_used": [t["tool"] for t in tool_trace],
            "num_tool_calls": len(tool_trace),
        })

        # 进度
        if (i + 1) % 50 == 0:
            acc = metrics["correct"] / metrics["total"] * 100
            print(f"  [{i+1}/{len(test_data)}] Accuracy: {acc:.1f}%")

    # 最终指标
    total = metrics["total"]
    print("\n" + "=" * 60)
    print("评测结果")
    print("=" * 60)
    print(f"  Execution Accuracy: {metrics['correct']}/{total} "
          f"= {metrics['correct']/total*100:.1f}%")
    print(f"  Tool Usage Rate:    {metrics['used_tools']}/{total} "
          f"= {metrics['used_tools']/total*100:.1f}%")
    print(f"  Multi-Tool Rate:    {metrics['multi_tool']}/{total} "
          f"= {metrics['multi_tool']/total*100:.1f}%")
    print(f"  Search calls:       {metrics.get('tool_search', 0)}")
    print(f"  Calculate calls:    {metrics.get('tool_calculate', 0)}")
    print(f"  SQL calls:          {metrics.get('tool_sql', 0)}")

    # 保存详细结果
    output_path = f"eval/results_{model_path.replace('/', '_')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": dict(metrics),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果保存至: {output_path}")

    return metrics

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/grpo/final"
    evaluate(model_path, max_samples=200)  # 先测 200 条看效果
```

## 7.2 消融实验脚本

```python
# eval/ablation.py
"""
消融实验：系统地比较不同配置的效果
这些数字将直接填入你的简历！
"""

from eval.evaluate import evaluate

experiments = [
    # Exp 1: 纯 LLM 基线（无工具）
    # → 直接用 base model 回答，不使用任何工具
    ("Qwen/Qwen2.5-3B", "Exp1: Base model (no tools)"),

    # Exp 2: Vanilla RAG 基线
    # → 检索 top-3 段落后拼接进 prompt，不做 RL
    # 需要单独实现 RAG pipeline

    # Exp 3: SFT 模型（冷启动后，无 RL）
    ("checkpoints/sft/final", "Exp3: SFT only (no RL)"),

    # Exp 4: GRPO 训练（仅 search 工具）
    # → 需要单独训练一个只有 search 的版本
    # ("checkpoints/grpo_search_only/final", "Exp4: GRPO + search only"),

    # Exp 5: GRPO 训练（三个工具，完整版）
    ("checkpoints/grpo/final", "Exp5: GRPO + 3 tools (full)"),
]

print("=" * 60)
print("消融实验")
print("=" * 60)

for model_path, exp_name in experiments:
    print(f"\n{'='*60}")
    print(f"实验: {exp_name}")
    print(f"模型: {model_path}")
    print(f"{'='*60}")
    try:
        metrics = evaluate(model_path, max_samples=200)
    except Exception as e:
        print(f"  ERROR: {e}")
```

```bash
python eval/ablation.py
```

**简历上的数字怎么算：**
```
"较 Vanilla RAG 提升 XX%"  = (Exp5_accuracy - Exp2_accuracy) / Exp2_accuracy × 100
"较无工具基线提升 XX%"     = (Exp5_accuracy - Exp1_accuracy) / Exp1_accuracy × 100
```

---

# 第八章：全栈 Agent Demo 开发

## 8.1 FastAPI 后端

```python
# demo/backend/main.py
"""
FinAgent-R1 Demo 后端
功能：
1. /api/ask - 非流式问答
2. /api/ask_stream - SSE 流式问答（实时展示 Agent 思考过程）
3. /api/health - 健康检查
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
sys.path.insert(0, '../..')
from tools.tool_dispatcher import (
    detect_tool_call, execute_tool, format_observation, extract_answer
)

app = FastAPI(title="FinAgent-R1 Demo", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("加载模型...")
    MODEL_PATH = "checkpoints/grpo/final"
    BASE_MODEL = "Qwen/Qwen2.5-3B"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, MODEL_PATH)
    model.eval()
    print("模型加载完成")


class QueryRequest(BaseModel):
    question: str
    question_id: str = None


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/api/ask")
async def ask(req: QueryRequest):
    """非流式问答"""
    prompt = f"""You are a financial analysis agent with access to search, calculate, and SQL tools.

Tools:
1. <search>query</search> - Search financial knowledge base
2. <calculate>expression</calculate> - Execute financial calculations
3. <sql>SQL query</sql> - Query financial data tables

Use <think>...</think> for reasoning. Give final answer in <answer>...</answer>.

Question: {req.question}"""

    full_text = ""
    tool_trace = []
    current_input = prompt

    for turn in range(5):
        inputs = tokenizer(current_input, return_tensors="pt",
                          truncation=True, max_length=3072).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512,
                temperature=0.7, do_sample=True,
            )
        new_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        full_text += new_text

        tool_name, query, _ = detect_tool_call(new_text)
        if tool_name:
            result = execute_tool(tool_name, query, question_id=req.question_id)
            observation = format_observation(result)
            full_text += observation
            tool_trace.append({
                "turn": turn + 1,
                "tool": tool_name,
                "query": query,
                "result": result[:500],
            })
            current_input = prompt + full_text
        else:
            break

    answer = extract_answer(full_text)

    return {
        "answer": answer,
        "full_text": full_text,
        "tool_trace": tool_trace,
        "num_tool_calls": len(tool_trace),
    }


@app.post("/api/ask_stream")
async def ask_stream(req: QueryRequest):
    """SSE 流式问答"""
    async def generate():
        prompt = f"""You are a financial analysis agent. Question: {req.question}"""
        # ... 类似逻辑，但每一步都 yield SSE event
        yield json.dumps({"type": "thinking", "content": "Analyzing the question..."})
        await asyncio.sleep(0.1)
        yield json.dumps({"type": "answer", "content": "Demo answer"})

    return EventSourceResponse(generate())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 8.2 启动 Demo

```bash
# 终端 1: 启动检索服务
python tools/retrieval_server.py &

# 终端 2: 启动后端
cd demo/backend
python main.py

# 测试
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the percentage change in revenue from 2018 to 2019?"}'
```

## 8.3 React 前端（简化版，用 Gradio 替代）

如果你不熟悉 React，用 Gradio 快速搭一个也完全可以：

```python
# demo/gradio_app.py
"""
Gradio 版 Demo（5 分钟搭完）
"""

import gradio as gr
import requests

def ask_agent(question):
    resp = requests.post(
        "http://localhost:8000/api/ask",
        json={"question": question},
        timeout=60,
    )
    data = resp.json()

    # 格式化输出
    output = f"**Answer:** {data['answer']}\n\n"
    output += f"**Tool Calls:** {data['num_tool_calls']}\n\n"

    for trace in data['tool_trace']:
        output += f"---\n"
        output += f"**Turn {trace['turn']}:** `{trace['tool']}` → `{trace['query']}`\n"
        output += f"```\n{trace['result'][:300]}\n```\n\n"

    output += f"---\n**Full reasoning:**\n```\n{data['full_text'][:2000]}\n```"
    return output

demo = gr.Interface(
    fn=ask_agent,
    inputs=gr.Textbox(label="Financial Question", lines=3,
                      placeholder="e.g., What was the revenue growth rate in 2019?"),
    outputs=gr.Markdown(label="Agent Response"),
    title="FinAgent-R1: Financial Multi-Tool Agent",
    description="An RL-trained agent that autonomously uses search, calculation, and SQL tools.",
    examples=[
        ["What was the percentage change in total revenue from 2018 to 2019?"],
        ["Calculate the gross margin ratio if revenue was $5.2B and COGS was $3.1B."],
        ["What is the compound annual growth rate of earnings over the past 3 years?"],
    ],
)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
```

```bash
pip install gradio
python demo/gradio_app.py
# 打开 http://localhost:7860
```

## 8.4 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  retrieval:
    build:
      context: .
      dockerfile: Dockerfile.retrieval
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    depends_on:
      - retrieval
    environment:
      - RETRIEVAL_URL=http://retrieval:5000/search
      - MODEL_PATH=/models/grpo_final
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  demo:
    build:
      context: .
      dockerfile: Dockerfile.demo
    ports:
      - "7860:7860"
    depends_on:
      - backend
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
- 💻 **Single GPU training** — Qwen2.5-3B + GRPO + LoRA on RTX 3090/4090
- 🚀 **Full-stack demo** — FastAPI + Gradio + Docker Compose

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
- [SimpleTIR](https://github.com/ltzheng/SimpleTIR) — Void turn filtering
- [FinQA](https://github.com/czyssrs/FinQA) — Financial QA dataset
- [Unsloth](https://github.com/unslothai/unsloth) — Efficient training
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

• 使用 Qwen2.5-3B + GRPO + LoRA 在单卡 RTX 3090 上完成训练，
  实现了 Retrieved Token Masking + Void Turn Filtering 双重稳定化机制。

• 设计了复合奖励函数（答案准确率 + 行为合规性 + 工具效率），
  在 FinQA 测试集上较 Vanilla RAG 提升约 XX% Execution Accuracy。

• 部署为全栈 Agent 应用（FastAPI + Gradio/React），
  支持实时展示 Agent 思考链和工具调用轨迹，Docker 一键部署。
```

## 10.2 面试必备 Q&A

### Q1: "Search-R1 和你的项目的关系是什么？"

Search-R1 是一个开源的 RL 训练框架，发表在 COLM 2025，
原版只支持搜索引擎一个工具。我在此基础上做了三件事：
1. 迁移到金融领域（换数据集、换检索语料）
2. 扩展为多工具 Agent（加了计算器和 SQL 查询）
3. 加入了 Void Turn Filtering 稳定化技术
4. 做了全栈部署 Demo

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

可以。启动 Docker，输入一个问题比如
"What was the CAGR of revenue from 2017 to 2019?"，
实时看到 Agent 先搜索研报找到数据，
再调用计算器算 CAGR，最后给出答案。

---

## 检查清单

```
□ Day 1:   conda 环境 + 全部依赖安装
□ Day 2:   下载 FinQA + 预处理 + 构建语料库 + 构建 SQL 数据库
□ Day 3:   BM25 索引 + FAISS 索引 + 三个工具实现 + 集成测试
□ Day 4:   生成 SFT 种子数据 + SFT 冷启动训练
□ Day 5:   验证 SFT 模型 + 编写奖励函数 + 配置 GRPO
□ Day 6-7: GRPO 训练（可能需要多次调参）
□ Day 8:   跑消融实验（至少 4-5 组对比）
□ Day 9:   整理实验结果 + 制作表格
□ Day 10:  FastAPI 后端开发
□ Day 11:  Gradio/React 前端开发
□ Day 12:  Docker 打包
□ Day 13:  README + 文档 + .gitignore
□ Day 14:  GitHub 发布 + 简历更新
```

---

> **最后提醒：** 简历上所有 XX% 的占位符，必须替换为你跑完实验后的真实数字。
> 真实数据即使低于预估也没关系，诚实比夸大重要得多。
> 面试官问的不是"你的数字有多高"，而是"你理不理解为什么是这个数字"。
