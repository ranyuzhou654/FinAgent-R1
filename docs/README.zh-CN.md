# FinAgent-R1 🏦🔎🧮

[English README 🇺🇸](../README.md)

FinAgent-R1 是一个受 Search-R1 启发的金融多工具 Agent 项目。它的目标不是只做
“检索增强生成”，而是训练一个模型在多轮交互中自主决定何时调用搜索、计算器和 SQL 工具。

## ✨ 仓库包含什么

- 📚 面向 FinQA 的数据处理流水线：从原始样本到 JSONL，再到 veRL parquet。
- 🛠️ 三类工具后端：金融文本检索、金融计算、SQLite 结构化查询。
- 🤖 以 Search-R1 风格 veRL 多轮 GRPO 为主线的训练方案。
- 🧪 可选的 SFT、TRL、Unsloth 兼容训练脚本。
- 🖥️ 评测脚本、FastAPI 后端、Gradio Demo、React Demo 和 Docker 部署文件。

## 🧱 项目结构

仓库可以理解为六层：

1. 📦 数据准备层
2. 🔍 工具层
3. 📝 SFT 冷启动可选层
4. 🚀 veRL 多轮 RL 主训练层
5. 🔁 TRL / Unsloth 兼容训练层
6. 📊 评测与 Demo 层（FastAPI、Gradio、React、Docker）

```
FinAgent-R1/
├── data/                          # 数据流水线产物
│   ├── raw/                       #   原始下载数据
│   ├── processed/                 #   训练/验证/测试 JSONL
│   ├── corpus/                    #   检索语料库
│   ├── indexes/                   #   BM25 + FAISS 索引
│   └── tables/                    #   SQLite 数据库
├── tools/                         # 三类工具后端
│   ├── search_tool.py             #   金融文本检索
│   ├── calculator_tool.py         #   金融计算引擎
│   ├── sql_tool.py                #   SQLite 查询执行
│   ├── tool_dispatcher.py         #   统一工具调度器
│   └── retrieval_server.py        #   FastAPI 检索服务
├── training/                      # 训练脚本
│   ├── finagent_verl_main.py      #   veRL 主训练入口（主线）
│   ├── finagent_generation.py     #   多轮 generation manager
│   ├── reward_functions.py        #   规则奖励函数
│   ├── search_r1_compat.py        #   Search-R1 vendor 兼容层
│   ├── tensor_helper.py           #   veRL 张量工具
│   ├── sft_coldstart.py           #   可选：SFT 冷启动
│   ├── grpo_train.py              #   可选：纯 TRL GRPO
│   ├── grpo_train_unsloth.py      #   可选：Unsloth GRPO
│   └── void_turn_filter.py        #   空轮次检测过滤
├── eval/                          # 评测
│   ├── evaluate.py                #   主评测脚本
│   └── ablation.py                #   消融实验
├── demo/                          # Demo 界面
│   ├── backend/main.py            #   FastAPI 后端
│   ├── gradio_app.py              #   Gradio 前端
│   └── frontend/src/              #   React 前端（完整实现）
├── scripts/                       # 流水线与工具脚本
├── configs/                       # 训练配置（veRL、SFT、TRL）
├── vendor/Search-R1/              #   vendored Search-R1 运行时
├── docker-compose.yml             # Docker 编排
├── Dockerfile.{backend,retrieval,demo}
├── README.md
└── requirements.txt
```

## ⚡ 快速开始

1. 创建 Python 3.10+ 环境
2. 安装依赖
3. 准备 FinQA 数据
4. 构建检索索引和 SQLite 数据库
5. 转换 veRL parquet 数据
6. 测试工具链
7. 启动主训练

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_training_data.py
python scripts/build_corpus.py
python scripts/build_sql_database.py
python scripts/build_question_table_map.py
bash scripts/build_indexes.sh
python scripts/prepare_verl_finqa_data.py
python scripts/test_tools.py
```

## 📘 使用指南

### 1. 🧰 环境准备

基础环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你要构建 BM25 检索索引，`pyserini>=0.41.0` 需要 `JDK 21`。先确认当前 shell
看到的就是正确版本：

```bash
java -version
python -c "import pyserini; print('pyserini OK')"
```

macOS 常见配置方式：

```bash
brew install --cask temurin@21
export JAVA_HOME=$(/usr/libexec/java_home -v 21)
export PATH="$JAVA_HOME/bin:$PATH"
```

如果要跑 React 前端：

```bash
cd demo/frontend
npm install
cd ../..
```

### 2. 🏗️ 数据流水线

原始数据默认放在 `data/raw/finqa_hf`。推荐按下面顺序执行：

```bash
# 可选：先看数据结构
python scripts/explore_data.py

# 原始 FinQA -> train/validation/test JSONL
python scripts/prepare_training_data.py

# 构建检索语料和结构化数据
python scripts/build_corpus.py
python scripts/build_sql_database.py
python scripts/build_question_table_map.py

# 构建 BM25 + dense 检索索引
bash scripts/build_indexes.sh

# JSONL -> veRL parquet
python scripts/prepare_verl_finqa_data.py
```

如果这里报 `jdk.incubator.vector not found`，说明当前 shell 仍然在使用过低版本的
Java，需要切到 JDK 21 后再重新执行。

关键产物包括：

- `data/processed/train.jsonl`
- `data/corpus/financial_passages.jsonl`
- `data/tables/financial_data.db`
- `data/tables/question_table_map.json`
- `data/verl/finqa/train.parquet`

### 3. ✅ 工具链检查

在训练之前，先确认三类工具都能正常工作：

```bash
python scripts/test_tools.py
```

### 4. 🤖 主训练路径

当前主训练路径是 veRL 多轮 GRPO。

终端 1：

```bash
python tools/retrieval_server.py
```

终端 2：

```bash
bash scripts/train_grpo_verl.sh
```

常见覆写方式：

```bash
bash scripts/train_grpo_verl.sh \
  trainer.total_training_steps=100 \
  trainer.save_freq=50 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B
```

默认配置文件：

- `configs/verl_ppo_finqa.yaml`

默认输出目录：

- `checkpoints/grpo_verl`

## 🧪 可选训练分支

如果你想先做冷启动或使用更轻量的兼容训练脚本，可以走下面这条支线：

```bash
python scripts/generate_sft_data.py
python training/sft_coldstart.py
python scripts/test_sft_model.py
python training/grpo_train.py
```

额外可用脚本：

- `python training/grpo_train_unsloth.py`

这条支线不是当前主线，但适合做小规模验证或对照实验。

## 🖥️ Demo 使用方式

支持四种界面：

- FastAPI 后端
- Gradio
- React 前端
- Docker Compose

典型本地启动方式：

终端 1：

```bash
python tools/retrieval_server.py
```

终端 2：

```bash
MODEL_PATH=Qwen/Qwen2.5-3B python demo/backend/main.py
```

终端 3 任选其一：

```bash
python demo/gradio_app.py
```

或：

```bash
cd demo/frontend
npm run dev
```

后端接口：

- `GET /api/health`
- `POST /api/ask`
- `POST /api/ask_stream`

如果你想用 Docker：

```bash
docker compose up --build
```

## 📏 评测方式

主评测：

```bash
python eval/evaluate.py Qwen/Qwen2.5-3B --max-samples 20
```

消融：

```bash
python eval/ablation.py
```

当前评测指标包括：

- ✅ Execution Accuracy
- 🧩 启发式 Program Accuracy
- 🛠️ Tool Usage Rate
- 🔀 Multi-Tool Rate

## 🔄 Pipeline 原理说明

### 🏋️ 训练流水线

训练流程可以概括为：

1. 原始 FinQA 样本先转换成带 `prompt / answer / program / context / table` 的 JSONL。
2. 报表文本被切成检索语料，表格数据进入 SQLite，并建立 `question_id -> table_name` 映射。
3. 构建 BM25 和 dense 检索索引。
4. 把处理后的 JSONL 转成 Search-R1 风格的 veRL parquet。
5. 模型在训练时生成多轮轨迹，例如 `<search>...</search>`、`<sql>...</sql>`。
6. 工具执行结果作为 `<observation>...</observation>` 回填给模型继续推理。
7. 用规则奖励函数对答案正确性和 Agent 行为打分，再通过 GRPO 更新策略。

### 🧭 推理流水线

在线推理、评测和 Demo 基本共享同一套 agent loop：

1. 构建 system + user prompt
2. 生成下一段 assistant 输出
3. 检测是否出现工具标签
4. 调用对应后端
   - `search -> 检索`
   - `calculate -> 计算器`
   - `sql -> SQLite`
5. 把结果包成 observation 回填
6. 持续多轮，直到输出 `<answer>...</answer>` 或达到回合上限

### 🆚 为什么不是普通 RAG

这个项目的核心不是“把 top-k 文本拼进 prompt”，而是训练模型做决策：

- 需不需要工具
- 该用哪个工具
- 是否要多工具联动
- 什么时候停止调用工具
- 如何利用 observation 继续推理

所以它更接近一个多轮 Agent，而不是普通单轮 RAG。

### 🎯 奖励设计

当前奖励设计大致由以下部分组成：

- 答案正确性
- 输出格式合规性
- 是否合理使用工具
- 是否实现多工具协同
- 对无工具回答、过度调用、无效 retry 的惩罚

veRL 主线直接消费这些奖励；TRL / Unsloth 兼容脚本通过适配接口复用相同逻辑。

## 📝 备注

- 默认依赖使用 `faiss-cpu`，如果你的训练环境有 GPU，可以替换成更合适的 FAISS 版本。
- 当前主训练路径是 veRL；TRL 和 Unsloth 是兼容分支。
- `vendor/Search-R1/` 被直接 vendored 到仓库里，作为 veRL 运行时依赖。
- 大体积数据、索引、checkpoint、实验日志默认不会进入 git。
