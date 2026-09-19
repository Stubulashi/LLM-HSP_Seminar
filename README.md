# IPLE — Incremental Pragmatic LLM Evaluation

研究开源 LLM 在逐步接收语言信息时如何动态形成、更新与修正语用解释的自动化实验框架（过程导向评估，proj.md L19-75）。

设计文档位于 `proj/`：proj.md（研究蓝图）、pipeline.md（系统设计 v1）、scaf.md（工程化 v2）。工程契约以 `docs/decisions.md` 为准（含 C1-C15 冲突裁决）。

## 数据流

```
raw story → 标注 → 校验 → 人工审查 → 增量实验（逐句累加 context）→ 原始响应 → 指标 → 统计结论
```

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py setup
```

## CLI 命令

| 命令 | 说明 |
|---|---|
| `python main.py setup` | 环境自检（Python/依赖/GPU）+ 创建数据目录，输出 `[OK]` 报告 |
| `python main.py annotate [--input data/raw_stories/]` | 对原始故事运行标注管线，产出结构化 JSON 至 data/annotated/ |
| `python main.py review <story_id>` | 人工审查标注（可修改 function/gold_answer/critical_sentence） |
| `python main.py run --model <m> [--task <t>] [--stories <s>] [--repetitions <n>]` | 增量实验，结果存 results/raw/{model}/{task}/{story_id}/run{N}.json |
| `python main.py analyze` | 计算指标并产出 accuracy.csv / emergence.csv / trajectory_similarity.csv / trajectory.png |

## 规模

- 原始设计（proj.md L971-991）：60 故事（Faux Pas 30 / False Belief 15 / Implicature 15）× 4 模型 × 3 重复 = 720 会话。
- 实际执行（详见 `docs/experiment_report.md`）：280 故事 × 10 个模型档 × 3 重复；正式 runs 9290（Condition A 6770 含量化对照 50 + Condition B 2520），judge 判定 9290/9290。计数口径：按 `task` 字段 = false_belief 70 / faux_pas 10 / implicature 200；按来源前缀 = fb 60 / sfp 20（sfp001-010 = faux_pas；sfp011-020 = false_belief 对照）。

## 目录结构

```
config/          # models.yaml / experiment.yaml / prompts.yaml / config_manager.py
data/            # raw_stories/（原始故事，只读） annotated/（标注 JSON）
annotation/      # story_parser / annotation_agent / validator / human_review
models/          # base_model / hf_model / factory
experiment/      # prompt_builder / incremental_runner / recorder / scheduler
analysis/        # confidence / accuracy / emergence / embedding / stability / statistics
visualization/   # plots.py
utils/           # logger / seeding
data_manager/    # 数据读写（原子写）
results/         # raw/ processed/ embeddings/（不入库）
logs/            # 运行日志（不入库）
tests/ scripts/  # 单测与冒烟测试
docs/            # decisions.md（工程契约）
```

## 规则速览

- 模块低耦合：实验引擎不感知模型实现；禁止 `if model=="qwen"` 式硬编码
- 可复现：每次 run 记录 model/prompt/dataset version、temperature、seed、timestamp
- 中间结果全保存：禁止"加载→运行→直接算分→丢弃响应"
- 所有模型只通过 config/models.yaml 管理，加模型 = 新增配置 + 实现 generate，实验代码零改动

详细规则与里程碑说明见 `.qoder/skills/iple-workflow/SKILL.md`。

## 数据材料（大文件，不入 git）

实验数据与产物按 `.gitignore` 策略不进入版本库（`data/`、`results/`、`logs/`）。完整归档：

- 文件：`cloud_full_final.tar.gz`（100,254,135 字节；72,719 条目；含 config、data（annotated 280 + manual_equivalence 4）、results（raw 6770 / processed 13 / processed_judge 8 / cosine_4200 6 / embeddings 44,064）与 results_condB（raw 2520 / processed 15 / embeddings 15,831））
- 下载：<待填写：网盘或 GitHub Release 链接>
- 校验：`MD5 = adb134917178008bfb2cb9d996724eb6`；`SHA256 = 82c72d837307d06e334fc252fda4bd62c2b8a56176fade2092aa15711d01a2b5`（复核：`certutil -hashfile cloud_full_final.tar.gz SHA256`）
- 解包：`tar -xzf cloud_full_final.tar.gz -C <目标目录>`（顶层为 `config/`、`data/`、`results/`、`results_condB/`）
