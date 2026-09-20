# IPLE — 增量语用理解的 LLM 评估框架

IPLE（Incremental Pragmatic LLM Evaluation）用来观察大语言模型在逐句接收故事信息的过程中，如何形成、更新和修正对语用含义的理解。运行一篇故事时，实验程序把句子一句一句地交给模型，收集它每一步的判断（当前情境解释、角色意图、0-100 的置信度），再对照关键句、标准答案和人工标注，计算理解在哪个位置出现、推理轨迹稳不稳定、最终回答对不对。

任务覆盖三类语用现象：faux pas（社交失言）、false belief（错误信念）和 implicature（会话含义）。

本仓库是研究项目（LLM-HSP Seminar）的核心代码发布版：源码、配置、脚本与测试都在这里；实验数据、报告和归档文件不在仓库内，原因与获取方式见「数据与材料」一节。

## 目录结构

```
main.py            CLI 入口，分发 setup / annotate / review / run / analyze
annotation/        标注管线：故事解析、LLM 标注、结果校验、人工审查
models/            模型层：统一接口 + HuggingFace / vLLM / Mock 实现 + 工厂
experiment/        实验引擎：prompt 组装、逐句运行、结果记录、任务调度
analysis/          分析：准确率、涌现、置信度、嵌入、稳定性、统计、LLM judge
config/            配置：三个 yaml + 配置管理器
data_manager/      数据读写（原子写）
utils/             日志、随机种子、环境自检
visualization/     绘图工具
scripts/           实验期间的辅助脚本（打包、批跑、评估、统计、绘图等）
tests/             单元测试
```

各模块的职责：

- **annotation/**：把原始故事文本转成结构化 JSON（逐句功能、关键句、标准答案、问题），用 `annotation_model` 指定的模型跑标注；`validator.py` 检查结构合法性，`human_review.py` 提供人工审查。
- **models/**：所有模型实现同一个接口（load / generate / unload）。`ModelFactory` 读 `models.yaml` 决定实例化哪个实现、如何量化；`MockModel` 是确定性假模型，用于测试和冒烟。
- **experiment/**：`PromptBuilder` 按 `prompts.yaml` 组装每一步的输入；`IncrementalRunner` 维护逐句累加的 context 并调用模型；`ResultRecorder` 将每一步的原始响应落盘；`ExperimentScheduler` 生成并执行"模型 × 任务 × 故事 × 重复"的任务矩阵。
- **analysis/**：`accuracy` / `emergence` / `confidence` / `embedding` / `stability` / `statistics` 分别计算各指标；`judge.py` 提供基于 LLM 的等价判定，替代或补充余弦相似度口径。
- **config/**：`ConfigManager` 加载并校验三个 yaml，是代码读取配置的唯一入口。
- **utils/**：统一日志（写入 `logs/iple.log`）、随机种子设置（保证同种子同输出）、`main.py setup` 的环境自检逻辑。
- **data_manager/**：统一管理 `data/` 与 `results/` 的读写路径；写入走临时文件加替换，避免写中途损坏。
- **scripts/**：实验期间积累的辅助脚本，按用途分几类：云端打包与执行（`pack_for_cloud.py`、`cloud_run.py`、`run_full_class_batch.py`）、模型与流程验证（`verify_models.py`、`smoke_test.py`）、judge 相关（`judge_sample.py`、`judge_analyze.py`、`run_judge_on_pool.py`、`assess_judges.py`）、统计与报表（`mixed_effects.py`、`confidence_analysis.py`、`build_report_tables_v2.py`、`calibrate_cosine_threshold.py` 等）、绘图（`plot_*.py`）、报告草稿同步（`sync_report_draft*.py`）。这些脚本面向项目内部流程，单独使用时需要按实际情况调整输入输出路径的参数。
- **tests/**：见「测试」一节。

## 运行环境与依赖

需要 Python 3.10 及以上（`main.py setup` 会检查版本）。依赖清单在 `requirements.txt`：

| 包 | 用途 |
|---|---|
| torch / transformers / accelerate / bitsandbytes | 本地 HuggingFace 后端推理与 4bit/8bit 量化 |
| vllm | GPU 上跑大模型档（7B/14B/32B）的推理后端 |
| sentence-transformers | 默认嵌入模型 all-MiniLM-L6-v2，用于余弦口径的 accuracy |
| pandas / statsmodels | 指标汇总与混合效应统计 |
| pyyaml / typer / tqdm / pytest | 配置、命令行、进度条、测试 |

安装与自检：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Linux / macOS 用 source .venv/bin/activate
pip install -r requirements.txt
python main.py setup
```

`setup` 会逐项输出 [OK] / [WARN]：Python 版本、依赖包、GPU 可用性（没有 CUDA 会降级提醒），并创建 `data/raw_stories`、`data/annotated`、`results/raw`、`results/processed`、`results/embeddings`、`logs` 目录，然后加载配置。

两点说明：`vllm` 只在用 GPU 跑大模型档时才需要，它对本机 torch 版本有强约束，没有 GPU 的环境可以删掉这一行再安装，不影响小模型档和全部测试；模型权重首次使用时从 HuggingFace 下载并缓存在本地，之后离线可用。

## 配置说明

`config/` 下三个 yaml 是全部实验参数的来源，代码里没有散落的魔法数字。

### models.yaml

模型的唯一事实源，"加一个模型"等于在这里加一段配置，实验代码不用改。每个条目的字段：

| 字段 | 含义 |
|---|---|
| family / path | 家族名与 HuggingFace 仓库 ID |
| backend | `huggingface`（本地推理）或 `vllm`（GPU 高吞吐） |
| quantization | none / 4bit / 8bit / awq / gptq / fp8；vLLM 要求预量化权重 |
| dtype | bfloat16 / float16（AWQ 只支持 float16） |
| size / training | 统计口径用（模型规模、instruction / reasoning） |
| gpu_memory_utilization / max_model_len / max_num_seqs / enable_prefix_caching | vLLM 专用旋钮，显存不够时优先下调前三项 |

预置条目有：`qwen05b`、`qwen15b`（本地小模型，1.5B 用 4bit 量化）、`qwen7b`、`qwen14b`、`qwen32b`、`deepseek7b`、`deepseek14b`、`deepseek32b`（vLLM + AWQ），以及对照用的 `qwen14b_bf16`、`deepseek14b_bf16`、`qwen05b_vllm`、`qwen15b_vllm`。

### experiment.yaml

- `experiment` 段：`seed_master`（母种子）、`temperature`、`max_tokens`、`repetitions`（每个任务重复几次）、`annotation_model`（标注用的模型键，默认 `qwen7b`）。
- `scoring` 段：`method`（`embedding_similarity` / `local_judge` / `api_judge`）、`judge_provider`（`local` / `deepseek` / `kimi`）、`threshold`（余弦阈值，只对 embedding 模式生效）、`embedding_model`（默认 `sentence-transformers/all-MiniLM-L6-v2`）。

### prompts.yaml

- `prompt_version`：写进实验记录的版本号。
- `conditions`：`condition_a` / `condition_b` 两套系统提示与起手模板（后者是"分析理解如何改变"的变体）。
- `tasks`：`default` / `false_belief` / `faux_pas` / `implicature` 四种答题模板，统一要求置信度放在最后一行。
- `annotation`：标注用的结构化 prompt。

### .env（不随仓库分发）

`api_judge` 模式需要 API 密钥，程序从项目根目录的 `.env` 读取（文件本身已被忽略，不要提交）：

```
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com    # 可省略，用默认值
DEEPSEEK_MODEL=deepseek-chat                   # 可省略，用默认值
KIMI_API_KEY=your-key                          # 使用 kimi 供应商时
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=moonshot-v1-8k
```

密钥缺失或请求失败时程序会直接报错，不会静默降级。

## 命令行用法

| 命令 | 说明 |
|---|---|
| `python main.py setup` | 环境自检并创建数据目录 |
| `python main.py annotate [--input 目录] [--task 任务] [--mock]` | 对原始故事跑标注，输出到 `data/annotated/*.json`；任务默认从输入目录名推断 |
| `python main.py review <story_id>` | 交互式人工审查标注（可修改句子功能、标准答案、关键句），完成后打上 reviewed 标记 |
| `python main.py run --model 模型键 [--task 任务] [--stories 列表] [--repetitions N] [--seed N] [--mock]` | 增量实验；只处理 reviewed 的故事；结果写入 `results/raw/模型/任务/故事/runN.json` |
| `python main.py analyze [--scoring-method 口径] [--judge-provider 供应商] [--results-dir 目录]` | 计算指标，CSV 输出到 `results/processed/` |

几个典型用法：

```powershell
# 标注 faux_pas 任务的全部原始故事
python main.py annotate --input data/raw_stories/ --task faux_pas

# 审查一篇标注
python main.py review fp001

# 跑增量实验（3 次重复取 experiment.yaml 的默认值，也可用 --repetitions 覆盖）
python main.py run --model qwen7b --task faux_pas --stories fp001,fp002

# 默认余弦口径分析
python main.py analyze

# 论文口径：DeepSeek API judge（需要 .env 密钥）
python main.py analyze --scoring-method api_judge --judge-provider deepseek

# Condition B 的结果目录
python main.py analyze --results-dir results_condB
```

`--mock` 是不加载真实模型的测试设施：`annotate`、`run` 都可以用 MockModel 走通整条链路，适合在没有 GPU、没有模型权重的机器上做冒烟。

## 数据与材料

仓库不含实验数据，原因有两条：体积（原始响应、嵌入和归档合计数百 MB 到 1 GB）和授权（FauxPas 材料只允许在研究内部使用，不能随公开仓库分发）。具体不包含：

- `data/`：原始故事、标注 JSON、人工等价判定子集；
- `results/`：原始响应、指标 CSV、嵌入文件；
- `datasets/`：FauxPas、OpenToM、SwordsmanImp 三个数据集的素材；
- `docs/`：报告、图表与工程决策记录；
- `logs/` 与云端备份归档。

完整归档文件 `cloud_full_final.tar.gz`（100,254,135 字节，72,719 个条目）包含：config、data（annotated 280 + manual_equivalence 4）、results（raw 6770、processed、embeddings 44,064）和 results_condB（raw 2520、embeddings 15,831）。

- 下载地址：（待补充：网盘或 GitHub Release 链接）
- 校验方式（下载后在所在目录执行）：

```powershell
certutil -hashfile cloud_full_final.tar.gz MD5
# 期望：adb134917178008bfb2cb9d996724eb6
certutil -hashfile cloud_full_final.tar.gz SHA256
# 期望：82c72d837307d06e334fc252fda4bd62c2b8a56176fade2092aa15711d01a2b5
```

- 解包：`tar -xzf cloud_full_final.tar.gz -C <目标目录>`，顶层是 `config/`、`data/`、`results/`、`results_condB/` 四个目录。

归档内含 FauxPas 衍生数据，只可按研究内部用途传递，不要公开发布或再分发。

另外还有一个小体积的云端工作包 `HSP_cloud.zip`（代码 + 标注数据，用于上传到 GPU 机器跑批量实验），随时可以本地重新生成：

```powershell
python -X utf8 scripts/pack_for_cloud.py
```

上传后在服务器上解压，用 `scripts/cloud_run.py` 执行（支持小规模 preflight 与全量运行）。

供对照的实验规模：280 篇故事 × 10 个模型档 × 3 次重复；正式 runs 9290（Condition A 6770、Condition B 2520），judge 判定 9290/9290。

## 复现步骤

复现需要 Python 环境和数据两部分。数据受限，所以完整复现要先拿到归档或自备符合授权的故事文本。

从原始故事开始（标注链路）：

1. 安装依赖并运行 `python main.py setup`；
2. 把原始故事按任务放进 `data/raw_stories/<任务名>/`；
3. `python main.py annotate --input data/raw_stories/ --task <任务名>`；
4. `python main.py review <story_id>` 人工确认每篇标注（未审查的故事不会进入实验）；
5. `python main.py run --model <模型键> [--task/--stories/--repetitions]`；
6. `python main.py analyze`，需要论文口径则加 `--scoring-method api_judge --judge-provider deepseek`（自备密钥）。

使用归档数据（跳过标注）：

1. 解包归档，把 `data/annotated/` 放到项目的 `data/annotated/`；
2. 同样从 `setup` 开始，随后直接 `run`／`analyze`；
3. 本地 4GB 显存环境可以跑 `qwen05b`、`qwen15b` 两档小模型；大模型档需要 GPU 服务器加 vLLM。

## 测试

在项目根目录执行：

```powershell
python -m pytest tests/ -q
```

6 个测试文件分别覆盖：指标计算与嵌入缓存（test_analysis）、故事解析与标注格式（test_annotation）、实验引擎与调度（test_experiment）、judge 输出解析与缺密钥报错（test_judge）、模型工厂分发（test_model_factory）、同种子输出一致性与"加模型不改代码"的扩展性（test_reproducibility）。

测试用 MockModel 和哈希嵌入（HashEmbedder）覆盖主要链路，不加载真实模型权重，也不需要 GPU，普通 CPU 环境即可运行。

## 版权、引用与许可

- FauxPas 材料（FauxPas_Adult.pdf 原件及由它衍生的故事文本、标注、判定数据和报告内容）仅限研究内部使用，禁止二次分发。本仓库不包含任何 FauxPas 原文或派生内容；通过外部归档获取的数据同样受此限制。
- OpenToM、SwordsmanImp 等第三方数据集素材不在本仓库内，取用与再分发请遵循各数据集自身的许可。
- 本仓库未附 LICENSE 文件。代码用于课程研究项目，若要在其他场景引用或复用，请先通过本仓库的 GitHub 页面联系作者。
- 引用本项目时请注明仓库地址与作者；报告和论文的引用信息以项目正式报告为准。
