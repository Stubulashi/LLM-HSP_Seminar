# IPLE — An Evaluation Framework for Incremental Pragmatic Understanding in LLMs

IPLE (Incremental Pragmatic LLM Evaluation) studies how a large language model builds, updates and revises its interpretation of pragmatic meaning while it receives story information sentence by sentence. For each story the program feeds the model one sentence at a time and records what the model produces at every step: its current interpretation of the situation, its inference about the characters' intentions, and a confidence score from 0 to 100. The step-level records are then compared with the critical sentence, the gold answer and the human-reviewed annotation, so that we can compute where in the story understanding emerges, how stable the reasoning trajectory is, and whether the final answer is correct.

The experiments cover three pragmatic phenomena: faux pas, false belief and implicature.

This repository is the core code release of a research project (LLM-HSP Seminar). It contains the source code, the configuration files, the analysis scripts and the tests, together with the experiment outputs produced on the cloud GPU server (`results/` and `results_condB/`). The original story materials are not included, and the bulk archives are kept outside git; see "Data and Materials（数据与材料）".

## Repository Layout（目录结构）

```
main.py            CLI entry point (setup / annotate / review / run / analyze)
annotation/        annotation pipeline: story parsing, LLM annotation, validation, human review
models/            model layer: one interface with HuggingFace / vLLM / Mock backends, plus a factory
experiment/        experiment engine: prompt building, step-by-step running, recording, scheduling
analysis/          metrics: accuracy, emergence, confidence, embeddings, stability, statistics, LLM judge
config/            configuration: three YAML files and the config manager
data_manager/      data access with atomic writes
utils/             logging, seeding, environment self-check
visualization/     plotting helpers
scripts/           helper scripts used during the project (packing, batch runs, evaluation, tables, plots)
tests/             unit tests
results/           experiment outputs: raw responses, metrics, embeddings, judge caches, logs
results_condB/     outputs of Condition B (same structure)
```

Role of each module:

- annotation/: turns raw story text into structured JSON (per-sentence function, critical sentence, gold answer, question) and runs the LLM annotation with the model named by `annotation_model`; `validator.py` checks structural validity and `human_review.py` provides the interactive review.
- models/: every backend implements the same interface (load / generate / unload). `ModelFactory` reads `models.yaml` and decides which backend to instantiate and how to quantize; `MockModel` is a deterministic stub used by the tests and smoke runs.
- experiment/: `PromptBuilder` assembles the per-step input from `prompts.yaml`; `IncrementalRunner` maintains the growing context and calls the model step by step; `ResultRecorder` persists every raw response; `ExperimentScheduler` creates and executes the "model × task × story × repetition" job matrix.
- analysis/: `accuracy` / `emergence` / `confidence` / `embedding` / `stability` / `statistics` compute the individual metrics; `judge.py` implements the LLM equivalence judge, an alternative (or complement) to cosine similarity.
- config/: `ConfigManager` loads and validates the three YAML files; it is the only entry point through which the code reads configuration.
- utils/: unified logging (to `logs/iple.log`), seeding (same seed, same output), and the environment self-check used by `python main.py setup`.
- data_manager/: centralises the read/write paths under `data/` and `results/`; writes go through a temporary file and a replace step, so an interrupted run cannot corrupt existing files.
- scripts/: helper scripts accumulated during the project. Rough groups: cloud packaging and batch execution (`pack_for_cloud.py`, `cloud_run.py`, `run_full_class_batch.py`), model and pipeline verification (`verify_models.py`, `smoke_test.py`), judge tooling (`judge_sample.py`, `judge_analyze.py`, `run_judge_on_pool.py`, `assess_judges.py`), statistics and report tables (`mixed_effects.py`, `confidence_analysis.py`, `build_report_tables_v2.py`, `calibrate_cosine_threshold.py` and more), plotting (`plot_*.py`), and report draft synchronisation (`sync_report_draft*.py`). They were written for the internal workflow; when used standalone, adjust their input/output paths to your own layout.
- tests/: see "Tests（测试）".

## Requirements and Installation（运行环境与依赖）

Python 3.10 or newer is required (`main.py setup` checks the version). The dependencies are listed in `requirements.txt`:

| Package | Purpose |
|---|---|
| torch / transformers / accelerate / bitsandbytes | local HuggingFace inference and 4-bit/8-bit quantization |
| vllm | inference backend for the large model tiers (7B/14B/32B) on GPU |
| sentence-transformers | the default embedding model all-MiniLM-L6-v2, used by the cosine-based accuracy |
| pandas / statsmodels | metric aggregation and mixed-effects statistics |
| pyyaml / typer / tqdm / pytest | configuration, CLI, progress bars, testing |

Installation and self-check:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # on Linux / macOS use: source .venv/bin/activate
pip install -r requirements.txt
python main.py setup
```

`setup` prints one [OK] / [WARN] line per check (Python version, dependencies, GPU availability; a machine without CUDA gets a warning and falls back to CPU), creates the `data/raw_stories`, `data/annotated`, `results/raw`, `results/processed`, `results/embeddings` and `logs` directories, and then loads the configuration.

Two notes. `vllm` is only needed to run the large model tiers on a GPU, and it pins the local torch version tightly; on a machine without a GPU you can delete that line before installing and nothing else is affected. Model weights are downloaded from HuggingFace on first use and cached locally afterwards.

## Configuration（配置说明）

The three YAML files under `config/` are the single source of all experiment parameters; there are no scattered magic numbers in the code.

### models.yaml

The single source of truth for models: "adding a model" means adding one entry here, with no changes to the experiment code. Fields per entry:

| Field | Meaning |
|---|---|
| family / path | model family and HuggingFace repository id |
| backend | `huggingface` (local) or `vllm` (high-throughput GPU) |
| quantization | none / 4bit / 8bit / awq / gptq / fp8; vLLM requires pre-quantized weights |
| dtype | bfloat16 / float16 (AWQ supports float16 only) |
| size / training | used for statistics (model size, instruction / reasoning) |
| gpu_memory_utilization / max_model_len / max_num_seqs / enable_prefix_caching | vLLM knobs; lower the first three when GPU memory runs short |

Predefined entries: `qwen05b`, `qwen15b` (local small models; 1.5B runs with 4-bit quantization), `qwen7b`, `qwen14b`, `qwen32b`, `deepseek7b`, `deepseek14b`, `deepseek32b` (vLLM + AWQ), plus the comparison entries `qwen14b_bf16`, `deepseek14b_bf16`, `qwen05b_vllm`, `qwen15b_vllm`.

### experiment.yaml

- `experiment`: `seed_master`, `temperature`, `max_tokens`, `repetitions`, and `annotation_model` (the model key used for annotation; default `qwen7b`).
- `scoring`: `method` (`embedding_similarity` / `local_judge` / `api_judge`), `judge_provider` (`local` / `deepseek` / `kimi`), `threshold` (cosine threshold, embedding mode only), and `embedding_model` (default `sentence-transformers/all-MiniLM-L6-v2`).

### prompts.yaml

- `prompt_version`: the version string recorded with every run.
- `conditions`: system prompts and opening templates for `condition_a` / `condition_b` (the latter adds the "analyse carefully why your interpretation changes" instruction).
- `tasks`: answer templates for `default` / `false_belief` / `faux_pas` / `implicature`; all require the confidence line to be the last line of the reply.
- `annotation`: the structured prompt used for annotation.

### .env (not distributed with the repository)

The `api_judge` mode needs an API key, read from a `.env` file in the project root (the file is git-ignored; never commit it):

```
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com    # optional, defaults shown
DEEPSEEK_MODEL=deepseek-chat                   # optional, defaults shown
KIMI_API_KEY=your-key                          # when using the kimi provider
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=moonshot-v1-8k
```

A missing key or a failed request raises an error immediately; nothing degrades silently.

## Command-Line Usage（命令行用法）

| Command | Description |
|---|---|
| `python main.py setup` | environment self-check and directory creation |
| `python main.py annotate [--input DIR] [--task T] [--mock]` | run the annotation pipeline on raw stories, output to `data/annotated/*.json`; the task defaults to the input directory name |
| `python main.py review <story_id>` | interactive human review (sentence functions, gold answer, critical sentence); marks the story as reviewed |
| `python main.py run --model KEY [--task T] [--stories LIST] [--repetitions N] [--seed N] [--mock]` | run the incremental experiment; only reviewed stories are processed; results go to `results/raw/<model>/<task>/<story>/runN.json` |
| `python main.py analyze [--scoring-method M] [--judge-provider P] [--results-dir DIR]` | compute the metrics; CSVs are written to `results/processed/` |

Typical invocations:

```powershell
# annotate every raw story of the faux_pas task
python main.py annotate --input data/raw_stories/ --task faux_pas

# review one annotation
python main.py review fp001

# run the incremental experiment (3 repetitions by default from experiment.yaml; --repetitions overrides)
python main.py run --model qwen7b --task faux_pas --stories fp001,fp002

# default cosine-based analysis
python main.py analyze

# paper configuration: DeepSeek API judge (needs the keys in .env)
python main.py analyze --scoring-method api_judge --judge-provider deepseek

# Condition B results directory
python main.py analyze --results-dir results_condB
```

`--mock` is a test facility that avoids loading real models: both `annotate` and `run` can go through the full pipeline with the MockModel, which is handy for smoke tests on machines without a GPU or model weights.

## Data and Materials（数据与材料）

This repository keeps the experiment outputs but not the source story materials. The reasons are the same as during the project: the original materials (the Faux Pas items for example) are for internal research use only and must not be redistributed, while the outputs here were produced by our own program running on the AutoDL GPU server.

Tracked in this repository:

- `results/raw/` — 6,770 raw response files of Condition A (one file per story × repetition, each holding the per-step records);
- `results/processed/`, `results/processed_judge/`, `results/processed_cosine_4200/`, `results/processed_judge_official/`, `results/processed_judge_condB/` — metric CSVs, figures and judge caches;
- `results/embeddings/` — 44,064 trajectory embedding files;
- `results_condB/` — the same kinds of outputs for Condition B (2,520 raw files, 15,831 embeddings, processed metrics);
- `results/logs/` — the four statistics logs (`logs_confidence.txt`, `logs_confidence_condB.txt`, `logs_mixed_effects.txt`, `logs_stability_position.txt`).

Not tracked, and why:

- `data/` and `datasets/` — the original story materials and dataset copies; internal research use only, not redistributed;
- the bulk archives below (they also contain copies of the annotated data);
- local run logs and caches.

The complete archive is `cloud_full_final.tar.gz` (100,254,135 bytes, 72,719 entries): config, data (annotated 280 + manual_equivalence 4), results (raw 6770, processed, embeddings 44,064) and results_condB (raw 2520, embeddings 15,831). It is delivered outside git:

- Download: (to be filled in — cloud drive or GitHub Release link)
- Verification (run in the directory containing the archive):

```powershell
certutil -hashfile cloud_full_final.tar.gz MD5
# expected: adb134917178008bfb2cb9d996724eb6
certutil -hashfile cloud_full_final.tar.gz SHA256
# expected: 82c72d837307d06e334fc252fda4bd62c2b8a56176fade2092aa15711d01a2b5
```

- Unpacking: `tar -xzf cloud_full_final.tar.gz -C <target>`; the top level contains `config/`, `data/`, `results/` and `results_condB/`.

The archive contains material-derived data and is for internal research use; do not publish or redistribute it. For the same reason the packing snapshot `cloud_backup_final.zip` (118 MB) stays off git. A small cloud work package, `HSP_cloud.zip` (code + annotated data for batch runs on a GPU machine), can be regenerated at any time:

```powershell
python -X utf8 scripts/pack_for_cloud.py
```

Upload the zip to the server, unpack it there and run `scripts/cloud_run.py` (it supports a small preflight mode and a full run).

For reference, the experiment scale: 280 stories × 10 model tiers × 3 repetitions; 9,290 official runs (6,770 for Condition A, 2,520 for Condition B), with 9,290/9,290 judge decisions.

## Reproducing the Results（结果复现）

Reproduction needs two things: the Python environment and the data. Because the source materials are restricted, a full end-to-end reproduction starts from an archive copy, or from stories you are allowed to use.

From raw stories (the annotation path):

1. install the dependencies and run `python main.py setup`;
2. put the raw stories under `data/raw_stories/<task>/`;
3. `python main.py annotate --input data/raw_stories/ --task <task>`;
4. `python main.py review <story_id>` for every story (unreviewed stories never enter the experiment);
5. `python main.py run --model <key> [--task/--stories/--repetitions]`;
6. `python main.py analyze`, adding `--scoring-method api_judge --judge-provider deepseek` for the paper configuration (bring your own key).

From the archive (skipping annotation):

1. unpack the archive and place its `data/annotated/` into the project's `data/annotated/`;
2. still start with `setup`, then go straight to `run` / `analyze`;
3. on a machine with 4 GB of GPU memory you can run the `qwen05b` and `qwen15b` tiers; the large tiers need a GPU server with vLLM.

## Tests（测试）

Run from the project root:

```powershell
python -m pytest tests/ -q
```

The six test files cover: metric computation and embedding caching (test_analysis), story parsing and annotation format (test_annotation), the experiment engine and scheduler (test_experiment), judge output parsing and missing-key errors (test_judge), model factory dispatch (test_model_factory), and same-seed reproducibility plus the "add a model without touching code" extension path (test_reproducibility).

The tests exercise the main pipeline with the MockModel and the hash-based embedder (HashEmbedder), so they load no real model weights and need no GPU; a plain CPU environment is enough.

## Copyright, Citation and Licensing（版权、引用与许可）

- The Faux Pas materials (the original PDF and the story texts) are for internal research use only and must not be redistributed. This repository does not include those source materials. The experiment outputs under `results/` and `results_condB/` were produced by our own runs and are kept here as a record of the research results; note that some records (for example the `context` field in `results/raw/`) quote the story texts.
- Third-party dataset materials (OpenToM, SwordsmanImp and others) are not part of this repository; their use and redistribution follow the licence of each dataset.
- This repository ships without a LICENSE file. The code was written for a course research project; if you want to reuse it elsewhere, please contact the author through the GitHub page of this repository.
- When citing this project, please refer to the repository URL and the author; the citation information of the report and paper follows the official project documents.
