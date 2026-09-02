# ST-Lite

> **ST-Lite: Training-Free KV Cache Compression with Spatio-Trajectory Guidance for Long-Horizon GUI Agents**

<p align="center">
  <a href="https://2026.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP-2026-b5179e.svg?style=flat-square" alt="EMNLP 2026" /></a>
  <a href="https://github.com/94wen94/ST-Lite"><img src="https://img.shields.io/badge/Code-ST--Lite-1f6feb.svg?style=flat-square" alt="Code" /></a>
  <a href="https://github.com/google-research/google-research/tree/master/android_in_the_wild"><img src="https://img.shields.io/badge/Benchmark-AITW-ff9800.svg?style=flat-square" alt="Android in the Wild" /></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab.svg?style=flat-square" alt="Python 3.10+" />
</p>

**ST-Lite is a training-free KV cache compression method for long-horizon GUI agents.** It combines spatial token importance with cross-frame trajectory redundancy to retain useful visual context while removing repeated information from earlier screenshots.

This repository provides the ST-Lite implementation and an evaluation pipeline for [Android in the Wild (AITW)](https://github.com/google-research/google-research/tree/master/android_in_the_wild), with support for UI-TARS- and OpenCUA-style models.

## News

- **2026** — ST-Lite at EMNLP 2026.
- **2026** — Initial code and AITW evaluation pipeline released.

## TL;DR

- **Training-free:** no additional model training or fine-tuning is required.
- **Spatial guidance:** prioritizes informative regions within the current screenshot.
- **Trajectory guidance:** detects visually redundant tokens across historical screenshots.
- **Plug-and-play evaluation:** compares ST-Lite with full cache, PyramidKV, SnapKV, and VL-Cache under the same AITW pipeline.
- **Multi-budget evaluation:** evaluates several KV cache budgets in one run and writes per-budget metrics.

## How it works

```text
Historical screenshots + current screenshot
                    │
                    ├── Spatial importance in the current frame
                    ├── Cross-frame similarity for trajectory redundancy
                    └── Recent-token window + attention importance
                                      │
                                      ▼
                           Compressed KV cache
```

ST-Lite keeps a recent context window, scores older tokens using attention and visual importance, and suppresses highly similar tokens from previous frames. The resulting cache preserves task-relevant spatial and temporal evidence within a configurable budget.

## Repository structure

```text
ST-Lite/
├── eval/
│   ├── aitw_eval.py          # AITW evaluation entry point
│   ├── attention_replace.py  # Attention patches and cache configuration
│   ├── action_matching.py    # AITW action matching
│   ├── ui_tars_utils.py      # UI-TARS preprocessing and parsing
│   └── opencua_utils.py      # OpenCUA preprocessing and parsing
├── utils/
│   └── methods.py            # ST-Lite and baseline KV-cache methods
├── requirements.txt
└── README.md
```

## Installation

We recommend Linux, an NVIDIA GPU, CUDA, and Python 3.10 or newer.

```bash
git clone https://github.com/94wen94/ST-Lite.git
cd ST-Lite

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The current implementation relies on the Qwen2.5-VL interfaces from the Transformers development version noted in [`requirements.txt`](./requirements.txt). Install a compatible Transformers build before running evaluation if it is not already present in your environment.

## Data preparation

Download and prepare the AITW test set with:

1. an image directory containing the screenshots referenced by the annotations; and
2. a test JSON file containing tasks, episodes, goals, actions, and screenshot filenames.

The evaluator expects each screenshot at:

```text
<aitw_imgs>/<img_filename>.png
```

See the official [AITW repository](https://github.com/google-research/google-research/tree/master/android_in_the_wild) for dataset access and annotation details.

## Evaluation

### ST-Lite

```bash
python eval/aitw_eval.py \
  --model_path /path/to/UI-TARS-1.5-7B \
  --aitw_imgs /path/to/aitw_images \
  --aitw_test /path/to/aitw_test.json \
  --kv_cache st_lite \
  --kv_cache_budget 20 40 \
  --his_num 16 \
  --results_dir ./results/aitw
```

`--kv_cache_budget 20 40` runs two evaluations, retaining the configured percentage of the prompt KV cache for each run.

### Baselines

Use the same command and change `--kv_cache` to one of:

```text
full_cache | pyramid_kv | snap_kv | vl_cache | st_lite
```

For a quick pipeline check, limit the number of episodes per task:

```bash
python eval/aitw_eval.py ... --debug 10
```

## Main arguments

| Argument | Default | Description |
|---|---:|---|
| `--model_path` | `/path/to/your/UI-TARS-1.5-7B` | Local path or model identifier. Paths containing `OpenCUA` use the OpenCUA branch; other supported runs use UI-TARS/Qwen2.5-VL. |
| `--aitw_imgs` | placeholder | Directory containing AITW screenshots. |
| `--aitw_test` | placeholder | AITW test annotation JSON. |
| `--his_num` | `16` | Maximum number of historical screenshot-action pairs. |
| `--kv_cache` | `st_lite` | Cache mode to evaluate. |
| `--kv_cache_budget` | `40 80` | One or more cache budgets, expressed as percentages. |
| `--window_size` | `8` | Recent-token attention window. |
| `--max_new_tokens` | `128` | Maximum generated action tokens. |
| `--model_dtype` | `bfloat16` | Model dtype: `auto`, `bfloat16`, `float16`, or `float32`. |
| `--attention_implementation` | `flash_attention_2` | Transformers attention backend. |
| `--device` | auto-detected | Explicit device override. |
| `--debug` | disabled | Maximum episodes per task for a smoke test. |
| `--results_dir` | `./results/aitw/` | Output root directory. |

## Output

```text
results/aitw/
├── budget_20/
│   ├── detailed_results.json
│   └── metrics.json
├── budget_40/
│   ├── detailed_results.json
│   └── metrics.json
└── multi_budget_summary.json
```

- `detailed_results.json` stores step-level task, episode, budget, and correctness records.
- `metrics.json` stores overall and per-task action-matching metrics.
- `multi_budget_summary.json` compares aggregate metrics across cache budgets.

## Citation

The paper citation and public paper link will be added with the paper release.

## Acknowledgements

This code builds on the AITW evaluation protocol and ideas or implementations from UI-TARS, OpenCUA, PyramidKV, SnapKV, VL-Cache, and AdaKV. We thank the authors of these projects for making their work available.
