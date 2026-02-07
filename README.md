

```markdown
# AITW Evaluation Script for ST-Lite

This repository contains the evaluation script for benchmarking **ST-Lite** (and other UI-TARS variants) on the **Android in the Wild (AITW)** dataset. It supports efficient KV cache compression methods including `st_lite`, `snap_kv`, and `pyramid_kv`.

## 📂 Directory Structure Requirements

Ensure your project directory contains the following helper modules alongside `eval_aitw.py`:

```text
.
├── eval_aitw.py          # Main evaluation script
├── attention_helpers.py  # Attention masking & KV cache logic
├── ui_tars_utils.py      # UI-TARS specific utilities
├── opencua_utils.py      # OpenCUA specific utilities
└── action_matching.py      # Action matching logic directory

```

## 🛠️ Dependencies

Install the required Python packages. It is recommended to use the `requirements.txt` generated for your specific environment (especially for `transformers` dev versions).

```bash
pip install -r requirements.txt

```

## 📊 Data Preparation

You need the AITW dataset (Google Research) prepared in the following format:

1. **Images Directory**: A folder containing all screenshots (referenced in the JSON).
2. **Test JSON**: A JSON file containing the test episodes, goals, and history.

## 🚀 Usage

### Basic Evaluation

Run the evaluation using the following command:

```bash
python eval_aitw.py \
    --model_path /path/to/your/UI-TARS-Model \
    --aitw_imgs /path/to/aitw_images_dir \
    --aitw_test /path/to/aitw_test.json \
    --kv_cache st_lite \
    --kv_cache_budget 20 40 \
    --results_dir ./results/aitw/

```

### Debugging

To test the pipeline with a small number of samples (e.g., first 10 episodes), use the `--debug` flag:

```bash
python eval_aitw.py ... --debug 10

```

## ⚙️ Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--model_path` | `str` | *Required* | Path to the pretrained model (UI-TARS / Qwen-VL). |
| `--aitw_imgs` | `str` | *Required* | Directory containing AITW screenshots. |
| `--aitw_test` | `str` | *Required* | Path to the AITW test set JSON file. |
| `--kv_cache` | `str` | `st_lite` | KV compression method. Choices: `full_cache`, `st_lite`, `snap_kv`, `pyramid_kv`, `vl_cache`. |
| `--kv_cache_budget` | `int` | `[40, 80]` | Budget for KV cache compression. |
| `--window_size` | `int` | `8` | Window size for attention mechanisms. |
| `--his_num` | `int` | `16` | Number of history steps/images to include in the context. |
| `--max_new_tokens` | `int` | `128` | Maximum new tokens to generate for the action. |
| `--device` | `str` | `auto` | Device to run on (`cuda`, `mps`, `cpu`). |
| `--results_dir` | `str` | `./results/` | Directory to save evaluation metrics and logs. |

## 📈 Output

Results are saved in the `results_dir` organized by the budget used.

* **`detailed_results.json`**: Contains step-by-step logs, model thoughts, predicted actions, and correctness checks.
* **`metrics.json`**: Summary of accuracy metrics:
* **Overall Acc**: Global action matching accuracy.
* **Task Acc**: Accuracy per task category (e.g., GoogleApps, WebShopping).
* **Action Type Acc**: Specific accuracy for Click, Scroll, and Type actions.



## ⚠️ Notes

* **JAX/TPU Warnings**: The script automatically suppresses JAX warnings (`JAX_PLATFORMS=cpu`) used by the action matching library.
* **Flash Attention**: Ensure your environment supports `flash_attention_2` for optimal performance.

```

```