# A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search

[**Project Page**](https://baekseongeun.github.io/lora-bo/) &nbsp;|&nbsp; [**arXiv**](https://arxiv.org/abs/2602.11171)

Bayesian-optimization driven hyperparameter search for LoRA fine-tuning.
A Gaussian-process surrogate (`gollum`) proposes LoRA hyperparameters, which are
trained and evaluated through the `PiSSA` training/evaluation pipeline, and the
resulting accuracy is fed back to the surrogate.

Two axes are configurable at run time:

- `--task {math, code, conversation}` selects which training script (and dataset)
  is used.
- `--mode {frozen, learnable}` selects how the surrogate featurizes the
  hyperparameter templates:
  - `frozen` — `get_huggingface_embeddings` + `ProjectionLayer` (frozen embeddings)
  - `learnable` — `get_tokens` + `LLMFeaturizer` (trainable LLM, i.e. learnable tokens)

## Requirements

Two conda environments are used. `run_hpo.py` runs in the first (GP/BO); it
launches training in the second via `conda run -n pissa bash <script>`.

### 1. gollum environment (runs run_hpo.py)

```bash
conda create -n gollum python=3.10 -y
conda activate gollum
pip install -e gollum
```

### 2. pissa environment (training + vLLM evaluation)

```bash
conda create -n pissa python=3.10 -y
conda activate pissa
pip install -r PiSSA/requirements.txt
pip install flash-attn --no-build-isolation
```

If the conda env is not named `pissa`, set `PISSA_ENV=<name>` when running.

### Access

- Base model: `meta-llama/Llama-2-7b-hf` is gated. Accept its license on
  Hugging Face and log in first: `huggingface-cli login`.

## Data

`PiSSA/pissa-dataset/` ships with the evaluation split (`<task>/test.json`) only.
The large training splits `<task>/train.json` are **not** included. Download them
from the original PiSSA dataset (`fxmeng/pissa-dataset` on Hugging Face) and place
each `train.json` into the matching folder:

```
PiSSA/pissa-dataset/metamath/train.json
PiSSA/pissa-dataset/python/train.json
PiSSA/pissa-dataset/conversation/train.json
```

The BO input tables under `data/*.csv` are **not** committed to the repo —
generate them with `make_templates.py` before a frozen/learnable run (see
**Template Generation**). The `--dry_run` sample uses its own small table in
`examples/` and needs no generation.

How much of `train.json` to train on is set in each `scripts/<task>/run_lora.sh`
via `--sub_task <task>:<N>` (e.g. `metamath:10000` trains on 10k samples), or pass
`--train_samples <N>` to `run_hpo.py` to override it without editing the script:

```bash
python run_hpo.py --task math --train_samples 10000 --seed 44
```

## Template Generation

The `data/*.csv` tables (the hyperparameter search space + `template` text) are
produced by `make_templates.py` and are **not** committed to the repo (they are
large and fully reproducible). Generate them once before a frozen/learnable run —
both modes read them:

```bash
python make_templates.py --mode frozen       # -> data/hyperparameters.csv
python make_templates.py --mode learnable    # -> data/hyperparameters_learnable_token.csv
# or a custom path:
python make_templates.py --mode learnable --output data/my_table.csv
```

Edit the search-space lists at the top of `make_templates.py` (`RANKS`,
`ALPHA_MULTIPLIERS`, `DROPOUTS`, `BATCH_SIZES`, `LEARNING_RATES`) to change the
space. `--mode learnable` appends `<TT>` to every template (see **Learnable
token**); `--mode frozen` does not. Accuracy columns are written unmeasured
(`-1`); the BO loop fills them in.

## Quick sample (no GPU training required)

A small offline example runs the full BO loop without launching any LoRA
training. With `--dry_run`, the objective is looked up from a template table of
already-measured configurations (`examples/sample_hyperparameters.csv`, 57 real
math results) instead of training a model. This needs neither the gated base
model, vLLM, nor `train.json` — only the `gollum` environment (a one-time
`t5-base` download for template embeddings; a small GPU is used if available,
otherwise CPU).

```bash
conda activate gollum
bash examples/run_sample.sh
# or:
python run_hpo.py --config examples/config_sample.yaml --task math --dry_run \
    --seed 44 --n_iters 5 --group sample
```

Each iteration prints the suggested configuration and its looked-up accuracy.
Drop `--dry_run` (and use a full config) to switch to real training.

## Learnable token (`<TT>`)

In `learnable` mode the trainable parameter is the embedding of a single special
token, `<TT>`. For it to be exercised, that token **must be present in the
template text** — the learnable-mode table appends `<TT>` at the end of every
`template`, the frozen-mode table does not.

Concretely: `LLMFeaturizer` registers `<TT>` as a new special token and makes only
its embedding trainable (`peft.TrainableTokensConfig`). The featurizer
`get_tokens` registers the same `<TT>` special token, so a template containing
`<TT>` is tokenized to that exact trainable id (everything else stays frozen).
Without `<TT>` in the template, the trainable token id never appears in the model
input and no learnable token is trained. If you build your own learnable-mode
table, keep the trailing `<TT>`.

## Running

Run from the repo root, in the `gollum` environment:

```bash
conda activate gollum

# frozen embedding surrogate (default mode)
python run_hpo.py --task math --seed 44 --n_iters 50 --group my_run

# learnable-token surrogate
python run_hpo.py --mode learnable --task math --seed 44 --n_iters 50 --group my_run

# other tasks
python run_hpo.py --task code --seed 44
python run_hpo.py --task conversation --seed 44
```

`--mode` picks the config (`configs/…`) and therefore the surrogate + template
table (`frozen` → `data/hyperparameters.csv`, `learnable` →
`data/hyperparameters_learnable_token.csv`). `--task` picks
`PiSSA/scripts/<task>/run_lora.sh`. On each BO iteration the entry script
overwrites the hyperparameter block of that training script (`RANK`, `LORA_ALPHA`,
`LR`, `DROPOUT`, `per_device_train_batch_size`, `gradient_accumulation_steps`,
`TOTAL_BATCH_SIZE`), runs it, and reads back the objective.

## Citation

```bibtex
@inproceedings{seong2026language,
  title={A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search},
  author={Seong-Eun, Baek and Jung-Mok, Lee and Sung-Bin, Kim and Oh, Tae-Hyun},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```
