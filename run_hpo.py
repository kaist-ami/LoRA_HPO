import warnings
from botorch.exceptions import InputDataWarning
import os
import re
import torch
import logging
import json
import gc
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore", category=InputDataWarning)
logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
warnings.filterwarnings(
    "ignore",
    message="ExpectedImprovement has known numerical issues that lead to suboptimal optimization performance",
)

class IgnoreDeviceFilter(logging.Filter):
    def filter(self, record):
        return "available:" not in record.getMessage()

logger.addFilter(IgnoreDeviceFilter())

warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
    module="pytorch_lightning.trainer.connectors.data_connector",
)

from gollum.test_acc import extract_answer_number, process_math_results
from gollum.data.module import BaseDataModule
from gollum.bo.optimizer import BotorchOptimizer
from gollum.metrics import (
    calculate_data_stats,
    log_bo_metrics,
    log_data_stats,
)

torch.set_float32_matmul_precision("high")

from transformers import Qwen2_5_VLConfig
if not hasattr(Qwen2_5_VLConfig, "vision_start_token_id"):
    Qwen2_5_VLConfig.vision_start_token_id = 151652

from pytorch_lightning import seed_everything
import wandb
from tqdm import tqdm
from gollum.utils.config import flatten

from jsonargparse import ArgumentParser
from gollum.utils.config import instantiate_class
from botorch.acquisition import AcquisitionFunction
from gollum.surrogate_models.gp import SurrogateModel

HERE = Path(__file__).resolve().parent
PISSA_DIR = HERE / "PiSSA"
RESULTS_DIR = HERE / "results"
CONFIG_DIR = HERE / "configs"
MODE_CONFIG = {
    "frozen": CONFIG_DIR / "pllm_qwen_llm.yaml",
    "learnable": CONFIG_DIR / "pllm_qwen_llm_learnable_token.yaml",
}
TRAIN_ENV = os.environ.get("PISSA_ENV", "pissa")
NUM_GPUS = int(os.environ.get("PISSA_NUM_GPUS", "2"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_EMBEDDING_SIZES = {
    "WhereIsAI/UAE-Large-V1": 1024,
    "nomic-ai/modernbert-embed-base": 768,
    "Qwen/Qwen2-7B-Instruct": 3584,
    "Qwen/Qwen2.5-7B-Instruct": 3584,
    "Qwen/Qwen2-1.5B-Instruct": 1536,
    "t5-base": 768,
    "mistralai/Mistral-7B-Instruct-v0.2": 4096,
    "text-embedding-3-large": 3072,
    "nomic-ai/modernbert-embed-base;get_huggingface_embeddings;normalize:False;pooling:cls": 768,
    "Qwen/Qwen2-7B-Instruct;get_huggingface_embeddings;normalize:False;pooling:last_token": 3584,
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": 768,
}

TASK_CONFIG = {
    "math": {
        "script": PISSA_DIR / "scripts" / "math" / "run_lora.sh",
        "output_root": "output/math",
        "response": "metamath_response.jsonl",
    },
    "code": {
        "script": PISSA_DIR / "scripts" / "code" / "run_lora.sh",
        "output_root": "output/code",
        "response": "python_response.jsonl",
    },
    "conversation": {
        "script": PISSA_DIR / "scripts" / "conversation" / "run_lora.sh",
        "output_root": "output/conversation",
        "response": "conversation_response.jsonl",
    },
}

def parse_hparams(s):
    params = {}
    for kv in s.split(','):
        key, val = kv.strip().split('=', 1)
        key = key.strip()
        val = val.strip()
        if key in ('rank', 'alpha', 'batch_size'):
            params[key] = str(int(float(val)))
        else:
            params[key] = str(float(val))
    return (
        params['rank'],
        params['alpha'],
        params['dropout'],
        params['batch_size'],
        params['learning_rate'],
    )

def patch_run_train(script_path, LORA_R, LORA_ALPHA, LORA_DROPOUT, NEW_LR, NEW_BATCH,
                    per_device_train_batch_size, gradient_accumulation_steps,
                    train_samples=None):
    subs = {
        "RANK": LORA_R,
        "LORA_ALPHA": LORA_ALPHA,
        "per_device_train_batch_size": str(per_device_train_batch_size),
        "gradient_accumulation_steps": str(gradient_accumulation_steps),
        "LR": NEW_LR,
        "DROPOUT": LORA_DROPOUT,
        "TOTAL_BATCH_SIZE": NEW_BATCH,
    }
    content = script_path.read_text()
    for var, val in subs.items():
        content = re.sub(rf"^(\s*{re.escape(var)}=).*", rf"\g<1>{val}", content, flags=re.MULTILINE)
    if train_samples is not None:
        content = re.sub(r"(--sub_task\s+\S+?):\d+", rf"\g<1>:{train_samples}", content)
    script_path.write_text(content)

def parse_math_acc(response_path):
    results = defaultdict(list)
    with open(response_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['type'] == 'gsm8k':
                y_pred = extract_answer_number(data['output'])
                results['gsm8k'].append(
                    float(y_pred) == float(data["answer"]) if y_pred is not None else False
                )
            elif data['type'] == 'math':
                results['math'].append(process_math_results(data['output'], data['answer']))

    gsm8k_acc = sum(results['gsm8k']) / len(results['gsm8k'])
    math_acc = sum(results['math']) / len(results['math'])
    print(f"GSM8K Accuracy : {gsm8k_acc:.3f}")
    print(f"MATH  Accuracy : {math_acc:.3f}")
    return (gsm8k_acc + math_acc) * 100

def _extract_pass_at_1(data, dataset):
    for container in (data, data.get(dataset, {}), data.get("pass_at_k", {}), data.get("eval", {})):
        if isinstance(container, dict) and "pass@1" in container:
            return float(container["pass@1"])
    raise KeyError(f"pass@1 not found in evalplus result for {dataset}")

def parse_code_acc(output_dir):
    scores = []
    for dataset in ("humaneval", "mbpp"):
        result_path = Path(output_dir) / f"{dataset}_eval_results.json"
        with open(result_path, 'r') as f:
            data = json.load(f)
        scores.append(_extract_pass_at_1(data, dataset))
    return sum(scores) / len(scores) * 100

def parse_conversation_acc(output_dir):
    score_path = Path(output_dir) / "conversation_score.json"
    if not score_path.exists():
        raise FileNotFoundError(
            f"{score_path} not found. Provide a judge score file for the conversation task."
        )
    with open(score_path, 'r') as f:
        data = json.load(f)
    return float(data["objective"])

def evaluate_task(task, output_dir):
    response_path = Path(output_dir) / TASK_CONFIG[task]["response"]
    if task == "math":
        return parse_math_acc(response_path)
    if task == "code":
        return parse_code_acc(output_dir)
    if task == "conversation":
        return parse_conversation_acc(output_dir)
    raise ValueError(f"Unknown task: {task}")

def lora_tuning(task, LORA_R, LORA_ALPHA, LORA_DROPOUT, NEW_LR, NEW_BATCH, train_samples=None):
    task_cfg = TASK_CONFIG[task]
    script_path = task_cfg["script"]

    per_device_train_batch_size = 4
    gradient_accumulation_steps = int(NEW_BATCH) // (per_device_train_batch_size * NUM_GPUS)
    if gradient_accumulation_steps < 1:
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 1

    output_name = (
        f"rank-{LORA_R}-alpha-{LORA_ALPHA}-batch-{NEW_BATCH}"
        f"-{per_device_train_batch_size}-{gradient_accumulation_steps}"
        f"-LR-{NEW_LR}-dropout-{LORA_DROPOUT}"
    )

    patch_run_train(
        script_path, LORA_R, LORA_ALPHA, LORA_DROPOUT, NEW_LR, NEW_BATCH,
        per_device_train_batch_size, gradient_accumulation_steps,
        train_samples=train_samples,
    )

    torch.cuda.empty_cache(); gc.collect()

    train_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}

    print("conda run --no-capture-output -n", TRAIN_ENV, "bash", str(script_path))
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", TRAIN_ENV, "bash", str(script_path)],
        check=True, cwd=str(PISSA_DIR), env=train_env,
    )

    torch.cuda.empty_cache(); gc.collect()

    output_dir = PISSA_DIR / task_cfg["output_root"] / output_name
    return evaluate_task(task, output_dir)

def configure_benchmark_datasets(config):
    benchmark = config["benchmark"]
    if benchmark.startswith("bh"):
        reaction_num = benchmark[-1]
        config["data"]["init_args"]["data_path"] = (
            f"data/reactions/buchwald-hartwig/bh_reaction_{reaction_num}_procedure_template_basic.csv"
        )
        config["data"]["init_args"]["target_column"] = "objective"
        config["data"]["init_args"]["maximize"] = True
    return config

def validate_configuration(config):
    surrogate_class = config["surrogate_model"]["class_path"]
    featurizer_config = config["data"]["init_args"]["featurizer"]["init_args"]
    representation = featurizer_config.get("representation")

    if surrogate_class == "gollum.surrogate_models.gp.GP" and representation == "get_tokens":
        raise ValueError("Standard GP or PLLM shouldn't use 'get_tokens'. This is for trainable LLM models only.")

    model_name = featurizer_config.get("model_name")
    if model_name in MODEL_EMBEDDING_SIZES:
        embedding_size = MODEL_EMBEDDING_SIZES[model_name]
        if "surrogate_model" in config and "init_args" in config["surrogate_model"]:
            if "finetuning_model" in config["surrogate_model"]["init_args"]:
                current_dim = config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"].get("input_dim")
                if current_dim != embedding_size:
                    print(f"Updating input_dim from {current_dim} to {embedding_size} for {model_name}")
                    config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]["input_dim"] = embedding_size

    return config

def resolve_data_path(config):
    data_path = config["data"]["init_args"]["data_path"]
    if not os.path.isabs(data_path):
        config["data"]["init_args"]["data_path"] = str(HERE / data_path)
    return config

def setup_data(config):
    initializer = instantiate_class(
        config["data"]["init_args"]["initializer"], seed=config["seed"]
    )
    featurizer = instantiate_class(config["data"]["init_args"]["featurizer"])
    dm = instantiate_class(
        config["data"],
        initializer=initializer,
        featurizer=featurizer,
        normalize_input=config["data"]["init_args"]["normalize_input"],
        maximize=config["data"]["init_args"]["maximize"],
    )
    return dm

def setup_bo_optimizer(config, design_space):
    bo_config = config["bo"]["init_args"]
    surrogate_model_config = config["surrogate_model"]
    acquisition_config = config["acquisition"]
    return BotorchOptimizer(
        design_space=design_space,
        surrogate_model_config=surrogate_model_config,
        acq_function_config=acquisition_config,
        batch_strategy=bo_config["batch_strategy"],
        batch_size=bo_config["batch_size"],
    )

def train(config):
    task = config["task"]
    dry_run = config.get("dry_run", False)
    train_samples = config.get("train_samples", None)
    target_column = config["data"]["init_args"]["target_column"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_setting = []
    acc_list = []

    if config.get("benchmark", None) is not None:
        config = configure_benchmark_datasets(config)

    config = resolve_data_path(config)
    config = validate_configuration(config)
    wandb_config = flatten(config)

    with wandb.init(project="gollum", config=wandb_config, group=config["group"]) as run:

        dm = setup_data(config)
        bo = setup_bo_optimizer(config, design_space=dm.heldout_x)

        data_stats = calculate_data_stats(dm.x, dm.y)
        log_data_stats(data_stats)

        json_path = RESULTS_DIR / f"lora_result_{task}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                full_data = json.load(f)
        else:
            full_data = []

        for i in tqdm(range(config["n_iters"]), colour="blue"):

            string = str(dm.data.loc[dm.train_indexes]['string'].values[-1])
            print(string)

            if dry_run:
                acc = float(dm.data.loc[dm.train_indexes][target_column].values[-1])
            else:
                rank, alpha, dropout, batch_size, learning_rate = parse_hparams(string)
                previous_data = [d for d in full_data if d['Configuration'] == string]

                if previous_data:
                    acc = previous_data[0]["Total Accuracy"]
                else:
                    acc = lora_tuning(task, rank, alpha, dropout, learning_rate, batch_size, train_samples)
                    full_data.append({"Configuration": string, "Total Accuracy": acc})
                    with open(json_path, 'w') as f:
                        json.dump(full_data, f, indent=4)

            config_setting.append(string)
            acc_list.append(acc)
            print(f"Configuration : {config_setting}")
            print(f"Total Accuracy: {acc}")

            train_x = dm.train_x.clone().to(DEVICE)
            if i == 0:
                dm.train_y = torch.tensor(acc, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
            else:
                dm.train_y = torch.cat(
                    [dm.train_y, torch.tensor(acc, dtype=torch.float64).unsqueeze(0).unsqueeze(0)],
                    dim=0,
                )
            train_y = dm.train_y.clone().to(DEVICE)
            design_space = dm.heldout_x.clone().to(DEVICE)

            x_next = bo.suggest_next_experiments(train_x, train_y, design_space, i)
            x_next = torch.stack(x_next)

            log_bo_metrics(data_stats, dm.train_y, epoch=i)

            matches = (design_space.unsqueeze(0).to(DEVICE) == x_next).all(dim=-1)
            indices = matches.nonzero(as_tuple=True)[1].to("cpu")

            if not torch.all(matches.sum(dim=-1) == 1):
                print("Unable to find a unique match for some x_next in the dataset.")

            wandb.log({"evaluated_suggestions": wandb.Histogram(dm.heldout_y[indices]), "epoch": i})

            x_next = x_next.squeeze(1)

            evaluated_original_indices = dm.heldout_indices[indices]
            dm.train_indexes = np.append(dm.train_indexes, evaluated_original_indices)
            dm.heldout_indices = np.delete(dm.heldout_indices, indices)
            dm.train_x = dm.x[dm.train_indexes]
            dm.heldout_x = dm.x[dm.heldout_indices]
            dm.heldout_y = dm.y[dm.heldout_indices]

            train_df = dm.data.loc[dm.train_indexes].copy()
            heldout_df = dm.data.loc[dm.heldout_indices.tolist()].copy()
            dm.data = pd.concat([train_df, heldout_df])

            assert len(np.unique(dm.train_indexes)) == len(dm.train_indexes), \
                "Duplicates found in dm.train_indexes"
            assert len(np.unique(dm.heldout_indices)) == len(dm.heldout_indices), \
                "Duplicates found in dm.heldout_indices"
            common_indices = np.intersect1d(dm.train_indexes, dm.heldout_indices)
            assert len(common_indices) == 0, \
                f"Common indices found between train and heldout: {common_indices}"
            assert len(dm.train_indexes) + len(dm.heldout_indices) == len(dm.x), \
                "Mismatch in the total number of indices"

            torch.cuda.empty_cache()
            gc.collect()

            combined = [
                {"Configuration": c, "Total Accuracy": a}
                for c, a in zip(config_setting, acc_list)
            ]
            with open(RESULTS_DIR / f"iteration_{task}_seed_{config['seed']}.json", 'w') as f:
                json.dump(combined, f, indent=4)

        log_bo_metrics(data_stats, dm.train_y, epoch=config["n_iters"])
        logger.setLevel(logging.INFO)
        wandb.finish()

def main():
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mode", default="frozen", choices=list(MODE_CONFIG.keys()))
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    config_file = pre_args.config or str(MODE_CONFIG[pre_args.mode])

    parser = ArgumentParser(
        description="Training script (Vanilla LoRA)",
        default_config_files=[config_file],
    )
    parser.add_argument("--mode", type=str, default="frozen", choices=list(MODE_CONFIG.keys()))
    parser.add_argument("--config", default=config_file)
    parser.add_argument("--task", type=str, default="math", choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--dry_run", action="store_true",
                        help="Look up the objective from the template table instead of training")
    parser.add_argument("--train_samples", type=int, default=None,
                        help="Override the training subset size (patches --sub_task <task>:N)")
    parser.add_argument("--seed", type=int, help="Random seeds to use")
    parser.add_argument("--benchmark", type=str, help="Run a specific benchmark")
    parser.add_argument("--n_iters", type=int, help="How many iterations to run")
    parser.add_argument("--group", type=str, help="Wandb group runs")

    parser.add_subclass_arguments(BaseDataModule, "data", instantiate=False)
    parser.add_subclass_arguments(SurrogateModel, "surrogate_model", instantiate=False)
    parser.add_subclass_arguments(
        AcquisitionFunction, "acquisition", instantiate=False, skip=["model", "best_f"],
    )
    parser.add_subclass_arguments(BotorchOptimizer, "bo", instantiate=False)

    args = parser.parse_args()
    seed_everything(args["seed"], workers=True)
    train(args.as_dict())

if __name__ == "__main__":
    main()
