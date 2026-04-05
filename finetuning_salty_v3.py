"""
finetuning_salty_v3.py  —  Performance Comparison Across Datasets
==================================================================
Benchmarks four adapter methods across four GLUE classification tasks.

Methods compared
----------------
    • LoRA          (PEFT baseline)
    • DoRA          (PEFT baseline)
    • SALT          (SVD head/tail + LoRA on tail)
    • SALTEdoraLinearV4   ← "latest model", eigen-dispersion split

Datasets
--------
    • glue/sst2   — binary sentiment          (~67K train,  872  val)
    • glue/rte    — textual entailment (small) (~2.5K train, 277  val)
    • glue/qnli   — question NLI (large)      (~105K train, 5.4K val)
    • glue/mnli   — multi-genre NLI (3-class) (~393K train, 9.8K val_matched)

Rank sweep
----------
    [8, 16, 32, 64, 128]

Each (dataset × method × rank) combination is trained for NUM_EPOCHS epochs
and evaluated. No weight matrices are saved — this script is optimised for
speed: the only outputs are accuracy, eval_loss, runtime, and trainable params.

Key design choice — eigen dispersion for SALTEdoraLinearV4
-----------------------------------------------------------
    r_top_override = None   →   the model uses choose_head_rank_by_eigen_dispersion()
                                to automatically split head / tail singular subspaces.
    energy_threshold = 0.9  →   fallback threshold inside the dispersion function.

Outputs
-------
results_v3/bert-base-uncased/<mode>/r_<r>/<task>/
    results.json
    loss_per_batch.csv
    loss_per_epoch.csv

results_v3/summary_v3_performance_comparison.csv
"""

import os
import json
import time
import logging
import itertools

import pandas as pd
import torch
import torch.nn as nn

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset
import evaluate
from peft import LoraConfig, get_peft_model

from models import SALT, SALTEdoraLinear, SALTEdoraLinearV2, SALTEdoraLinearV3, SALTEdoraLinearV4

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
MODEL_NAME  = "bert-base-uncased"
NUM_EPOCHS  = 5
LR          = 5e-5
WEIGHT_DECAY = 0.01
TRAIN_BS    = 16
EVAL_BS     = 64

RANKS       = [8, 16, 32, 64, 128]
MODES       = ["lora", "dora", "salt", "saltedora_v4"]

# Head/tail split bounds for cumulative energy knee (SALTEdoraLinearV4)
MIN_FRAC = 0.10   # head gets at least 10% of singular values
MAX_FRAC = 0.60   # head gets at most 60% of singular values


# ===========================================================================
# Dataset registry
# ===========================================================================
TASKS = {
    "glue/sst2": dict(
        subset       = "sst2",
        text         = ("sentence", None),
        num_labels   = 2,
        problem_type = "single_label_classification",
        metrics      = ["accuracy"],
        val_split    = "validation",
    ),
    "glue/rte": dict(
        subset       = "rte",
        text         = ("sentence1", "sentence2"),
        num_labels   = 2,
        problem_type = "single_label_classification",
        metrics      = ["accuracy"],
        val_split    = "validation",
    ),
    "glue/qnli": dict(
        subset       = "qnli",
        text         = ("question", "sentence"),
        num_labels   = 2,
        problem_type = "single_label_classification",
        metrics      = ["accuracy"],
        val_split    = "validation",
    ),
    "glue/mnli": dict(
        subset       = "mnli",
        text         = ("premise", "hypothesis"),
        num_labels   = 3,
        problem_type = "single_label_classification",
        metrics      = ["accuracy"],
        val_split    = "validation_matched",   # MNLI uses matched split
    ),
}


def load_task_dataset(task_key, tokenizer):
    cfg  = TASKS[task_key]
    raw  = load_dataset("glue", cfg["subset"])
    t1, t2 = cfg["text"]

    def tokenize_fn(batch):
        if t2 is None:
            return tokenizer(batch[t1], truncation=True, padding="max_length", max_length=128)
        return tokenizer(batch[t1], batch[t2], truncation=True, padding="max_length", max_length=128)

    tokenized = raw.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    train_ds = tokenized["train"]
    val_ds   = tokenized[cfg["val_split"]]
    return train_ds, val_ds, cfg["num_labels"], cfg["problem_type"]


def build_metrics(task_key):
    cfg  = TASKS[task_key]
    mets = {m: evaluate.load(m) for m in cfg["metrics"]}

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        out   = {}
        for mname, metric in mets.items():
            out[mname] = metric.compute(predictions=preds, references=labels)[mname]
        if "accuracy" in out:
            out["eval_accuracy"] = out["accuracy"]
        return out

    return compute


# ===========================================================================
# Adapter replacement
# ===========================================================================
def replace_qkv_with_adapter(model, r: int, mode: str):
    """
    Freeze the base model then replace Q/K/V linears with the chosen adapter.

    For SALTEdoraLinearV4 the head/tail split is determined automatically by
    the cumulative energy knee method (r_top_override=None), clamped to
    [MIN_FRAC, MAX_FRAC] of the total singular values.
    """
    for p in model.parameters():
        p.requires_grad = False

    if mode in ["lora", "dora"]:
        config = LoraConfig(
            r            = r,
            lora_alpha   = 16,
            target_modules = ["query", "key", "value"],
            use_dora     = (mode == "dora"),
        )
        return get_peft_model(model, config)

    def _recurse(parent):
        for name, module in parent.named_children():
            if isinstance(module, nn.Linear) and any(k in name for k in ["query", "key", "value"]):
                if mode == "salt":
                    setattr(parent, name, SALT(module, r=r * 2, lora_rank=r))
                elif mode == "saltedora_v4":
                    # Cumulative energy knee: r_top_override=None activates the
                    # knee method, clamped to [MIN_FRAC, MAX_FRAC] of singular values.
                    setattr(parent, name, SALTEdoraLinearV4(
                        module,
                        r_intrinsic = r,
                        r_top_override = None,
                        min_frac    = MIN_FRAC,
                        max_frac    = MAX_FRAC,
                    ))
            elif len(list(module.children())) > 0:
                _recurse(module)

    _recurse(model)
    return model


# ===========================================================================
# Training utilities
# ===========================================================================
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TrackingCallback(TrainerCallback):
    """Lightweight callback: logs per-batch loss and per-epoch summary."""

    def __init__(self, output_dir):
        self.output_dir       = output_dir
        self.batch_logs       = []
        self.epoch_logs       = []
        self.epoch_start_time = None
        self.train_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and "loss" in state.log_history[-1]:
            self.batch_logs.append({
                "global_step": state.global_step,
                "epoch":       state.epoch,
                "loss":        state.log_history[-1]["loss"],
            })

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        metrics    = state.log_history[-1] if state.log_history else {}
        self.epoch_logs.append({
            "epoch":      state.epoch,
            "train_loss": metrics.get("loss"),
            "eval_loss":  metrics.get("eval_loss"),
            "epoch_time": epoch_time,
        })

    def on_train_end(self, args, state, control, **kwargs):
        total_runtime = time.time() - self.train_start_time if self.train_start_time else None
        os.makedirs(self.output_dir, exist_ok=True)
        if self.batch_logs:
            pd.DataFrame(self.batch_logs).to_csv(
                os.path.join(self.output_dir, "loss_per_batch.csv"), index=False)
        epoch_df = pd.DataFrame(self.epoch_logs)
        epoch_df["total_runtime_s"] = total_runtime
        epoch_df.to_csv(os.path.join(self.output_dir, "loss_per_epoch.csv"), index=False)


def train_and_evaluate(
    model, train_ds, val_ds, tokenizer, metrics_fn,
    model_name: str, mode: str, r: int, task_key: str,
    num_labels: int, problem_type: str,
) -> dict:
    safe_task  = task_key.replace("/", "_")
    output_dir = f"./results_v3/{model_name}/{mode}/r_{r}/{safe_task}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        learning_rate               = LR,
        per_device_train_batch_size = TRAIN_BS,
        per_device_eval_batch_size  = EVAL_BS,
        num_train_epochs            = NUM_EPOCHS,
        weight_decay                = WEIGHT_DECAY,
        save_strategy               = "no",
        logging_dir                 = os.path.join(output_dir, "logs"),
        logging_steps               = 50,
        report_to                   = [],
    )

    model.config.num_labels   = num_labels
    model.config.problem_type = problem_type

    tracking_cb = TrackingCallback(output_dir)

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        compute_metrics = metrics_fn,
        callbacks       = [tracking_cb],
    )

    start_time    = time.time()
    trainer.train()
    results       = trainer.evaluate(val_ds)
    total_runtime = time.time() - start_time

    n_trainable = count_trainable_params(model)
    results["trainable_params"] = n_trainable
    results["runtime_total_s"]  = total_runtime

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    acc = results.get("eval_accuracy", "N/A")
    if isinstance(acc, float):
        acc = f"{acc:.4f}"
    logger.info(
        f"  ✓ {mode:14s} | r={r:3d} | {task_key:10s} | "
        f"acc={acc} | params={n_trainable:,} | {total_runtime:.0f}s"
    )
    return results


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Pre-load datasets once per task (avoids re-downloading)
    logger.info("Pre-loading all datasets …")
    task_data: dict = {}
    for task_key in TASKS:
        logger.info(f"  Loading {task_key} …")
        train_ds, val_ds, num_labels, problem_type = load_task_dataset(task_key, tokenizer)
        metrics_fn = build_metrics(task_key)
        task_data[task_key] = (train_ds, val_ds, num_labels, problem_type, metrics_fn)

    summary_rows = []
    total_runs   = len(TASKS) * len(MODES) * len(RANKS)
    run_idx      = 0

    # Outer loop: task → mode → rank
    # (group by task so the dataset stays in memory across mode/rank combos)
    for task_key in TASKS:
        train_ds, val_ds, num_labels, problem_type, metrics_fn = task_data[task_key]

        logger.info(f"\n{'='*64}")
        logger.info(f"TASK: {task_key}  ({num_labels} labels)")
        logger.info(f"{'='*64}")

        for mode, r in itertools.product(MODES, RANKS):
            run_idx += 1
            logger.info(f"\n[{run_idx}/{total_runs}] {mode} | r={r} | {task_key}")

            base_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=num_labels
            )
            model = replace_qkv_with_adapter(base_model, r=r, mode=mode)

            results = train_and_evaluate(
                model, train_ds, val_ds, tokenizer, metrics_fn,
                MODEL_NAME, mode, r, task_key, num_labels, problem_type,
            )

            summary_rows.append({
                "task":             task_key,
                "mode":             mode,
                "rank":             r,
                "accuracy":         results.get("eval_accuracy"),
                "eval_loss":        results.get("eval_loss"),
                "runtime_s":        results.get("runtime_total_s"),
                "trainable_params": results.get("trainable_params"),
            })

    # -----------------------------------------------------------------------
    # Save global summary
    # -----------------------------------------------------------------------
    os.makedirs("./results_v3", exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    out_csv    = "./results_v3/summary_v3_performance_comparison.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"\n✅ Done. Global summary → {out_csv}")
    print(summary_df.to_string(index=False))
