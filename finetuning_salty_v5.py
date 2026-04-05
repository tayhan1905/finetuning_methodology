"""
finetuning_salty_v5.py  —  Hyperparameter Tuning for SALTEdoraLinearV4
=======================================================================
Identifies the best hyperparameter combination for SALTEDORA-V4 by sweeping
rank, learning rate, and weight decay on SST-2 (the fastest GLUE task).

Key design choice — eigen dispersion throughout
------------------------------------------------
    r_top_override = None  for every run.
    The model uses choose_head_rank_by_eigen_dispersion() to automatically
    determine the head/tail SVD split based on the curvature of the
    log-singular-value spectrum. No fixed energy fraction is imposed.
    energy_threshold=0.9 only acts as a fallback when the curvature signal
    is too weak (i.e. the dispersion is nearly flat).

Hyperparameter grids
--------------------
    rank           : [8, 16, 32, 64, 128]      — r_intrinsic for SALTEDORA
    learning_rate  : [1e-5, 3e-5, 5e-5, 1e-4]
    weight_decay   : [0.001, 0.01, 0.1]
    batch_size     : 16 (fixed — reduces run count while keeping fair comparison)

    Total SALTEDORA runs : 5 × 4 × 3 = 60
    + 1 full_ft reference run

The full summary CSV is sorted by accuracy so the best configuration is
immediately visible at the top.

Outputs
-------
results_v5/bert-base-uncased/saltedora_v4/r_<r>/lr_<lr>/wd_<wd>/glue_sst2/
    results.json
    loss_per_batch.csv
    loss_per_epoch.csv

results_v5/bert-base-uncased/full_ft/reference/glue_sst2/
    (same structure — single reference run)

results_v5/summary_v5_hp_tuning.csv    ← sorted best → worst accuracy
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
# Fixed settings
# ---------------------------------------------------------------------------
MODEL_NAME       = "bert-base-uncased"
TASK_KEY         = "glue/sst2"
NUM_EPOCHS       = 5
TRAIN_BS         = 16          # fixed — not swept to keep run count manageable
EVAL_BS          = 64
MIN_FRAC = 0.10   # head gets at least 10% of singular values
MAX_FRAC = 0.60   # head gets at most 60% of singular values

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
RANK_GRID = [8, 16, 32, 64, 128]
LR_GRID   = [1e-5, 3e-5, 5e-5, 1e-4]
WD_GRID   = [0.001, 0.01, 0.1]

# Reference full_ft settings (fixed)
FULLFT_LR = 5e-5
FULLFT_WD = 0.01


# ===========================================================================
# Dataset helpers
# ===========================================================================
def load_sst2(tokenizer):
    raw = load_dataset("glue", "sst2")

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True, padding="max_length", max_length=128,
        )

    tokenized = raw.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized["train"], tokenized["validation"]


def build_accuracy_metric():
    acc_metric = evaluate.load("accuracy")

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        res   = acc_metric.compute(predictions=preds, references=labels)
        return {"eval_accuracy": res["accuracy"], "accuracy": res["accuracy"]}

    return compute


# ===========================================================================
# Adapter replacement
# ===========================================================================
def replace_qkv_with_adapter(model, r: int, mode: str):
    """
    Freeze the base model then inject the chosen adapter into every Q/K/V layer.

    For SALTEdoraLinearV4:
        r_top_override=None  →  cumulative energy knee determines the head/tail split.
        min_frac/max_frac    →  clamp the knee result to [10%, 60%] of singular values.
        r_intrinsic=r        →  controls the intrinsic rank of the tail update.
    """
    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        return model

    for p in model.parameters():
        p.requires_grad = False

    if mode in ["lora", "dora"]:
        config = LoraConfig(
            r              = r,
            lora_alpha     = 16,
            target_modules = ["query", "key", "value"],
            use_dora       = (mode == "dora"),
        )
        return get_peft_model(model, config)

    def _recurse(parent):
        for name, module in parent.named_children():
            if isinstance(module, nn.Linear) and any(k in name for k in ["query", "key", "value"]):
                if mode == "saltedora_v4":
                    setattr(parent, name, SALTEdoraLinearV4(
                        module,
                        r_intrinsic    = r,
                        r_top_override = None,    # ← cumulative energy knee
                        min_frac       = MIN_FRAC,
                        max_frac       = MAX_FRAC,
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


def _make_output_dir(mode: str, r: int, lr: float, wd: float) -> str:
    lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
    if mode == "full_ft":
        return f"./results_v5/{MODEL_NAME}/full_ft/reference/glue_sst2"
    return f"./results_v5/{MODEL_NAME}/saltedora_v4/r_{r}/lr_{lr_str}/wd_{wd}/glue_sst2"


def train_and_evaluate(
    model, train_ds, val_ds, tokenizer, metrics_fn,
    mode: str, r: int, lr: float, wd: float,
) -> tuple[dict, str]:
    output_dir = _make_output_dir(mode, r, lr, wd)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        learning_rate               = lr,
        per_device_train_batch_size = TRAIN_BS,
        per_device_eval_batch_size  = EVAL_BS,
        num_train_epochs            = NUM_EPOCHS,
        weight_decay                = wd,
        save_strategy               = "no",
        logging_dir                 = os.path.join(output_dir, "logs"),
        logging_steps               = 50,
        report_to                   = [],
    )

    model.config.num_labels   = 2
    model.config.problem_type = "single_label_classification"

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
    results["rank"]             = r
    results["lr"]               = lr
    results["weight_decay"]     = wd
    results["mode"]             = mode

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    acc    = results.get("eval_accuracy", "N/A")
    lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
    logger.info(
        f"  ✓ {mode:14s} | r={r:3d} | lr={lr_str} | wd={wd} | "
        f"acc={acc:.4f} | params={n_trainable:,} | {total_runtime:.0f}s"
    )
    return results, output_dir


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds = load_sst2(tokenizer)
    metrics_fn = build_accuracy_metric()

    summary_rows = []

    # -----------------------------------------------------------------------
    # Step 1: Full fine-tuning reference baseline (run once)
    # -----------------------------------------------------------------------
    logger.info("\n" + "="*64)
    logger.info("STEP 1: Full fine-tuning reference baseline")
    logger.info("="*64)

    ft_base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    ft_model = replace_qkv_with_adapter(ft_base, r=0, mode="full_ft")

    ft_results, _ = train_and_evaluate(
        ft_model, train_ds, val_ds, tokenizer, metrics_fn,
        mode="full_ft", r=0, lr=FULLFT_LR, wd=FULLFT_WD,
    )

    summary_rows.append({
        "mode":             "full_ft",
        "rank":             0,
        "lr":               FULLFT_LR,
        "weight_decay":     FULLFT_WD,
        "accuracy":         ft_results.get("eval_accuracy"),
        "eval_loss":        ft_results.get("eval_loss"),
        "runtime_s":        ft_results.get("runtime_total_s"),
        "trainable_params": ft_results.get("trainable_params"),
    })

    # -----------------------------------------------------------------------
    # Step 2: SALTEDORA-V4 HP sweep  (rank × lr × wd)
    # -----------------------------------------------------------------------
    hp_grid    = list(itertools.product(RANK_GRID, LR_GRID, WD_GRID))
    total_runs = len(hp_grid)

    logger.info(f"\n{'='*64}")
    logger.info(f"STEP 2: SALTEDORA-V4 HP sweep — {total_runs} runs")
    logger.info(f"  Ranks : {RANK_GRID}")
    logger.info(f"  LRs   : {LR_GRID}")
    logger.info(f"  WDs   : {WD_GRID}")
    logger.info(f"{'='*64}")

    for idx, (r, lr, wd) in enumerate(hp_grid, 1):
        lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
        logger.info(f"\n[{idx}/{total_runs}] r={r} | lr={lr_str} | wd={wd}")

        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model      = replace_qkv_with_adapter(base_model, r=r, mode="saltedora_v4")

        results, _ = train_and_evaluate(
            model, train_ds, val_ds, tokenizer, metrics_fn,
            mode="saltedora_v4", r=r, lr=lr, wd=wd,
        )

        summary_rows.append({
            "mode":             "saltedora_v4",
            "rank":             r,
            "lr":               lr,
            "weight_decay":     wd,
            "accuracy":         results.get("eval_accuracy"),
            "eval_loss":        results.get("eval_loss"),
            "runtime_s":        results.get("runtime_total_s"),
            "trainable_params": results.get("trainable_params"),
        })

    # -----------------------------------------------------------------------
    # Save global summary (sorted best → worst)
    # -----------------------------------------------------------------------
    os.makedirs("./results_v5", exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    out_csv    = "./results_v5/summary_v5_hp_tuning.csv"
    summary_df.to_csv(out_csv, index=False)

    logger.info(f"\n✅ Done. Global summary (best → worst) → {out_csv}")
    print(summary_df.to_string(index=False))

    # Print top 5
    logger.info("\nTop 5 configurations:")
    print(summary_df.head(5).to_string(index=False))
