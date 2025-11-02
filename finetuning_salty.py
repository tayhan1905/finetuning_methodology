import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset
import evaluate
import json
import os
import pandas as pd
import itertools
import logging
import time
from peft import LoraConfig, get_peft_model
from models import SALT, SALTEdoraLinear, SALTEdoraLinearV2, SALTEdoraLinearV3
from utils.svd_utils import svd_head_tail, truncated_svd
import argparse

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# ----------------------------
# Adapter Replacement function
# ----------------------------
def replace_qkv_with_adapter(model, r=8, mode="lora"):
    """
    Replaces ONLY the query/key/value linear layers in BERT's attention modules
    with the specified adapter type (LoRA, DoRA, SALT, SALT-EDoRA, SALT-EDoRA-V2).

    Ensures the base model is frozen before replacement so that
    trainable parameter counts remain comparable and small.
    """
    # 1️⃣ Freeze everything first (crucial for fair comparison)
    for p in model.parameters():
        p.requires_grad = False

    # 2️⃣ Handle LoRA / DoRA using PEFT
    if mode in ["lora", "dora"]:
        use_dora = (mode == "dora")
        config = LoraConfig(
            r=r,
            lora_alpha=16,
            target_modules=["query", "key", "value"],
            use_dora=use_dora,
        )
        return get_peft_model(model, config)

    # 3️⃣ Custom adapters (SALT / SALT-EDoRA)
    for name, module in model.named_children():
        # Only touch Q, K, V linears
        if isinstance(module, nn.Linear) and any(k in name for k in ["query", "key", "value"]):
            if mode == "salt":
                setattr(model, name, SALT(module, r = r * 2, lora_rank=r))
            elif mode == "saltedora":
                setattr(model, name, SALTEdoraLinear(module, r=r))
            elif mode == "saltedora_v2":
                setattr(model, name, SALTEdoraLinearV2(module, r=r))
            elif mode == "saltedora_v3":
                setattr(model, name, SALTEdoraLinearV3(module, r_intrinsic=r))
        else:
            # Recurse only if the child has its own children
            # (prevents repeated descent into leaf modules)
            if len(list(module.children())) > 0:
                replace_qkv_with_adapter(module, r=r, mode=mode)

    return model

# ----------------------------
# Dataset registry (GLUE only, accuracy metrics)
# ----------------------------
TASKS = {
    "glue/sst2": dict(hub_id="glue/sst2", subset="sst2",
                      text=("sentence", None), num_labels=2,
                      problem_type="single_label_classification",
                      metrics=["accuracy"]),
    # "glue/rte": dict(hub_id="glue/rte", subset="rte",
    #                  text=("sentence1", "sentence2"), num_labels=2,
    #                  problem_type="single_label_classification",
    #                  metrics=["accuracy"]),
#     "glue/qnli": dict(hub_id="glue/qnli", subset="qnli",
#                       text=("question", "sentence"), num_labels=2,
#                       problem_type="single_label_classification",
#                       metrics=["accuracy"]),
#     "glue/mnli": dict(hub_id="glue/mnli", subset="mnli",
#                       text=("premise", "hypothesis"), num_labels=3,
#                       problem_type="single_label_classification",
#                       metrics=["accuracy"]),
#     "glue/qqp": dict(hub_id="glue/qqp", subset="qqp",
#                      text=("question1", "question2"), num_labels=2,
#                      problem_type="single_label_classification",
#                      metrics=["accuracy"]),
}

# ----------------------------
# Dataset loading
# ----------------------------
def load_task_dataset(task_key, tokenizer, model_name):
    cfg = TASKS[task_key]

    raw = load_dataset("glue", cfg["subset"])
    label_col = "label"
    t1, t2 = cfg["text"]

    def tokenize_fn(batch):
        if t2 is None:
            return tokenizer(batch[t1], truncation=True, padding="max_length", max_length=128)
        else:
            return tokenizer(batch[t1], batch[t2], truncation=True, padding="max_length", max_length=128)

    ## Map all the inputs and tokenize them beofre throwing them into the model
    tokenized = raw.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column(label_col, "labels")
    tokenized.set_format("torch")

    if task_key == "glue/mnli":
        train_ds = tokenized["train"]
        val_ds = tokenized["validation_matched"]
    else:
        train_ds = tokenized["train"]
        val_ds = tokenized["validation"]

    return train_ds, val_ds, cfg["num_labels"], cfg["problem_type"], cfg

# ----------------------------
# Metric builder
# ----------------------------
def build_metrics(task_key):
    cfg = TASKS[task_key]
    mets = {m: evaluate.load(m) for m in cfg["metrics"]}

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        out = {}
        for mname, metric in mets.items():
            out[mname] = metric.compute(predictions=preds, references=labels)[mname]
        if "accuracy" in out:
            out["eval_accuracy"] = out["accuracy"]
        return out

    return compute

# ----------------------------
# Utilities
# ----------------------------
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TrackingCallback(TrainerCallback):
    """
    Tracks and saves:
    - Loss per batch (step)
    - Loss and eval_loss per epoch
    - Time per epoch
    - Total training runtime
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.batch_logs = []
        self.epoch_logs = []
        self.epoch_start_time = None
        self.train_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Logs loss after every batch."""
        if state.log_history and "loss" in state.log_history[-1]:
            self.batch_logs.append({
                "global_step": state.global_step,
                "epoch": state.epoch,
                "loss": state.log_history[-1]["loss"]
            })

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Logs eval + train loss and time per epoch."""
        epoch_time = time.time() - self.epoch_start_time
        metrics = state.log_history[-1] if state.log_history else {}
        self.epoch_logs.append({
            "epoch": state.epoch,
            "train_loss": metrics.get("loss", None),
            "eval_loss": metrics.get("eval_loss", None),
            "epoch_time": epoch_time
        })

    def on_train_end(self, args, state, control, **kwargs):
        """Save batch- and epoch-level logs."""
        total_runtime = time.time() - self.train_start_time if self.train_start_time else None
        os.makedirs(self.output_dir, exist_ok=True)

        # Save per-batch losses
        if self.batch_logs:
            batch_df = pd.DataFrame(self.batch_logs)
            batch_df.to_csv(os.path.join(self.output_dir, "loss_per_batch.csv"), index=False)

        # Save per-epoch losses + time + total runtime
        epoch_df = pd.DataFrame(self.epoch_logs)
        epoch_df["total_runtime_s"] = total_runtime
        epoch_df.to_csv(os.path.join(self.output_dir, "loss_per_epoch.csv"), index=False)

        print(f"✅ Training complete. Total runtime: {total_runtime:.2f}s saved to {self.output_dir}")

# ----------------------------
# Training + Evaluation
# ----------------------------
def train_and_evaluate(model, train_ds, val_ds, tokenizer, model_name, r, mode, task_key, num_labels, problem_type, metrics_fn):
    safe_task = task_key.replace("/", "_")
    output_dir = f"./results/{model_name}/{mode}/r_{r}/{safe_task}"
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="no",
    )

    model.config.num_labels = num_labels
    model.config.problem_type = "single_label_classification"

    tracking_cb = TrackingCallback(output_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics_fn,
        callbacks=[tracking_cb],
    )

    # Start timer for total training duration
    start_time = time.time()
    trainer.train()
    results = trainer.evaluate(val_ds)
    total_runtime = time.time() - start_time

    # Add metadata for this run
    results["trainable_params"] = count_trainable_params(model)
    print(count_trainable_params(model))
    logger.info(count_trainable_params(model))
    results["runtime_total_s"] = total_runtime

    # Save per-run summary JSON
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"🔹 Completed {mode} | {task_key} | r={r} | Accuracy={results.get('eval_accuracy', 'N/A'):.4f} | Runtime={total_runtime:.2f}s")
    logger.info(f"🔹 Completed {mode} | {task_key} | r={r} | Accuracy={results.get('eval_accuracy', 'N/A'):.4f} | Runtime={total_runtime:.2f}s")
    return results

# ----------------------------
# Main experiment runner
# ----------------------------
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ranks = [8, 16, 32, 64]
    modes = ["lora", "dora", "salt", "saltedora_v3"]
    # modes = ["saltedora_v2"]
    datasets_to_run = list(TASKS.keys())

    summary_rows = []
    logger.info("Starting GLUE classification experiments...")

    for r, mode, task_key in itertools.product(ranks, modes, datasets_to_run):
        logger.info(f"Running {mode} (rank={r}) on {task_key}...")
        train_ds, val_ds, num_labels, problem_type, cfg = load_task_dataset(task_key, tokenizer, model_name)
        metrics_fn = build_metrics(task_key)

        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model = replace_qkv_with_adapter(base_model, r=r, mode=mode)
        # train_and_evaluate(model, train_ds, val_ds, tokenizer, model_name, r, mode, task_key, num_labels, problem_type, metrics_fn)

        results = train_and_evaluate(model, train_ds, val_ds, tokenizer, model_name, r, mode, task_key, num_labels, problem_type, metrics_fn)

        summary_rows.append({
            "rank": r,
            "mode": mode,
            "task": task_key,
            "accuracy": results.get("eval_accuracy", None),
            "eval_loss": results.get("eval_loss", None),
            "runtime_s": results.get("runtime_total"),
            "trainable_params": results.get("trainable_params")
        })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("./results", exist_ok=True)
    summary_df.to_csv("./results/summary_glue_classification.csv", index=False)
    logger.info("Global summary saved to ./results/summary_glue_classification.csv")
    print(summary_df)
