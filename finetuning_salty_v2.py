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
from models import SALT, SALTEdoraLinear, SALTEdoraLinearV2, SALTEdoraLinearV3, SALTEdoraLinearV4
from utils.svd_utils import svd_head_tail, truncated_svd
import argparse

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


# ===================== [NEW] Weight extraction helpers =====================
def _is_qkv_fullname(fullname: str) -> bool:
    return any(fullname.endswith(suffix) for suffix in [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value"
    ])


def extract_qkv_weights_and_adapter_state(model: nn.Module):
    payload = {"qkv": {}}

    for fullname, module in model.named_modules():
        if _is_qkv_fullname(fullname):
            entry = {}

            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                entry["weight"] = module.weight.detach().cpu()

            sd = module.state_dict()
            entry["state_dict_no_weight"] = {
                k: v.detach().cpu() for k, v in sd.items() if k != "weight"
            }

            payload["qkv"][fullname] = entry

    return payload


# ===================== [NEW] Callback to save weights each epoch + final =====================
class WeightMatrixSaveCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.weights_dir = os.path.join(output_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return
        epoch = state.epoch
        if epoch is None:
            return

        epoch_int = int(round(epoch))

        payload = extract_qkv_weights_and_adapter_state(model)
        out_path = os.path.join(self.weights_dir, f"epoch_{epoch_int}.pt")
        torch.save(payload, out_path)
        logger.info(f"💾 Saved QKV weights @ epoch {epoch_int} -> {out_path}")

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return
        payload = extract_qkv_weights_and_adapter_state(model)
        out_path = os.path.join(self.weights_dir, "final.pt")
        torch.save(payload, out_path)
        logger.info(f"💾 Saved FINAL QKV weights -> {out_path}")


# ----------------------------
# Adapter Replacement function
# ----------------------------
# ===================== [UPDATED] add energy_threshold param =====================
def replace_qkv_with_adapter(model, r=8, mode="lora", energy_threshold: float | None = None):
    """
    Replaces ONLY the query/key/value linear layers in BERT's attention modules
    with the specified adapter type (LoRA, DoRA, SALT, SALT-EDoRA, SALT-EDoRA-V2, SALT-EDoRA-V3, SALT-EDoRA-V4).

    Ensures the base model is frozen before replacement so that
    trainable parameter counts remain comparable and small.
    """

    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        return model

    for p in model.parameters():
        p.requires_grad = False

    if mode in ["lora", "dora"]:
        use_dora = (mode == "dora")
        config = LoraConfig(
            r=r,
            lora_alpha=16,
            target_modules=["query", "key", "value"],
            use_dora=use_dora,
        )
        return get_peft_model(model, config)

    def _recurse(parent: nn.Module):
        for name, module in parent.named_children():
            if isinstance(module, nn.Linear) and any(k in name for k in ["query", "key", "value"]):
                if mode == "salt":
                    setattr(parent, name, SALT(module, r=r * 2, lora_rank=r))
                elif mode == "saltedora":
                    setattr(parent, name, SALTEdoraLinear(module, r=r))
                elif mode == "saltedora_v2":
                    setattr(parent, name, SALTEdoraLinearV2(module, r=r))
                elif mode == "saltedora_v3":
                    # ===================== [UPDATED] use passed energy_threshold, default=0.9 =====================
                    et = 0.9 if energy_threshold is None else float(energy_threshold)
                    setattr(parent, name, SALTEdoraLinearV3(module, r_intrinsic=r, energy_threshold=et))
                elif mode == "saltedora_v4":
                    # ===================== [UPDATED] use passed energy_threshold, default=0.9 =====================
                    # TODO: Try to make this more elegant >> I use the override here to test out different patterns and see which one offers the best accuracy >> right now the metric gives about half, we should see whether we can attempt to make this btr
                    et = 0.9 if energy_threshold is None else float(energy_threshold)
                    setattr(parent, name, SALTEdoraLinearV4(module, r_intrinsic=r, r_top_override=et, energy_threshold=et))
            else:
                if len(list(module.children())) > 0:
                    _recurse(module)

    _recurse(model)
    return model


# ----------------------------
# Dataset registry (GLUE only, accuracy metrics)
# ----------------------------
TASKS = {
    "glue/sst2": dict(hub_id="glue/sst2", subset="sst2",
                      text=("sentence", None), num_labels=2,
                      problem_type="single_label_classification",
                      metrics=["accuracy"]),
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

    tokenized = raw.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column(label_col, "labels")
    tokenized.set_format("torch")

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
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.batch_logs = []
        self.epoch_logs = []
        self.epoch_start_time = None
        self.train_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and "loss" in state.log_history[-1]:
            self.batch_logs.append({
                "global_step": state.global_step,
                "epoch": state.epoch,
                "loss": state.log_history[-1]["loss"]
            })

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        metrics = state.log_history[-1] if state.log_history else {}
        self.epoch_logs.append({
            "epoch": state.epoch,
            "train_loss": metrics.get("loss", None),
            "eval_loss": metrics.get("eval_loss", None),
            "epoch_time": epoch_time
        })

    def on_train_end(self, args, state, control, **kwargs):
        total_runtime = time.time() - self.train_start_time if self.train_start_time else None
        os.makedirs(self.output_dir, exist_ok=True)

        if self.batch_logs:
            batch_df = pd.DataFrame(self.batch_logs)
            batch_df.to_csv(os.path.join(self.output_dir, "loss_per_batch.csv"), index=False)

        epoch_df = pd.DataFrame(self.epoch_logs)
        epoch_df["total_runtime_s"] = total_runtime
        epoch_df.to_csv(os.path.join(self.output_dir, "loss_per_epoch.csv"), index=False)

        print(f"✅ Training complete. Total runtime: {total_runtime:.2f}s saved to {self.output_dir}")


# ----------------------------
# Training + Evaluation
# ----------------------------
# ===================== [UPDATED] add energy_threshold to signature + output_dir =====================
def train_and_evaluate(
    model, train_ds, val_ds, tokenizer, model_name, r, mode, task_key,
    num_labels, problem_type, metrics_fn, energy_threshold: float | None = None
):
    safe_task = task_key.replace("/", "_")
    et_tag = "na" if energy_threshold is None else f"{float(energy_threshold):.2f}"
    output_dir = f"./results/{model_name}/{mode}/r_{r}/et_{et_tag}/{safe_task}"
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
        report_to=[],
    )

    model.config.num_labels = num_labels
    model.config.problem_type = "single_label_classification"

    tracking_cb = TrackingCallback(output_dir)
    weight_cb = WeightMatrixSaveCallback(output_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics_fn,
        callbacks=[tracking_cb, weight_cb],
    )

    start_time = time.time()
    trainer.train()
    results = trainer.evaluate(val_ds)
    total_runtime = time.time() - start_time

    results["trainable_params"] = count_trainable_params(model)
    results["runtime_total_s"] = total_runtime
    results["energy_threshold"] = energy_threshold

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"🔹 Completed {mode} | {task_key} | r={r} | et={et_tag} | Accuracy={results.get('eval_accuracy', 'N/A'):.4f} | Runtime={total_runtime:.2f}s")
    logger.info(f"🔹 Completed {mode} | {task_key} | r={r} | et={et_tag} | Accuracy={results.get('eval_accuracy', 'N/A'):.4f} | Runtime={total_runtime:.2f}s")
    return results

# ----------------------------
# Main experiment runner
# ----------------------------
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dora_rank = 8
    saltedora_v4_ranks = [4, 8, 16, 32, 64, 80, 96, 112, 128, 160]

    # ===================== [NEW] energy threshold sweep =====================
    energy_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ===================== [UPDATED] runs now include energy_threshold =====================
    runs = []
    runs.append((0, "full_ft", None))
    runs.append((dora_rank, "dora", None))
    for r in saltedora_v4_ranks:
        for et in energy_thresholds:
            runs.append((r, "saltedora_v4", et))

    datasets_to_run = ["glue/sst2"]

    summary_rows = []
    logger.info("Starting SST-2 experiments (Full FT vs DoRA vs SALTEDORA_V4 w/ energy-override sweep)...")

    # ===================== [UPDATED] unpack (r, mode, et) =====================
    for (r, mode, et), task_key in itertools.product(runs, datasets_to_run):
        logger.info(f"Running {mode} (rank={r}, et={et}) on {task_key}...")

        train_ds, val_ds, num_labels, problem_type, cfg = load_task_dataset(task_key, tokenizer, model_name)
        metrics_fn = build_metrics(task_key)

        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # ===================== [UPDATED] pass energy_threshold through =====================
        model = replace_qkv_with_adapter(base_model, r=r, mode=mode, energy_threshold=et)

        results = train_and_evaluate(
            model, train_ds, val_ds, tokenizer,
            model_name, r, mode, task_key,
            num_labels, problem_type, metrics_fn,
            energy_threshold=et
        )

        summary_rows.append({
            "rank": r,
            "mode": mode,
            "energy_threshold": et,
            "task": task_key,
            "accuracy": results.get("eval_accuracy", None),
            "eval_loss": results.get("eval_loss", None),
            "runtime_s": results.get("runtime_total_s", None),
            "trainable_params": results.get("trainable_params", None)
        })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs("./results", exist_ok=True)
    summary_df.to_csv("./results/summary_sst2_fullft_dora_saltedoraV4_energy_sweep.csv", index=False)
    logger.info("Global summary saved to ./results/summary_sst2_fullft_dora_saltedoraV4_energy_sweep.csv")
    print(summary_df)