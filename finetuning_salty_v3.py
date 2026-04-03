"""
finetuning_salty_v3.py  —  Dataset Sweep
=========================================
Runs SALTEDORA-V4 and full fine-tuning (as baseline) across three GLUE tasks:
    • SST-2  (sentiment)
    • RTE    (textual entailment, small)
    • QNLI   (question NLI, large)

For each (task, mode) pair the script:
    1. Fine-tunes bert-base-uncased for NUM_EPOCHS epochs.
    2. Saves QKV *effective* weight matrices at every epoch + final via
       WeightMatrixSaveCallback.
    3. After both modes finish for a task, runs two principal-angles analyses:

       (a) FINAL comparison   — PA( SALTEDORA_final,  full_ft_final ) per QKV layer
       (b) EPOCH TRAJECTORY   — for each saved epoch e:
                                PA( SALTEDORA_e,      full_ft_e )    per QKV layer

Outputs
-------
results/<model>/<mode>/r_<r>/et_<et>/<task>/
    results.json
    loss_per_batch.csv
    loss_per_epoch.csv
    weights/
        epoch_1.pt  … epoch_N.pt
        final.pt

results/principal_angles/v3/<task>/
    final_comparison_summary.csv         — per-layer mean & max angle at final epoch
    epoch_trajectory_summary.csv         — per-(epoch, layer) mean & max angle
    <layer>_trajectory_angles.npz        — full top-k angle arrays per epoch + final
"""

import os
import json
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from utils.svd_utils import svd_head_tail, truncated_svd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# ---------------------------------------------------------------------------
# Fixed experiment configuration
# ---------------------------------------------------------------------------
MODEL_NAME     = "bert-base-uncased"
SALTEDORA_RANK = 8        # r for SALTEDORA V4
ENERGY_THRESH  = 0.9      # energy_threshold / r_top_override fraction
NUM_EPOCHS     = 5
LR             = 5e-5
WEIGHT_DECAY   = 0.01
TRAIN_BS       = 16
EVAL_BS        = 64
TOP_K_ANGLES   = 32       # top-k principal angles to retain per comparison


# ===========================================================================
# Dataset registry  —  SST-2, RTE, QNLI
# ===========================================================================
TASKS = {
    "glue/sst2": dict(
        subset   = "sst2",
        text     = ("sentence", None),
        num_labels     = 2,
        problem_type   = "single_label_classification",
        metrics        = ["accuracy"],
    ),
    "glue/rte": dict(
        subset   = "rte",
        text     = ("sentence1", "sentence2"),
        num_labels     = 2,
        problem_type   = "single_label_classification",
        metrics        = ["accuracy"],
    ),
    "glue/qnli": dict(
        subset   = "qnli",
        text     = ("question", "sentence"),
        num_labels     = 2,
        problem_type   = "single_label_classification",
        metrics        = ["accuracy"],
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
    val_ds   = tokenized["validation"]
    return train_ds, val_ds, cfg["num_labels"], cfg["problem_type"], cfg


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
def replace_qkv_with_adapter(model, r=8, mode="saltedora_v4",
                              energy_threshold: float | None = None):
    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        return model

    for p in model.parameters():
        p.requires_grad = False

    if mode in ["lora", "dora"]:
        config = LoraConfig(
            r=r, lora_alpha=16,
            target_modules=["query", "key", "value"],
            use_dora=(mode == "dora"),
        )
        return get_peft_model(model, config)

    def _recurse(parent):
        for name, module in parent.named_children():
            if isinstance(module, nn.Linear) and any(k in name for k in ["query", "key", "value"]):
                if mode == "salt":
                    setattr(parent, name, SALT(module, r=r * 2, lora_rank=r))
                elif mode == "saltedora":
                    setattr(parent, name, SALTEdoraLinear(module, r=r))
                elif mode == "saltedora_v2":
                    setattr(parent, name, SALTEdoraLinearV2(module, r=r))
                elif mode == "saltedora_v3":
                    et = 0.9 if energy_threshold is None else float(energy_threshold)
                    setattr(parent, name, SALTEdoraLinearV3(module, r_intrinsic=r, energy_threshold=et))
                elif mode == "saltedora_v4":
                    et = 0.9 if energy_threshold is None else float(energy_threshold)
                    setattr(parent, name, SALTEdoraLinearV4(
                        module, r_intrinsic=r, r_top_override=et, energy_threshold=et))
            elif len(list(module.children())) > 0:
                _recurse(module)

    _recurse(model)
    return model


# ===========================================================================
# Weight extraction  (saves all necessary state for W_eff reconstruction)
# ===========================================================================
def _is_qkv_fullname(fullname: str) -> bool:
    return any(fullname.endswith(s) for s in [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
    ])


def extract_qkv_weights_and_adapter_state(model: nn.Module) -> dict:
    """
    For every QKV layer, save:
        "weight"              – the nn.Linear weight tensor (only if the module
                                exposes .weight directly, i.e. full_ft)
        "state_dict_no_weight"– everything else in the module state_dict
                                (buffers U_top, S_top, … and params alpha, beta, D, R
                                for SALTEDORA; just "bias" for plain linear)
    """
    payload = {"qkv": {}}
    for fullname, module in model.named_modules():
        if _is_qkv_fullname(fullname):
            entry: dict = {}
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                entry["weight"] = module.weight.detach().cpu()
            sd = module.state_dict()
            entry["state_dict_no_weight"] = {
                k: v.detach().cpu() for k, v in sd.items() if k != "weight"
            }
            payload["qkv"][fullname] = entry
    return payload


class WeightMatrixSaveCallback(TrainerCallback):
    """Saves QKV weight state at the end of every epoch and after training."""

    def __init__(self, output_dir: str):
        self.weights_dir = os.path.join(output_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or state.epoch is None:
            return
        epoch_int = int(round(state.epoch))
        payload   = extract_qkv_weights_and_adapter_state(model)
        out_path  = os.path.join(self.weights_dir, f"epoch_{epoch_int}.pt")
        torch.save(payload, out_path)
        logger.info(f"💾 Saved QKV weights @ epoch {epoch_int} → {out_path}")

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        payload  = extract_qkv_weights_and_adapter_state(model)
        out_path = os.path.join(self.weights_dir, "final.pt")
        torch.save(payload, out_path)
        logger.info(f"💾 Saved FINAL QKV weights → {out_path}")


# ===========================================================================
# Training callbacks
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
        print(f"✅ Runtime: {total_runtime:.2f}s → {self.output_dir}")


# ===========================================================================
# Training
# ===========================================================================
def train_and_evaluate(
    model, train_ds, val_ds, tokenizer,
    model_name, r, mode, task_key, num_labels, problem_type,
    metrics_fn, energy_threshold=None,
    learning_rate=LR, weight_decay=WEIGHT_DECAY,
    train_batch_size=TRAIN_BS,
):
    safe_task  = task_key.replace("/", "_")
    et_tag     = "na" if energy_threshold is None else f"{float(energy_threshold):.2f}"
    output_dir = f"./results/{model_name}/{mode}/r_{r}/et_{et_tag}/{safe_task}"
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir                  = output_dir,
        learning_rate               = learning_rate,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size  = EVAL_BS,
        num_train_epochs            = NUM_EPOCHS,
        weight_decay                = weight_decay,
        evaluation_strategy         = "epoch",
        save_strategy               = "no",
        logging_dir                 = os.path.join(output_dir, "logs"),
        logging_steps               = 10,
        report_to                   = [],
    )

    model.config.num_labels   = num_labels
    model.config.problem_type = "single_label_classification"

    tracking_cb = TrackingCallback(output_dir)
    weight_cb   = WeightMatrixSaveCallback(output_dir)

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        compute_metrics = metrics_fn,
        callbacks       = [tracking_cb, weight_cb],
    )

    start_time    = time.time()
    trainer.train()
    results       = trainer.evaluate(val_ds)
    total_runtime = time.time() - start_time

    results["trainable_params"] = count_trainable_params(model)
    results["runtime_total_s"]  = total_runtime
    results["energy_threshold"] = energy_threshold

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    acc = results.get("eval_accuracy", "N/A")
    if isinstance(acc, float):
        acc = f"{acc:.4f}"
    logger.info(f"🔹 {mode} | {task_key} | r={r} | et={et_tag} | acc={acc} | {total_runtime:.1f}s")
    return results, output_dir


# ===========================================================================
# Principal angles analysis
# ===========================================================================

def reconstruct_effective_weight(entry: dict) -> torch.Tensor | None:
    """
    Reconstruct the effective weight matrix W_eff from a saved payload entry.

    full_ft (plain nn.Linear):
        entry["weight"] is the trained weight  → return directly.

    SALTEdoraLinearV4:
        base.weight is FROZEN (does not change during training).
        W_eff = W_head + W_tail_base + delta_tail, where:

            W_head      = U_top  diag(ReLU(S_top * α + β))  Vh_top
            W_tail_base = U_tail diag(S_tail)                Vh_tail
            delta_tail  = (U_r diag(S_r)) diag(ReLU(D)) R   Vh_r

        All buffers (U_*, S_*, Vh_*) and params (alpha, beta, D, R) are
        saved in state_dict_no_weight.
    """
    sd = entry.get("state_dict_no_weight", {})
    W0 = entry.get("weight")  # None for SALTEDORA, tensor for full_ft

    is_saltedora_v4 = ("alpha" in sd and "D" in sd and "R" in sd and "U_top" in sd)

    if is_saltedora_v4:
        U_top     = sd["U_top"].float()
        S_top     = sd["S_top"].float()
        Vh_top    = sd["Vh_top"].float()
        alpha     = sd["alpha"].float()
        beta      = sd["beta"].float()

        U_tail    = sd["U_tail"].float()
        S_tail    = sd["S_tail"].float()
        Vh_tail   = sd["Vh_tail"].float()

        U_tail_r  = sd["U_tail_r"].float()
        S_tail_r  = sd["S_tail_r"].float()
        Vh_tail_r = sd["Vh_tail_r"].float()

        D = sd["D"].float()
        R = sd["R"].float()

        # Head: U_top diag(ReLU(S_top * α + β)) Vh_top
        sigma  = F.relu(S_top * alpha + beta)
        W_head = U_top @ torch.diag(sigma) @ Vh_top

        # Tail base (frozen): U_tail diag(S_tail) Vh_tail
        W_tail_base = U_tail @ torch.diag(S_tail) @ Vh_tail

        # Intrinsic tail delta: (U_r diag(S_r)) diag(ReLU(D)) R Vh_r
        d          = F.relu(D)
        delta_tail = (U_tail_r @ torch.diag(S_tail_r)) @ torch.diag(d) @ R @ Vh_tail_r

        return (W_head + W_tail_base + delta_tail).cpu()

    elif W0 is not None:
        return W0.float()

    else:
        # Fallback: recover base.weight from state dict
        base_w = sd.get("base.weight")
        return base_w.float() if base_w is not None else None


def compute_principal_angles(
    W1: torch.Tensor, W2: torch.Tensor, top_k: int = TOP_K_ANGLES
) -> dict:
    """
    Compute the principal angles between the column spaces of W1 and W2.

    Algorithm (Björck & Golub, 1973):
        1. Economy SVD of W1 and W2 → orthonormal bases U1, U2.
        2. Cross-Gram matrix  M = U1ᵀ U2.
        3. SVD of M: singular values σᵢ = cos(θᵢ).
        4. θᵢ = arccos(σᵢ)  in degrees.

    Args:
        W1, W2  – (out_features × in_features) weight matrices.
        top_k   – retain only the first top_k principal angles.

    Returns:
        dict with cosines, angles_deg, mean_angle_deg, max_angle_deg.
    """
    U1, _, _ = torch.linalg.svd(W1.float(), full_matrices=False)
    U2, _, _ = torch.linalg.svd(W2.float(), full_matrices=False)

    k  = min(top_k, U1.shape[1], U2.shape[1])
    U1 = U1[:, :k]
    U2 = U2[:, :k]

    M       = U1.T @ U2                          # (k × k)
    cosines = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles  = torch.acos(cosines) * (180.0 / torch.pi)

    return {
        "cosines":        cosines.numpy(),
        "angles_deg":     angles.numpy(),
        "mean_angle_deg": float(angles.mean()),
        "max_angle_deg":  float(angles.max()),
    }


def _list_epoch_files(weights_dir: str) -> list[tuple[int, str]]:
    """
    Return sorted list of (epoch_int, filepath) for all epoch_*.pt files.
    """
    pairs = []
    if not os.path.isdir(weights_dir):
        return pairs
    for fname in os.listdir(weights_dir):
        if fname.startswith("epoch_") and fname.endswith(".pt"):
            try:
                epoch_int = int(fname[len("epoch_"):-len(".pt")])
                pairs.append((epoch_int, os.path.join(weights_dir, fname)))
            except ValueError:
                pass
    return sorted(pairs, key=lambda x: x[0])


def analyze_principal_angles(
    saltedora_dir: str,
    full_ft_dir:   str,
    out_dir:       str,
    top_k:         int = TOP_K_ANGLES,
) -> None:
    """
    Two-part principal-angle analysis comparing SALTEDORA and full-FT weight matrices.

    Part 1 — FINAL COMPARISON
        For each QKV layer:
            PA( SALTEDORA_final,  full_ft_final )
        Saved to: {out_dir}/final_comparison_summary.csv

    Part 2 — EPOCH TRAJECTORY
        For each saved epoch e (1 … N) where both runs have a checkpoint:
            For each QKV layer:
                PA( SALTEDORA_epoch_e,  full_ft_epoch_e )
        Saved to:
            {out_dir}/epoch_trajectory_summary.csv
            {out_dir}/<layer>_trajectory_angles.npz
                → keys: "epoch_<e>_saltedora", "epoch_<e>_full_ft",
                         "epoch_<e>_angles_deg"   for every epoch e,
                  plus   "final_angles_deg"
    """
    os.makedirs(out_dir, exist_ok=True)

    def _load(run_dir, fname):
        path = os.path.join(run_dir, "weights", fname)
        if not os.path.exists(path):
            logger.warning(f"Weight file not found: {path}")
            return None
        return torch.load(path, map_location="cpu")

    # ---- final payloads ----
    salty_final  = _load(saltedora_dir, "final.pt")
    full_ft_final = _load(full_ft_dir,  "final.pt")

    if salty_final is None or full_ft_final is None:
        logger.error("final.pt missing for one or both runs — aborting PA analysis.")
        return

    layer_names = list(salty_final["qkv"].keys())

    # ------------------------------------------------------------------
    # Part 1: FINAL COMPARISON
    # ------------------------------------------------------------------
    logger.info("  📐 PA Part 1: SALTEDORA_final vs full_ft_final …")
    final_rows = []
    for layer in layer_names:
        W_s = reconstruct_effective_weight(salty_final["qkv"][layer])
        W_f = reconstruct_effective_weight(full_ft_final["qkv"][layer])
        if W_s is None or W_f is None:
            logger.warning(f"    Skipping {layer} (reconstruction failed).")
            continue
        pa = compute_principal_angles(W_s, W_f, top_k)
        final_rows.append({
            "layer":         layer,
            "mean_angle_deg": pa["mean_angle_deg"],
            "max_angle_deg":  pa["max_angle_deg"],
        })

    pd.DataFrame(final_rows).to_csv(
        os.path.join(out_dir, "final_comparison_summary.csv"), index=False)
    logger.info(f"    → saved final_comparison_summary.csv ({len(final_rows)} layers)")

    # ------------------------------------------------------------------
    # Part 2: EPOCH TRAJECTORY
    # ------------------------------------------------------------------
    logger.info("  📐 PA Part 2: epoch-by-epoch trajectory …")

    salty_epochs  = {e: p for e, p in _list_epoch_files(
        os.path.join(saltedora_dir, "weights"))}
    full_ft_epochs = {e: p for e, p in _list_epoch_files(
        os.path.join(full_ft_dir,  "weights"))}

    common_epochs = sorted(set(salty_epochs.keys()) & set(full_ft_epochs.keys()))
    if not common_epochs:
        logger.warning("    No common epoch checkpoints found — skipping trajectory.")
        return

    # Per-layer storage for npz output:  layer → {key: array}
    layer_trajectory_data: dict[str, dict] = {l: {} for l in layer_names}

    traj_rows = []
    for epoch in common_epochs:
        logger.info(f"    Epoch {epoch} …")
        salty_payload  = torch.load(salty_epochs[epoch],   map_location="cpu")
        full_ft_payload = torch.load(full_ft_epochs[epoch], map_location="cpu")

        for layer in layer_names:
            if layer not in salty_payload["qkv"] or layer not in full_ft_payload["qkv"]:
                continue
            W_s = reconstruct_effective_weight(salty_payload["qkv"][layer])
            W_f = reconstruct_effective_weight(full_ft_payload["qkv"][layer])
            if W_s is None or W_f is None:
                continue

            pa = compute_principal_angles(W_s, W_f, top_k)

            traj_rows.append({
                "epoch":          epoch,
                "layer":          layer,
                "mean_angle_deg": pa["mean_angle_deg"],
                "max_angle_deg":  pa["max_angle_deg"],
            })

            # Store angles for npz
            layer_trajectory_data[layer][f"epoch_{epoch}_angles_deg"] = pa["angles_deg"]
            layer_trajectory_data[layer][f"epoch_{epoch}_cosines"]    = pa["cosines"]

    # Add final angles to the per-layer npz data as well
    for layer in layer_names:
        if layer not in salty_final["qkv"] or layer not in full_ft_final["qkv"]:
            continue
        W_s = reconstruct_effective_weight(salty_final["qkv"][layer])
        W_f = reconstruct_effective_weight(full_ft_final["qkv"][layer])
        if W_s is not None and W_f is not None:
            pa  = compute_principal_angles(W_s, W_f, top_k)
            layer_trajectory_data[layer]["final_angles_deg"] = pa["angles_deg"]
            layer_trajectory_data[layer]["final_cosines"]    = pa["cosines"]

    # Save epoch trajectory summary CSV
    pd.DataFrame(traj_rows).to_csv(
        os.path.join(out_dir, "epoch_trajectory_summary.csv"), index=False)
    logger.info(f"    → saved epoch_trajectory_summary.csv ({len(traj_rows)} rows)")

    # Save per-layer npz files
    for layer, data in layer_trajectory_data.items():
        if not data:
            continue
        safe_name = layer.replace(".", "_")
        np.savez(os.path.join(out_dir, f"{safe_name}_trajectory_angles.npz"), **data)

    logger.info(f"  ✅ PA analysis complete → {out_dir}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    datasets_to_run = list(TASKS.keys())          # ["glue/sst2", "glue/rte", "glue/qnli"]
    # Run full_ft first so its weights are ready before PA analysis
    modes_to_run    = ["full_ft", "saltedora_v4"]

    summary_rows = []

    for task_key in datasets_to_run:
        logger.info(f"\n{'='*64}")
        logger.info(f"TASK: {task_key}")
        logger.info(f"{'='*64}")

        train_ds, val_ds, num_labels, problem_type, cfg = load_task_dataset(task_key, tokenizer)
        metrics_fn = build_metrics(task_key)

        run_dirs: dict[str, str] = {}  # mode → output_dir

        for mode in modes_to_run:
            logger.info(f"\n──── Mode: {mode} ────")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=num_labels
            )
            model = replace_qkv_with_adapter(
                base_model, r=SALTEDORA_RANK, mode=mode,
                energy_threshold=ENERGY_THRESH if mode == "saltedora_v4" else None,
            )

            et = ENERGY_THRESH if mode == "saltedora_v4" else None
            results, out_dir = train_and_evaluate(
                model, train_ds, val_ds, tokenizer,
                MODEL_NAME, SALTEDORA_RANK, mode, task_key,
                num_labels, problem_type, metrics_fn,
                energy_threshold=et,
            )
            run_dirs[mode] = out_dir

            summary_rows.append({
                "task":             task_key,
                "mode":             mode,
                "rank":             SALTEDORA_RANK,
                "energy_threshold": et,
                "accuracy":         results.get("eval_accuracy"),
                "eval_loss":        results.get("eval_loss"),
                "runtime_s":        results.get("runtime_total_s"),
                "trainable_params": results.get("trainable_params"),
            })

        # ---------------------------------------------------------------
        # Principal angles analysis (after both modes finished for task)
        # ---------------------------------------------------------------
        if "full_ft" in run_dirs and "saltedora_v4" in run_dirs:
            safe_task = task_key.replace("/", "_")
            pa_out    = f"./results/principal_angles/v3/{safe_task}"
            logger.info(f"\n📐 Principal angles analysis for {task_key} …")
            analyze_principal_angles(
                saltedora_dir = run_dirs["saltedora_v4"],
                full_ft_dir   = run_dirs["full_ft"],
                out_dir       = pa_out,
                top_k         = TOP_K_ANGLES,
            )

    # -----------------------------------------------------------------------
    # Global summary CSV
    # -----------------------------------------------------------------------
    os.makedirs("./results", exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("./results/summary_v3_dataset_sweep.csv", index=False)
    logger.info("\n✅ Global summary → ./results/summary_v3_dataset_sweep.csv")
    print(summary_df.to_string(index=False))
