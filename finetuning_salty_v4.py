"""
finetuning_salty_v4.py  —  Hyperparameter Tuning
==================================================
Sweeps hyperparameters for SALTEDORA-V4 on SST-2 (glue/sst2).
Runs full fine-tuning (full_ft) once as a fixed reference baseline.

Hyperparameter grids
--------------------
    Primary sweep  (--sweep lr_rank)   : lr × r
        learning_rate  : [1e-5, 3e-5, 5e-5, 1e-4]
        rank r         : [4, 8, 16, 32, 64]
        (wd=0.01, bs=16 fixed)

    Secondary sweep (--sweep wd_bs)    : weight_decay × batch_size
        weight_decay   : [0.001, 0.01, 0.1]
        batch_size     : [8, 16, 32]
        (lr=5e-5, r=8 fixed)

    Full sweep      (--sweep all)      : lr × r × wd × bs (warning: 180 runs)

Usage
-----
    python finetuning_salty_v4.py --sweep lr_rank      # default
    python finetuning_salty_v4.py --sweep wd_bs
    python finetuning_salty_v4.py --sweep all          # expensive

Principal-angles analysis
--------------------------
For every SALTEDORA HP configuration the script runs the same two-part PA
analysis as finetuning_salty_v3.py, comparing against the shared full_ft
reference baseline:

    Part 1 — FINAL COMPARISON  : PA( SALTEDORA_final,  full_ft_final )
    Part 2 — EPOCH TRAJECTORY  : PA( SALTEDORA_e,      full_ft_e )   ∀ epoch e

Outputs
-------
results/bert-base-uncased/saltedora_v4/lr_<lr>/r_<r>/wd_<wd>/bs_<bs>/glue_sst2/
    results.json
    loss_per_batch.csv
    loss_per_epoch.csv
    weights/epoch_<n>.pt  …  final.pt

results/bert-base-uncased/full_ft/reference/glue_sst2/
    (same structure — run once as shared baseline)

results/principal_angles/v4/hp_<tag>/
    final_comparison_summary.csv
    epoch_trajectory_summary.csv
    <layer>_trajectory_angles.npz

results/summary_v4_hp_tuning.csv
"""

import os
import json
import time
import logging
import argparse
import itertools

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
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME       = "bert-base-uncased"
TASK_KEY         = "glue/sst2"
ENERGY_THRESH    = 0.9        # fixed for HP sweep (already explored in v2)
NUM_EPOCHS       = 5
EVAL_BS          = 64
TOP_K_ANGLES     = 32

# Default / fixed values used when a HP is not being swept
DEFAULT_LR       = 5e-5
DEFAULT_RANK     = 8
DEFAULT_WD       = 0.01
DEFAULT_BS       = 16

# HP grids
LR_GRID          = [1e-5, 3e-5, 5e-5, 1e-4]
RANK_GRID        = [16, 32, 64, 128]
WD_GRID          = [0.001, 0.01, 0.1]
BS_GRID          = [8, 16, 32]


# ===========================================================================
# Dataset helpers
# ===========================================================================
def load_sst2(tokenizer):
    raw = load_dataset("glue", "sst2")

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True, padding="max_length", max_length=128
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
                        module, r_intrinsic=r, r_top_override=None, energy_threshold=et))
            elif len(list(module.children())) > 0:
                _recurse(module)

    _recurse(model)
    return model


# ===========================================================================
# Weight extraction
# ===========================================================================
def _is_qkv_fullname(fullname: str) -> bool:
    return any(fullname.endswith(s) for s in [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
    ])


def extract_qkv_weights_and_adapter_state(model: nn.Module) -> dict:
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
def _make_output_dir(mode: str, lr: float, r: int, wd: float, bs: int) -> str:
    lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
    if mode == "full_ft":
        return f"./results_v4/{MODEL_NAME}/full_ft/reference/glue_sst2"
    return (
        f"./results_v4/{MODEL_NAME}/saltedora_v4"
        f"/lr_{lr_str}/r_{r}/wd_{wd}/bs_{bs}/glue_sst2"
    )


def train_and_evaluate(
    model, train_ds, val_ds, tokenizer, metrics_fn,
    mode: str, lr: float, r: int, wd: float, bs: int,
    energy_threshold: float | None = None,
) -> tuple[dict, str]:
    output_dir = _make_output_dir(mode, lr, r, wd, bs)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        learning_rate               = lr,
        per_device_train_batch_size = bs,
        per_device_eval_batch_size  = EVAL_BS,
        num_train_epochs            = NUM_EPOCHS,
        weight_decay                = wd,
        # evaluation_strategy         = "epoch",
        save_strategy               = "no",
        logging_dir                 = os.path.join(output_dir, "logs"),
        logging_steps               = 10,
        report_to                   = [],
    )

    model.config.num_labels   = 2
    model.config.problem_type = "single_label_classification"

    tracking_cb = TrackingCallback(output_dir)
    weight_cb   = WeightMatrixSaveCallback(output_dir)

    trainer = Trainer(
        model           = model,
        args            = training_args,
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
    results["lr"]               = lr
    results["rank"]             = r
    results["weight_decay"]     = wd
    results["batch_size"]       = bs

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    acc = results.get("eval_accuracy", "N/A")
    if isinstance(acc, float):
        acc = f"{acc:.4f}"
    lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
    logger.info(
        f"🔹 {mode} | lr={lr_str} | r={r} | wd={wd} | bs={bs} | "
        f"acc={acc} | {total_runtime:.1f}s"
    )
    return results, output_dir


# ===========================================================================
# Principal angles analysis  (shared with finetuning_salty_v3.py)
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

        sigma  = F.relu(S_top * alpha + beta)
        W_head = U_top @ torch.diag(sigma) @ Vh_top

        W_tail_base = U_tail @ torch.diag(S_tail) @ Vh_tail

        d          = F.relu(D)
        delta_tail = (U_tail_r @ torch.diag(S_tail_r)) @ torch.diag(d) @ R @ Vh_tail_r

        return (W_head + W_tail_base + delta_tail).cpu()

    elif W0 is not None:
        return W0.float()

    else:
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
    """
    U1, _, _ = torch.linalg.svd(W1.float(), full_matrices=False)
    U2, _, _ = torch.linalg.svd(W2.float(), full_matrices=False)

    k  = min(top_k, U1.shape[1], U2.shape[1])
    U1 = U1[:, :k]
    U2 = U2[:, :k]

    M       = U1.T @ U2
    cosines = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles  = torch.acos(cosines) * (180.0 / torch.pi)

    return {
        "cosines":        cosines.numpy(),
        "angles_deg":     angles.numpy(),
        "mean_angle_deg": float(angles.mean()),
        "max_angle_deg":  float(angles.max()),
    }


def _list_epoch_files(weights_dir: str) -> list[tuple[int, str]]:
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
    Two-part principal-angle analysis:

    Part 1 — FINAL COMPARISON
        For each QKV layer:
            PA( SALTEDORA_final,  full_ft_final )
        → {out_dir}/final_comparison_summary.csv

    Part 2 — EPOCH TRAJECTORY
        For each saved epoch e where both runs have a checkpoint:
            For each QKV layer:
                PA( SALTEDORA_epoch_e,  full_ft_epoch_e )
        → {out_dir}/epoch_trajectory_summary.csv
           {out_dir}/<layer>_trajectory_angles.npz
    """
    os.makedirs(out_dir, exist_ok=True)

    def _load(run_dir, fname):
        path = os.path.join(run_dir, "weights", fname)
        if not os.path.exists(path):
            logger.warning(f"Weight file not found: {path}")
            return None
        return torch.load(path, map_location="cpu")

    salty_final   = _load(saltedora_dir, "final.pt")
    full_ft_final = _load(full_ft_dir,   "final.pt")

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
            logger.warning(f"    Skipping {layer}")
            continue
        pa = compute_principal_angles(W_s, W_f, top_k)
        final_rows.append({
            "layer":          layer,
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

    salty_epochs   = {e: p for e, p in _list_epoch_files(
        os.path.join(saltedora_dir, "weights"))}
    full_ft_epochs = {e: p for e, p in _list_epoch_files(
        os.path.join(full_ft_dir,  "weights"))}

    common_epochs = sorted(set(salty_epochs.keys()) & set(full_ft_epochs.keys()))
    if not common_epochs:
        logger.warning("    No common epoch checkpoints — skipping trajectory.")
        return

    layer_traj_data: dict[str, dict] = {l: {} for l in layer_names}
    traj_rows = []

    for epoch in common_epochs:
        logger.info(f"    Epoch {epoch} …")
        salty_p   = torch.load(salty_epochs[epoch],   map_location="cpu")
        full_ft_p = torch.load(full_ft_epochs[epoch], map_location="cpu")

        for layer in layer_names:
            if layer not in salty_p["qkv"] or layer not in full_ft_p["qkv"]:
                continue
            W_s = reconstruct_effective_weight(salty_p["qkv"][layer])
            W_f = reconstruct_effective_weight(full_ft_p["qkv"][layer])
            if W_s is None or W_f is None:
                continue

            pa = compute_principal_angles(W_s, W_f, top_k)
            traj_rows.append({
                "epoch":          epoch,
                "layer":          layer,
                "mean_angle_deg": pa["mean_angle_deg"],
                "max_angle_deg":  pa["max_angle_deg"],
            })

            layer_traj_data[layer][f"epoch_{epoch}_angles_deg"] = pa["angles_deg"]
            layer_traj_data[layer][f"epoch_{epoch}_cosines"]    = pa["cosines"]

    # Add final angles to the per-layer npz data
    for layer in layer_names:
        if layer not in salty_final["qkv"] or layer not in full_ft_final["qkv"]:
            continue
        W_s = reconstruct_effective_weight(salty_final["qkv"][layer])
        W_f = reconstruct_effective_weight(full_ft_final["qkv"][layer])
        if W_s is not None and W_f is not None:
            pa = compute_principal_angles(W_s, W_f, top_k)
            layer_traj_data[layer]["final_angles_deg"] = pa["angles_deg"]
            layer_traj_data[layer]["final_cosines"]    = pa["cosines"]

    pd.DataFrame(traj_rows).to_csv(
        os.path.join(out_dir, "epoch_trajectory_summary.csv"), index=False)
    logger.info(f"    → saved epoch_trajectory_summary.csv ({len(traj_rows)} rows)")

    for layer, data in layer_traj_data.items():
        if not data:
            continue
        safe_name = layer.replace(".", "_")
        np.savez(os.path.join(out_dir, f"{safe_name}_trajectory_angles.npz"), **data)

    logger.info(f"  ✅ PA analysis complete → {out_dir}")


# ===========================================================================
# HP grid construction
# ===========================================================================
def build_hp_runs(sweep: str) -> list[tuple[float, int, float, int]]:
    """
    Returns a list of (lr, r, wd, bs) tuples.
    """
    if sweep == "lr_rank":
        return [
            (lr, r, DEFAULT_WD, DEFAULT_BS)
            for lr, r in itertools.product(LR_GRID, RANK_GRID)
        ]
    elif sweep == "wd_bs":
        return [
            (DEFAULT_LR, DEFAULT_RANK, wd, bs)
            for wd, bs in itertools.product(WD_GRID, BS_GRID)
        ]
    elif sweep == "all":
        logger.warning(
            f"⚠️  'all' sweep: {len(LR_GRID)*len(RANK_GRID)*len(WD_GRID)*len(BS_GRID)} "
            "runs — this is expensive!"
        )
        return list(itertools.product(LR_GRID, RANK_GRID, WD_GRID, BS_GRID))
    else:
        raise ValueError(f"Unknown sweep type: {sweep!r}. Choose lr_rank, wd_bs, or all.")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SALTEDORA-V4")
    parser.add_argument(
        "--sweep", choices=["lr_rank", "wd_bs", "all"], default="lr_rank",
        help="Which HP grid to run (default: lr_rank)"
    )
    args_cli = parser.parse_args()

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds = load_sst2(tokenizer)
    metrics_fn = build_accuracy_metric()

    # -----------------------------------------------------------------------
    # Step 1: Run full_ft reference baseline (once)
    # -----------------------------------------------------------------------
    logger.info("\n" + "="*64)
    logger.info("STEP 1: Full fine-tuning reference baseline")
    logger.info("="*64)

    full_ft_base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    full_ft_model = replace_qkv_with_adapter(full_ft_base_model, mode="full_ft")

    _, full_ft_dir = train_and_evaluate(
        full_ft_model, train_ds, val_ds, tokenizer, metrics_fn,
        mode="full_ft",
        lr=DEFAULT_LR, r=0, wd=DEFAULT_WD, bs=DEFAULT_BS,
        energy_threshold=None,
    )

    # -----------------------------------------------------------------------
    # Step 2: SALTEDORA HP sweep
    # -----------------------------------------------------------------------
    hp_runs = build_hp_runs(args_cli.sweep)
    logger.info(f"\n{'='*64}")
    logger.info(f"STEP 2: SALTEDORA-V4 HP sweep ({args_cli.sweep}) — {len(hp_runs)} runs")
    logger.info(f"{'='*64}")

    summary_rows = []

    for idx, (lr, r, wd, bs) in enumerate(hp_runs, 1):
        lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
        hp_tag = f"lr{lr_str}_r{r}_wd{wd}_bs{bs}"
        logger.info(f"\n──── Run {idx}/{len(hp_runs)}: {hp_tag} ────")

        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        model = replace_qkv_with_adapter(
            base_model, r=r, mode="saltedora_v4",
            energy_threshold=ENERGY_THRESH,
        )

        results, salty_dir = train_and_evaluate(
            model, train_ds, val_ds, tokenizer, metrics_fn,
            mode="saltedora_v4",
            lr=lr, r=r, wd=wd, bs=bs,
            energy_threshold=ENERGY_THRESH,
        )

        summary_rows.append({
            "sweep":            args_cli.sweep,
            "lr":               lr,
            "rank":             r,
            "weight_decay":     wd,
            "batch_size":       bs,
            "energy_threshold": ENERGY_THRESH,
            "accuracy":         results.get("eval_accuracy"),
            "eval_loss":        results.get("eval_loss"),
            "runtime_s":        results.get("runtime_total_s"),
            "trainable_params": results.get("trainable_params"),
        })

        # -------------------------------------------------------------------
        # Principal angles: this SALTEDORA config vs full_ft reference
        # -------------------------------------------------------------------
        pa_out = f"./results_v4/principal_angles/v4/hp_{hp_tag}"
        logger.info(f"  📐 Principal angles analysis …")
        analyze_principal_angles(
            saltedora_dir = salty_dir,
            full_ft_dir   = full_ft_dir,
            out_dir       = pa_out,
            top_k         = TOP_K_ANGLES,
        )

    # -----------------------------------------------------------------------
    # Global summary
    # -----------------------------------------------------------------------
    os.makedirs("./results_v4", exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    out_csv    = f"./results_v4/summary_v4_hp_tuning_{args_cli.sweep}.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"\n✅ Global summary → {out_csv}")
    print(summary_df.sort_values("accuracy", ascending=False).to_string(index=False))
