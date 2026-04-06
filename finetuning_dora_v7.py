"""
finetuning_dora_v7.py  —  DoRA (r=64) vs Full Fine-Tuning
==========================================================
Trains DoRA with rank 64 and full fine-tuning on SST-2, then computes
principal angles between their weight subspaces at each epoch and at
the final checkpoint.  Mirrors finetuning_salty_v7.py exactly so results
are directly comparable.

DoRA (Weight-Decomposed Low-Rank Adaptation)
--------------------------------------------
    W_eff = m * (W0 + scaling * B @ A) / ||(W0 + scaling * B @ A)||_row
    where
        W0       — frozen pretrained weight
        A ∈ R^{r×in},  B ∈ R^{out×r}  — learned low-rank matrices
        m ∈ R^{out}   — learned per-output magnitude vector
        scaling  = alpha / r

Outputs
-------
results_dora_v7/bert-base-uncased/full_ft/reference/glue_sst2/
    results.json  |  loss_per_batch.csv  |  loss_per_epoch.csv
    weights/epoch_<n>.pt  …  final.pt

results_dora_v7/bert-base-uncased/dora/r_64/glue_sst2/
    results.json  |  loss_per_batch.csv  |  loss_per_epoch.csv
    weights/epoch_<n>.pt  …  final.pt

results_dora_v7/principal_angles/
    final_comparison_summary.csv
    epoch_trajectory_summary.csv
    <layer>_trajectory_angles.npz

results_dora_v7/summary_dora_v7.csv
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME   = "bert-base-uncased"
TASK_KEY     = "glue/sst2"
RANK         = 64
ALPHA        = 64          # scaling = alpha / r = 1.0
NUM_EPOCHS   = 5
LR           = 1e-4
WEIGHT_DECAY = 0.1
TRAIN_BS     = 16
EVAL_BS      = 64
TOP_K_ANGLES = 32

RESULTS_ROOT = "./results_dora_v7"


# ===========================================================================
# DoRA linear layer
# ===========================================================================
class DoRALinear(nn.Module):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).

    W_eff = m * V / ||V||_row,   V = W0 + scaling * B @ A

    Parameters
    ----------
    module  : pretrained nn.Linear to replace
    r       : intrinsic rank
    alpha   : LoRA-style scaling factor (scaling = alpha / r)
    """

    def __init__(self, module: nn.Linear, r: int = 64, alpha: float = 64.0):
        super().__init__()
        out_features = module.out_features
        in_features  = module.in_features

        # Freeze pretrained weight as a buffer (not a parameter)
        self.register_buffer("W0", module.weight.data.clone().float())
        if module.bias is not None:
            self.register_buffer("bias", module.bias.data.clone())
        else:
            self.bias = None

        self.scaling = alpha / r

        # Low-rank matrices  (standard LoRA initialisation)
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        # Magnitude vector — initialised to row-norms of W0
        with torch.no_grad():
            m_init = module.weight.data.float().norm(dim=1)   # shape: (out_features,)
        self.magnitude = nn.Parameter(m_init)

        logger.debug(f"  DoRALinear  r={r}  in={in_features}  out={out_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # V = W0 + low-rank update
        V     = self.W0 + self.scaling * (self.B @ self.A)
        # Normalise each row to unit length
        V_norm = V / (V.norm(dim=1, keepdim=True).clamp_min(1e-8))
        # Scale by learned magnitude
        W_eff  = self.magnitude.unsqueeze(1) * V_norm
        return F.linear(x, W_eff.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


# ===========================================================================
# Dataset
# ===========================================================================
def load_sst2(tokenizer):
    raw = load_dataset("glue", "sst2")

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"],
                         truncation=True, padding="max_length", max_length=128)

    tok = raw.map(tokenize_fn, batched=True)
    tok = tok.rename_column("label", "labels")
    tok.set_format("torch")
    return tok["train"], tok["validation"]


def build_accuracy_metric():
    acc = evaluate.load("accuracy")

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        r = acc.compute(predictions=preds, references=labels)
        return {"eval_accuracy": r["accuracy"], "accuracy": r["accuracy"]}

    return compute


# ===========================================================================
# Adapter replacement
# ===========================================================================
def replace_qkv_with_adapter(model, mode: str):
    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        return model

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    def _recurse(parent):
        for name, module in parent.named_children():
            if isinstance(module, nn.Linear) and any(
                k in name for k in ["query", "key", "value"]
            ):
                setattr(parent, name, DoRALinear(module, r=RANK, alpha=ALPHA))
            elif list(module.children()):
                _recurse(module)

    _recurse(model)
    return model


# ===========================================================================
# Weight extraction & reconstruction
# ===========================================================================
def _is_qkv(name: str) -> bool:
    return any(name.endswith(s) for s in [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
    ])


def extract_qkv_payload(model: nn.Module) -> dict:
    """Save per-layer state dicts for all QKV modules."""
    payload = {"qkv": {}}
    for fullname, module in model.named_modules():
        if _is_qkv(fullname):
            entry = {}
            # Separate the base weight from learned parameters
            if hasattr(module, "W0"):
                entry["W0"] = module.W0.detach().cpu()
            elif hasattr(module, "weight"):
                entry["weight"] = module.weight.detach().cpu()
            entry["state_dict_no_base"] = {
                k: v.detach().cpu()
                for k, v in module.state_dict().items()
                if k not in ("W0", "weight")
            }
            # Store scaling as a plain float alongside
            if hasattr(module, "scaling"):
                entry["scaling"] = module.scaling
            payload["qkv"][fullname] = entry
    return payload


def reconstruct_effective_weight(entry: dict) -> torch.Tensor | None:
    """Rebuild W_eff from a saved payload entry."""
    sd      = entry.get("state_dict_no_base", {})
    W0      = entry.get("W0")
    weight  = entry.get("weight")
    scaling = entry.get("scaling", 1.0)

    is_dora = ("magnitude" in sd and "A" in sd and "B" in sd)

    if is_dora and W0 is not None:
        V      = W0.float() + scaling * (sd["B"].float() @ sd["A"].float())
        V_norm = V / (V.norm(dim=1, keepdim=True).clamp_min(1e-8))
        W_eff  = sd["magnitude"].float().unsqueeze(1) * V_norm
        return W_eff.cpu()
    elif weight is not None:
        return weight.float()
    elif W0 is not None:
        return W0.float()
    else:
        return None


# ===========================================================================
# Callbacks
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
        total = time.time() - self.train_start_time if self.train_start_time else None
        os.makedirs(self.output_dir, exist_ok=True)
        if self.batch_logs:
            pd.DataFrame(self.batch_logs).to_csv(
                os.path.join(self.output_dir, "loss_per_batch.csv"), index=False)
        df = pd.DataFrame(self.epoch_logs)
        df["total_runtime_s"] = total
        df.to_csv(os.path.join(self.output_dir, "loss_per_epoch.csv"), index=False)


class WeightSnapshotCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.weights_dir = os.path.join(output_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or state.epoch is None:
            return
        epoch_int = int(round(state.epoch))
        path = os.path.join(self.weights_dir, f"epoch_{epoch_int}.pt")
        torch.save(extract_qkv_payload(model), path)
        logger.info(f"  💾 Saved QKV snapshot → epoch_{epoch_int}.pt")

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        path = os.path.join(self.weights_dir, "final.pt")
        torch.save(extract_qkv_payload(model), path)
        logger.info(f"  💾 Saved QKV snapshot → final.pt")


# ===========================================================================
# Training
# ===========================================================================
def train_and_evaluate(model, train_ds, val_ds, tokenizer, metrics_fn,
                       mode: str, output_dir: str) -> dict:
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

    model.config.num_labels   = 2
    model.config.problem_type = "single_label_classification"

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        compute_metrics = metrics_fn,
        callbacks       = [TrackingCallback(output_dir), WeightSnapshotCallback(output_dir)],
    )

    start   = time.time()
    trainer.train()
    results = trainer.evaluate(val_ds)
    results["runtime_total_s"]  = time.time() - start
    results["trainable_params"] = count_trainable_params(model)
    results["mode"]             = mode
    results["rank"]             = RANK if mode != "full_ft" else 0

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    acc     = results.get("eval_accuracy", "N/A")
    acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
    logger.info(
        f"  ✓ {mode} | r={RANK if mode != 'full_ft' else '-'} | "
        f"acc={acc_str} | "
        f"params={results['trainable_params']:,} | {results['runtime_total_s']:.0f}s"
    )
    return results


# ===========================================================================
# Principal angles
# ===========================================================================
def compute_principal_angles(W1: torch.Tensor, W2: torch.Tensor,
                              top_k: int = TOP_K_ANGLES) -> dict:
    U1, S1, _ = torch.linalg.svd(W1.float(), full_matrices=False)
    U2, _,  _ = torch.linalg.svd(W2.float(), full_matrices=False)
    k   = min(top_k, U1.shape[1], U2.shape[1])

    # Cross-Gram matrix — pairwise cosines between directions
    M   = U1[:, :k].T @ U2[:, :k]

    # SVD of M finds the optimal direction pairing; singular values = cos(θᵢ)
    cos = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / torch.pi)

    # Singular-value weighted mean: upweight high-energy directions of W1
    w             = S1[:k]
    w             = w / w.sum()
    weighted_mean = float((w * ang).sum())

    return {
        "cosines":           cos.numpy(),
        "angles_deg":        ang.numpy(),
        "mean_angle_deg":    float(ang.mean()),
        "weighted_mean_deg": weighted_mean,
        "max_angle_deg":     float(ang.max()),
        "min_angle_deg":     float(ang.min()),
    }


def _list_epoch_files(weights_dir: str) -> list[tuple[int, str]]:
    pairs = []
    if not os.path.isdir(weights_dir):
        return pairs
    for fname in os.listdir(weights_dir):
        if fname.startswith("epoch_") and fname.endswith(".pt"):
            try:
                pairs.append((int(fname[6:-3]), os.path.join(weights_dir, fname)))
            except ValueError:
                pass
    return sorted(pairs)


def analyze_principal_angles(dora_dir: str, full_ft_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _load(run_dir, fname):
        p = os.path.join(run_dir, "weights", fname)
        return torch.load(p, map_location="cpu") if os.path.exists(p) else None

    d_final = _load(dora_dir,    "final.pt")
    f_final = _load(full_ft_dir, "final.pt")
    if d_final is None or f_final is None:
        logger.error("final.pt missing — aborting PA analysis.")
        return

    layer_names = list(d_final["qkv"].keys())

    # ---- Part 1: FINAL ----
    logger.info("  📐 PA — final comparison …")
    final_rows = []
    for layer in layer_names:
        Wd = reconstruct_effective_weight(d_final["qkv"][layer])
        Wf = reconstruct_effective_weight(f_final["qkv"][layer])
        if Wd is None or Wf is None:
            continue
        pa = compute_principal_angles(Wd, Wf)
        final_rows.append({
            "layer":             layer,
            "mean_angle_deg":    pa["mean_angle_deg"],
            "weighted_mean_deg": pa["weighted_mean_deg"],
            "max_angle_deg":     pa["max_angle_deg"],
            "min_angle_deg":     pa["min_angle_deg"],
        })
    pd.DataFrame(final_rows).to_csv(
        os.path.join(out_dir, "final_comparison_summary.csv"), index=False)
    logger.info(f"    → final_comparison_summary.csv ({len(final_rows)} layers)")

    # ---- Part 2: EPOCH TRAJECTORY ----
    logger.info("  📐 PA — epoch trajectory …")
    d_epochs = dict(_list_epoch_files(os.path.join(dora_dir,    "weights")))
    f_epochs = dict(_list_epoch_files(os.path.join(full_ft_dir, "weights")))
    common   = sorted(set(d_epochs) & set(f_epochs))

    traj_rows  = []
    layer_data: dict[str, dict] = {l: {} for l in layer_names}

    for epoch in common:
        dp = torch.load(d_epochs[epoch], map_location="cpu")
        fp = torch.load(f_epochs[epoch], map_location="cpu")
        for layer in layer_names:
            if layer not in dp["qkv"] or layer not in fp["qkv"]:
                continue
            Wd = reconstruct_effective_weight(dp["qkv"][layer])
            Wf = reconstruct_effective_weight(fp["qkv"][layer])
            if Wd is None or Wf is None:
                continue
            pa = compute_principal_angles(Wd, Wf)
            traj_rows.append({
                "epoch":             epoch,
                "layer":             layer,
                "mean_angle_deg":    pa["mean_angle_deg"],
                "weighted_mean_deg": pa["weighted_mean_deg"],
                "max_angle_deg":     pa["max_angle_deg"],
            })
            layer_data[layer][f"epoch_{epoch}_angles_deg"] = pa["angles_deg"]
            layer_data[layer][f"epoch_{epoch}_cosines"]    = pa["cosines"]

    # Add final checkpoint to per-layer npz
    for layer in layer_names:
        Wd = reconstruct_effective_weight(d_final["qkv"].get(layer, {}))
        Wf = reconstruct_effective_weight(f_final["qkv"].get(layer, {}))
        if Wd is not None and Wf is not None:
            pa = compute_principal_angles(Wd, Wf)
            layer_data[layer]["final_angles_deg"] = pa["angles_deg"]
            layer_data[layer]["final_cosines"]    = pa["cosines"]

    pd.DataFrame(traj_rows).to_csv(
        os.path.join(out_dir, "epoch_trajectory_summary.csv"), index=False)
    logger.info(f"    → epoch_trajectory_summary.csv ({len(traj_rows)} rows)")

    for layer, data in layer_data.items():
        if data:
            np.savez(os.path.join(out_dir,
                     f"{layer.replace('.', '_')}_trajectory_angles.npz"), **data)

    logger.info(f"  ✅ PA analysis complete → {out_dir}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    tokenizer            = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds     = load_sst2(tokenizer)
    metrics_fn           = build_accuracy_metric()

    # ---- Step 1: Full fine-tuning reference ----
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Full fine-tuning reference")
    logger.info("="*60)

    full_ft_dir   = f"{RESULTS_ROOT}/{MODEL_NAME}/full_ft/reference/glue_sst2"
    full_ft_model = replace_qkv_with_adapter(
        AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2),
        mode="full_ft",
    )
    full_ft_results = train_and_evaluate(
        full_ft_model, train_ds, val_ds, tokenizer, metrics_fn,
        mode="full_ft", output_dir=full_ft_dir,
    )

    # ---- Step 2: DoRA (r=64) ----
    logger.info("\n" + "="*60)
    logger.info(f"STEP 2: DoRA  r={RANK}  alpha={ALPHA}  scaling={ALPHA/RANK:.2f}")
    logger.info("="*60)

    dora_dir   = f"{RESULTS_ROOT}/{MODEL_NAME}/dora/r_{RANK}/glue_sst2"
    dora_model = replace_qkv_with_adapter(
        AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2),
        mode="dora",
    )
    dora_results = train_and_evaluate(
        dora_model, train_ds, val_ds, tokenizer, metrics_fn,
        mode="dora", output_dir=dora_dir,
    )

    # ---- Step 3: Principal angles ----
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Principal angles analysis")
    logger.info("="*60)

    pa_dir = f"{RESULTS_ROOT}/principal_angles"
    analyze_principal_angles(dora_dir, full_ft_dir, pa_dir)

    # ---- Summary CSV ----
    summary = pd.DataFrame([
        {"mode": "full_ft", "rank": 0,    "alpha": None,
         "accuracy":          full_ft_results.get("eval_accuracy"),
         "trainable_params":  full_ft_results.get("trainable_params"),
         "runtime_s":         full_ft_results.get("runtime_total_s")},
        {"mode": "dora",    "rank": RANK, "alpha": ALPHA,
         "accuracy":          dora_results.get("eval_accuracy"),
         "trainable_params":  dora_results.get("trainable_params"),
         "runtime_s":         dora_results.get("runtime_total_s")},
    ])
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    summary.to_csv(f"{RESULTS_ROOT}/summary_dora_v7.csv", index=False)
    logger.info(f"\n✅ Done. Summary → {RESULTS_ROOT}/summary_dora_v7.csv")
    print(summary.to_string(index=False))
