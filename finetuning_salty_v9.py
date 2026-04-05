"""
finetuning_salty_v9.py  —  SALTEDORA-V4 (Fixed Energy Cuts, r=64) vs Full Fine-Tuning
========================================================================================
Trains SALTEDORA-V4 with a fixed head fraction (r_top_override) at six different
energy cut values on SST-2, comparing each against the full fine-tuning baseline
using principal angles between their weight subspaces.

Energy cuts (r_top_override)
-----------------------------
    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Each value specifies what fraction of the total singular values are placed
    in the head subspace (SALT α/β scaling). The remaining fraction goes to the
    tail (EDoRA intrinsic rank update). For example, cut=0.6 means 60% of
    singular values are in the head.

Rank
----
    r_intrinsic = 64  (intrinsic rank of the EDoRA tail update)

Outputs
-------
results_v9/bert-base-uncased/full_ft/reference/glue_sst2/
    results.json  |  loss_per_batch.csv  |  loss_per_epoch.csv
    weights/epoch_<n>.pt  …  final.pt

results_v9/bert-base-uncased/saltedora_v4_cut_<X>/r_64/glue_sst2/
    results.json  |  loss_per_batch.csv  |  loss_per_epoch.csv
    weights/epoch_<n>.pt  …  final.pt

results_v9/principal_angles/cut_<X>/
    final_comparison_summary.csv
    epoch_trajectory_summary.csv
    <layer>_trajectory_angles.npz

results_v9/summary_v9_energy_cuts.csv
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

from models import SALTEdoraLinearV4

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
NUM_EPOCHS   = 5
LR           = 5e-5
WEIGHT_DECAY = 0.01
TRAIN_BS     = 16
EVAL_BS      = 64
TOP_K_ANGLES = 32

# Energy cut sweep — each value sets the head fraction directly
ENERGY_CUTS  = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

RESULTS_ROOT = "./results_v9"


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
def replace_qkv_with_adapter(model, mode: str, r_top_override: float | None = None):
    """
    mode="full_ft"      : unfreeze all parameters.
    mode="saltedora_v4" : freeze base, inject SALTEdoraLinearV4.
        r_top_override  : float in [0,1] — fraction of singular values in head.
                          Bypasses the knee algorithm entirely.
    """
    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        return model

    for p in model.parameters():
        p.requires_grad = False

    def _recurse(parent):
        for name, module in parent.named_children():
            if isinstance(module, nn.Linear) and any(
                k in name for k in ["query", "key", "value"]
            ):
                setattr(parent, name, SALTEdoraLinearV4(
                    module,
                    r_intrinsic    = RANK,
                    r_top_override = r_top_override,  # fixed fraction
                ))
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
    payload = {"qkv": {}}
    for fullname, module in model.named_modules():
        if _is_qkv(fullname):
            entry = {}
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                entry["weight"] = module.weight.detach().cpu()
            entry["state_dict_no_weight"] = {
                k: v.detach().cpu()
                for k, v in module.state_dict().items()
                if k != "weight"
            }
            payload["qkv"][fullname] = entry
    return payload


def reconstruct_effective_weight(entry: dict) -> torch.Tensor | None:
    sd = entry.get("state_dict_no_weight", {})
    W0 = entry.get("weight")
    is_saltedora = ("alpha" in sd and "D" in sd and "R" in sd and "U_top" in sd)

    if is_saltedora:
        sigma      = F.relu(sd["S_top"].float() * sd["alpha"].float() + sd["beta"].float())
        W_head     = sd["U_top"].float() @ torch.diag(sigma) @ sd["Vh_top"].float()
        W_tail_b   = sd["U_tail"].float() @ torch.diag(sd["S_tail"].float()) @ sd["Vh_tail"].float()
        d          = F.relu(sd["D"].float())
        delta_tail = (sd["U_tail_r"].float() @ torch.diag(sd["S_tail_r"].float())) \
                     @ torch.diag(d) @ sd["R"].float() @ sd["Vh_tail_r"].float()
        return (W_head + W_tail_b + delta_tail).cpu()
    elif W0 is not None:
        return W0.float()
    else:
        bw = sd.get("base.weight")
        return bw.float() if bw is not None else None


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
                       mode: str, output_dir: str,
                       r_top_override: float | None = None) -> dict:
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
    results["r_top_override"]   = r_top_override

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    acc = results.get("eval_accuracy", "N/A")
    cut_str = f"cut={r_top_override:.1f}" if r_top_override is not None else "full_ft"
    logger.info(
        f"  ✓ {cut_str} | r={RANK if mode != 'full_ft' else '-'} | "
        f"acc={acc:.4f if isinstance(acc, float) else acc} | "
        f"params={results['trainable_params']:,} | {results['runtime_total_s']:.0f}s"
    )
    return results


# ===========================================================================
# Principal angles
# ===========================================================================
def compute_principal_angles(W1: torch.Tensor, W2: torch.Tensor,
                              top_k: int = TOP_K_ANGLES) -> dict:
    U1, _, _ = torch.linalg.svd(W1.float(), full_matrices=False)
    U2, _, _ = torch.linalg.svd(W2.float(), full_matrices=False)
    k  = min(top_k, U1.shape[1], U2.shape[1])
    M  = U1[:, :k].T @ U2[:, :k]
    cos = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / torch.pi)
    return {
        "cosines":        cos.numpy(),
        "angles_deg":     ang.numpy(),
        "mean_angle_deg": float(ang.mean()),
        "max_angle_deg":  float(ang.max()),
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


def analyze_principal_angles(saltedora_dir: str, full_ft_dir: str,
                              out_dir: str, cut_tag: str) -> dict:
    """
    Returns summary dict with mean final PA across layers.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _load(run_dir, fname):
        p = os.path.join(run_dir, "weights", fname)
        return torch.load(p, map_location="cpu") if os.path.exists(p) else None

    s_final = _load(saltedora_dir, "final.pt")
    f_final = _load(full_ft_dir,   "final.pt")
    if s_final is None or f_final is None:
        logger.error(f"[{cut_tag}] final.pt missing — skipping PA.")
        return {}

    layer_names = list(s_final["qkv"].keys())

    # ---- Part 1: FINAL ----
    logger.info(f"  [{cut_tag}] 📐 final comparison …")
    final_rows = []
    for layer in layer_names:
        Ws = reconstruct_effective_weight(s_final["qkv"][layer])
        Wf = reconstruct_effective_weight(f_final["qkv"][layer])
        if Ws is None or Wf is None:
            continue
        pa = compute_principal_angles(Ws, Wf)
        final_rows.append({"cut": cut_tag, "layer": layer,
                            "mean_angle_deg": pa["mean_angle_deg"],
                            "max_angle_deg":  pa["max_angle_deg"]})
    pd.DataFrame(final_rows).to_csv(
        os.path.join(out_dir, "final_comparison_summary.csv"), index=False)

    mean_final_pa = float(pd.DataFrame(final_rows)["mean_angle_deg"].mean()) \
        if final_rows else float("nan")

    # ---- Part 2: EPOCH TRAJECTORY ----
    logger.info(f"  [{cut_tag}] 📐 epoch trajectory …")
    s_epochs = dict(_list_epoch_files(os.path.join(saltedora_dir, "weights")))
    f_epochs = dict(_list_epoch_files(os.path.join(full_ft_dir,   "weights")))
    common   = sorted(set(s_epochs) & set(f_epochs))

    traj_rows = []
    layer_data: dict[str, dict] = {l: {} for l in layer_names}

    for epoch in common:
        sp = torch.load(s_epochs[epoch], map_location="cpu")
        fp = torch.load(f_epochs[epoch], map_location="cpu")
        for layer in layer_names:
            if layer not in sp["qkv"] or layer not in fp["qkv"]:
                continue
            Ws = reconstruct_effective_weight(sp["qkv"][layer])
            Wf = reconstruct_effective_weight(fp["qkv"][layer])
            if Ws is None or Wf is None:
                continue
            pa = compute_principal_angles(Ws, Wf)
            traj_rows.append({"cut": cut_tag, "epoch": epoch, "layer": layer,
                               "mean_angle_deg": pa["mean_angle_deg"],
                               "max_angle_deg":  pa["max_angle_deg"]})
            layer_data[layer][f"epoch_{epoch}_angles_deg"] = pa["angles_deg"]
            layer_data[layer][f"epoch_{epoch}_cosines"]    = pa["cosines"]

    # Add final
    for layer in layer_names:
        Ws = reconstruct_effective_weight(s_final["qkv"].get(layer, {}))
        Wf = reconstruct_effective_weight(f_final["qkv"].get(layer, {}))
        if Ws is not None and Wf is not None:
            pa = compute_principal_angles(Ws, Wf)
            layer_data[layer]["final_angles_deg"] = pa["angles_deg"]
            layer_data[layer]["final_cosines"]    = pa["cosines"]

    pd.DataFrame(traj_rows).to_csv(
        os.path.join(out_dir, "epoch_trajectory_summary.csv"), index=False)

    for layer, data in layer_data.items():
        if data:
            np.savez(os.path.join(out_dir,
                     f"{layer.replace('.', '_')}_trajectory_angles.npz"), **data)

    logger.info(f"  [{cut_tag}] ✅ PA done → {out_dir}")
    return {"mean_final_pa_deg": mean_final_pa}


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds = load_sst2(tokenizer)
    metrics_fn = build_accuracy_metric()

    # ---- Step 1: Full fine-tuning reference ----
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Full fine-tuning reference")
    logger.info("="*60)

    full_ft_dir = f"{RESULTS_ROOT}/{MODEL_NAME}/full_ft/reference/glue_sst2"
    full_ft_model = replace_qkv_with_adapter(
        AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2),
        mode="full_ft",
    )
    full_ft_results = train_and_evaluate(
        full_ft_model, train_ds, val_ds, tokenizer, metrics_fn,
        mode="full_ft", output_dir=full_ft_dir,
    )

    # ---- Step 2: Sweep energy cuts ----
    total_cuts  = len(ENERGY_CUTS)
    summary_rows = [{
        "mode":             "full_ft",
        "r_top_override":   None,
        "rank":             0,
        "accuracy":         full_ft_results.get("eval_accuracy"),
        "trainable_params": full_ft_results.get("trainable_params"),
        "runtime_s":        full_ft_results.get("runtime_total_s"),
        "mean_final_pa_deg": None,
    }]

    for idx, cut in enumerate(ENERGY_CUTS, 1):
        cut_tag = f"{cut:.1f}"
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP 2.{idx}/{total_cuts}: SALTEDORA-V4  r={RANK}  cut={cut_tag}")
        logger.info("="*60)

        salty_dir = (
            f"{RESULTS_ROOT}/{MODEL_NAME}/saltedora_v4_cut_{cut_tag}/r_{RANK}/glue_sst2"
        )
        salty_model = replace_qkv_with_adapter(
            AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2),
            mode="saltedora_v4",
            r_top_override=cut,
        )
        salty_results = train_and_evaluate(
            salty_model, train_ds, val_ds, tokenizer, metrics_fn,
            mode="saltedora_v4", output_dir=salty_dir,
            r_top_override=cut,
        )

        # PA analysis for this cut
        pa_out = f"{RESULTS_ROOT}/principal_angles/cut_{cut_tag}"
        pa_summary = analyze_principal_angles(salty_dir, full_ft_dir, pa_out, cut_tag)

        summary_rows.append({
            "mode":              "saltedora_v4",
            "r_top_override":    cut,
            "rank":              RANK,
            "accuracy":          salty_results.get("eval_accuracy"),
            "trainable_params":  salty_results.get("trainable_params"),
            "runtime_s":         salty_results.get("runtime_total_s"),
            "mean_final_pa_deg": pa_summary.get("mean_final_pa_deg"),
        })

    # ---- Summary CSV ----
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    out_csv    = f"{RESULTS_ROOT}/summary_v9_energy_cuts.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"\n✅ Done. Summary → {out_csv}")
    print(summary_df.to_string(index=False))
