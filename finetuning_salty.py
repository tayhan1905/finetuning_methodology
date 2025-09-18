import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import json
import os
import pandas as pd
import itertools
import logging
from datetime import datetime
# ----------------------------
# Adapter Definitions
# ----------------------------

class LoRA(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.A = nn.Parameter(torch.randn(self.in_features, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, self.out_features) * 0.01)

    def forward(self, x):
        return F.linear(x, self.base.weight, self.base.bias) + F.linear(x, self.A @ self.B)


class DoRA(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        # magnitude and direction params
        self.magnitude = nn.Parameter(torch.zeros(self.in_features))
        self.direction = nn.Parameter(torch.randn(self.in_features, self.out_features) * 0.01)

    def forward(self, x):
        direction_norm = F.normalize(self.direction, p=2, dim=0)
        delta_w = self.magnitude.unsqueeze(0) * direction_norm
        return F.linear(x, self.base.weight, self.base.bias) + F.linear(x, delta_w)

def truncated_svd(W, rank):
    # thin SVD; rank <= min(out, in)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = min(rank, S.numel())
    return U[:, :r].contiguous(), S[:r].contiguous(), Vh[:r, :].contiguous()

def svd_head_tail(W, r):
    # return top-r and bottom-r blocks (disjoint if 2r <= p)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    p = S.numel()
    r_top = min(r, p)
    r_bot = min(r, p - r_top) if (2*r <= p) else min(r, max(0, p - r_top))
    U_top, S_top, Vh_top = U[:, :r_top], S[:r_top], Vh[:r_top, :]
    U_bot, S_bot, Vh_bot = U[:, p-r_bot:], S[p-r_bot:], Vh[p-r_bot:, :]
    return (U_top, S_top, Vh_top), (U_bot, S_bot, Vh_bot)

class SaltEdoraLinear(nn.Module):
    """
    Additive adapter around a frozen Linear:
    y = x @ W^T + SALT_top(x) + eDoRA_tail(x)
    - SALT: scale/shift top-r singulars (α, β)
    - eDoRA: r x r core R in tail subspace
    """
    def __init__(self, base_linear: nn.Linear, r: int, tail_mode='free'):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias = base_linear.bias is not None

        ## Freezing the parameters from the pre trained model
        # freeze the original weights that do not require training
        self.base.weight.requires_grad_(False) 
        # freeze the original bias that do not need training as well
        if self.bias:
            self.base.bias.requires_grad_(False)

        # SVD to provide some singular understanding of the pre trained matrix to guide our 
        W = self.base.weight.detach().to(torch.float32)
        (U_top, S_top, Vh_top), (U_bot, S_bot, Vh_bot) = svd_head_tail(W, r)
        dtype = self.base.weight.dtype
        device = self.base.weight.device

        # store basis as buffers >> saved as like self.U_top >>based on the string that is provided
        self.register_buffer("U_top", U_top.to(dtype).to(device))
        self.register_buffer("S_top", S_top.to(dtype).to(device))
        self.register_buffer("Vh_top", Vh_top.to(dtype).to(device))
        self.register_buffer("U_bot", U_bot.to(dtype).to(device))
        self.register_buffer("S_bot", S_bot.to(dtype).to(device))
        self.register_buffer("Vh_bot", Vh_bot.to(dtype).to(device))

        # Understanding the total ranks of each of the weight matrices (top and tail)
        self.r_top = S_top.numel()
        self.r_bot = S_bot.numel()

        # SALT params >> this is for the scale shift parameters
        self.alpha = nn.Parameter(torch.zeros(self.r_top, dtype=dtype, device=device))
        self.beta  = nn.Parameter(torch.zeros(self.r_top, dtype=dtype, device=device))

        # eDoRA core >> free and polar
        ## free >> the LORA style, no segregation of magnitude and directionality in this case
        ## polar >> the DORA style, clear seegregation of magnitude and directionality
        self.tail_mode = tail_mode
        if self.r_bot > 0:
            if tail_mode == 'free':
                R0 = torch.zeros(self.r_bot, self.r_bot, dtype=dtype, device=device)
                self.R = nn.Parameter(R0)  # neutral (no delta)
            else:
                raise ValueError("tail_mode must be 'free'")

    def forward(self, x):
        # base
        y = F.linear(x, self.base.weight, self.base.bias) # y = x W^T + b

        # SALT head delta
        if self.r_top > 0:
            zH = F.linear(x, self.Vh_top) # zH = x V_top^T

            # Computation of the top singular values as scale shifts
            delta_sigma = self.S_top * self.alpha + self.beta  # Δσ = S_top ⊙ α + β
            y = y + F.linear(zH * delta_sigma, self.U_top) # y ← y + U_top diag(Δσ) V_top^T x

        # eDoRA tail delta >> utilising the rxr matrix to run the update 
        if self.r_bot > 0:
            zT = F.linear(x, self.Vh_bot) # zT = x V_bot^T                   
            zR = F.linear(zT, self.R.T) # zR = zT R^T
            y = y + F.linear(zR, self.U_bot)
        return y
        

class SALT(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        W = self.base.weight.detach().to(torch.float32)
        U, S, Vh = torch.svd(W)

        self.register_buffer("U", U)
        self.register_buffer("S", S)
        self.register_buffer("Vh", Vh)

        self.alpha = nn.Parameter(torch.zeros(r, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(r, dtype=torch.float32))

    def forward(self, x):
        delta_sigma = self.S[:self.alpha.size(0)] * self.alpha + self.beta
        update = self.U[:, :self.alpha.size(0)] @ torch.diag(delta_sigma) @ self.Vh[:self.alpha.size(0), :]
        return F.linear(x, self.base.weight + update, self.base.bias)


# ----------------------------
# Replacement function
# ----------------------------

def replace_qkv_with_adapter(model, r=8, mode="lora"):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and ("query" in name or "key" in name or "value" in name):
            if mode == "lora":
                setattr(model, name, LoRA(module, r=r))
            elif mode == "dora":
                setattr(model, name, DoRA(module, r=r))
            elif mode == "saltedora":
                setattr(model, name, SaltEdoraLinear(module, r=r, tail_mode='free'))
            elif mode == "salt":
                setattr(model, name, SALT(module, r=r))
        else:
            replace_qkv_with_adapter(module, r=r, mode=mode)
    return model


# ----------------------------
# Dataset setup
# ----------------------------

def get_dataset_and_tokenizer(model_name="bert-base-uncased"):
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized["train"], tokenized["validation"], tokenizer


# ----------------------------
# Training utilities
# ----------------------------

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return accuracy.compute(predictions=preds, references=labels)


def train_and_evaluate(model, train_ds, val_ds, model_name, r, mode):
    args = TrainingArguments(
        output_dir=f"./results/{model_name}/{mode}/r_{r}",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=1,  # keep small for quick test, increase as needed
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate(val_ds)

    # Save results in a JSON file
    result_path = f"./results/{model_name}/{mode}/r_{r}/results.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


# ----------------------------
# Main experiment runner
# ----------------------------

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    train_ds, val_ds, tokenizer = get_dataset_and_tokenizer(model_name)

    ranks = [4, 8, 16, 32, 64]  # List of ranks to experiment with
    modes = ["lora", "dora", "saltedora", "salt"]  # The adapter types

    results = []

    logger.info("Starting experiments...")

    # Loop over rank and model modes
    for r, mode in itertools.product(ranks, modes):
        logger.info(f"Running experiment for {mode} with rank {r}...")
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model = replace_qkv_with_adapter(base_model, r=r, mode=mode)
        
        # Log the model configuration before training
        logger.info(f"Model: {mode}, Rank: {r}, Training started.")
        
        result = train_and_evaluate(model, train_ds, val_ds, model_name, r, mode)
        results.append({"rank": r, "mode": mode, "results": result})

    logger.info("Experiments completed. Generating results table...")

    # Convert results to a DataFrame for easy analysis
    result_table = []
    for res in results:
        rank = res["rank"]
        mode = res["mode"]
        accuracy = res["results"].get("eval_accuracy", None)
        result_table.append([rank, mode, accuracy])

    df = pd.DataFrame(result_table, columns=["Rank", "Mode", "Accuracy"])

    # Save the result table to a CSV file for further analysis
    os.makedirs("./results", exist_ok=True)
    df.to_csv("./results/comparison_table.csv", index=False)

    # Log the final result table
    logger.info("Results table saved to ./results/comparison_table.csv")

    # Print out the table
    print(df)