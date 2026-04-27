"""
Fine-tune Qwen3-8B on Pokemon Showdown replay data using LoRA.

Trains the model to predict battle actions (move/switch) given battle state context.
Uses parameter-efficient fine-tuning (LoRA) so it fits on a single GPU.

Usage:
    # Basic training on collected replay data
    python train_sft.py --data sft_data.jsonl --epochs 3

    # With custom settings
    python train_sft.py --data sft_data.jsonl --epochs 5 --lr 2e-4 --batch-size 4

    # Resume from checkpoint
    python train_sft.py --data sft_data.jsonl --resume checkpoints/sft_qwen/checkpoint-500

    # Merge LoRA weights into full model after training
    python train_sft.py --merge checkpoints/sft_qwen/final

Requirements (install once):
    pip install peft trl datasets
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3-8B")
CHECKPOINT_DIR = Path("checkpoints/sft_qwen")

SYSTEM_PROMPT = """You are a competitive Pokemon battle AI. Given the current battle state, choose the best action.

Respond with JSON: {"action_type": "move" or "switch", "target_name": "<move or pokemon name>"}

DECISION PRIORITIES:
1. KO THREAT: If you can KO the opponent, ATTACK. Do not switch.
2. SURVIVE: If they can KO you and you can't KO them, switch to a better matchup.
3. ATTACK WHEN AHEAD: If you deal major damage and they can't OHKO, keep attacking.
4. MOMENTUM: Use the best damaging move or pivot (U-turn, Volt Switch).
5. SETUP: Only set up if the opponent can't threaten you."""


# ── Data loading ─────────────────────────────────────────────────────────────

def load_sft_data(path: str, max_samples: int = 0) -> Dataset:
    """Load JSONL and format as chat-style training data."""
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            })
    if max_samples > 0:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples):,} training samples from {path}")
    return Dataset.from_list(samples)


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    print(f"\n{'='*50}")
    print(f"  SFT Training: {args.model}")
    print(f"{'='*50}")
    print(f"  Data:       {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  LoRA rank:  {args.lora_rank}")

    # Device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"  Device:     {device} ({dtype})")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Load data
    dataset = load_sft_data(args.data, max_samples=args.max_samples)

    # Split 95/5 for eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    # Output dir
    output_dir = str(CHECKPOINT_DIR)
    if args.resume:
        output_dir = str(Path(args.resume).parent)

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
        max_length=2048,
        remove_unused_columns=False,
        dataset_text_field="text",
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=42,
    )

    # Format dataset for chat template
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_ds = train_ds.map(format_chat, remove_columns=["messages"])
    eval_ds = eval_ds.map(format_chat, remove_columns=["messages"])

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        print("\nStarting training...")
        trainer.train()

    # Save final
    final_path = CHECKPOINT_DIR / "final"
    print(f"\nSaving final LoRA adapter to {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print("Done!")

    # Print eval results
    results = trainer.evaluate()
    print(f"\nFinal eval loss: {results.get('eval_loss', '?'):.4f}")


# ── Merge LoRA into base model ───────────────────────────────────────────────

def merge(adapter_path: str):
    """Merge LoRA weights into base model for faster inference."""
    from peft import PeftModel

    print(f"Merging LoRA adapter: {adapter_path}")

    # Load adapter config to get base model name
    adapter_config = json.load(open(Path(adapter_path) / "adapter_config.json"))
    base_model_name = adapter_config.get("base_model_name_or_path", DEFAULT_MODEL)
    print(f"  Base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    out_path = Path(adapter_path).parent / "merged"
    print(f"  Saving merged model to {out_path}")
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print("Done! Use this model path for inference.")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT fine-tune Qwen on Pokemon replay data")
    sub = parser.add_subparsers(dest="command")

    # Train subcommand (default)
    train_p = sub.add_parser("train", help="Fine-tune with LoRA")
    train_p.add_argument("--data", default="sft_data.jsonl", help="JSONL training data")
    train_p.add_argument("--model", default=DEFAULT_MODEL, help="Base model")
    train_p.add_argument("--epochs", type=int, default=3)
    train_p.add_argument("--batch-size", type=int, default=2)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--lora-rank", type=int, default=16)
    train_p.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=all)")
    train_p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Merge subcommand
    merge_p = sub.add_parser("merge", help="Merge LoRA into base model")
    merge_p.add_argument("adapter_path", help="Path to LoRA adapter")

    args = parser.parse_args()

    if args.command == "merge":
        merge(args.adapter_path)
    elif args.command == "train":
        train(args)
    else:
        # Default: treat all args as train args
        parser = argparse.ArgumentParser(description="SFT fine-tune Qwen")
        parser.add_argument("--data", default="sft_data.jsonl")
        parser.add_argument("--model", default=DEFAULT_MODEL)
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch-size", type=int, default=2)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--lora-rank", type=int, default=16)
        parser.add_argument("--max-samples", type=int, default=0)
        parser.add_argument("--resume", type=str, default=None)
        args = parser.parse_args()
        train(args)
