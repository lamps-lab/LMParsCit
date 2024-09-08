#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 02:34:50 2024

@author: muntabir
"""

import pandas as pd
import json
import torch
from datasets import Dataset
from transformers import TrainingArguments

from unsloth import FastLanguageModel
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

import gc

## Training dataset
df = pd.read_csv("./../data-giant/input/citation_data-GIANT10K-llama3.csv")

dataset = Dataset.from_pandas(df)
#print(dataset)

max_seq_length = 1056 # Choose any! They auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,

    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True, # True or "unsloth" for very long context
    use_rslora = False,  # We support rank stabilized LoRA
    use_dora = False,   # We support dynamic LoRA
    random_state = 3407, # for reproducibility
    loftq_config = None, # And LoftQ
)


"""## Finetuning and Training"""

print("Llama-3-8b-instruct Finetuning and Training......")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 40,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(), # will use 16-bit precision for its computations, and speed up training and reduce memory usage, allowing for larger batch sizes or models to fit in memory
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./../llama3-lmparsCit-model/LMParsCit_output_10K",
        eval_strategy = "steps",  # Evaluate during training
        eval_steps = 10,  # Evaluate every 10 steps
        save_strategy = "steps",  # Save model every few steps
        save_steps = 10,  # Save every 10 steps
        save_total_limit = 2,  # Limit the total number of checkpoints
        load_best_model_at_end = True,  # Load best model at the end
    ),
)

## Show Current Memeory Stats
#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
    

trainer_stats = trainer.train()

# Saving the trainer stats
with open("trainer_stats_10K.json", "w") as f:
    json.dump(trainer_stats, f, indent=4)

print("Model is saved.....")
model.save_pretrained("./../llama3-lmparsCit-model/llama-3-8b-Instruct-bnb-4bit-lmparscit-10K") # Local saving
tokenizer.save_pretrained("./../llama3-lmparsCit-model/llama-3-8b-Instruct-bnb-4bit-lmparscit-10K")

# Clear CUDA cache
torch.cuda.empty_cache()
# Manually trigger garbage collection
gc.collect()

