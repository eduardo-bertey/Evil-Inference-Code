import torch
import gc
import numpy as np
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments, 
    TrainerCallback, default_data_collator
)
import transformers
from torch.utils.data import DataLoader, IterableDataset
from LLM_2 import GPT, GPTConfig 
from manager import MANAGER
import math

# 1. Dataset: Sequential & Optimized for RAM
class BinDataset(IterableDataset):
    def __init__(self, data_path, block_size):
        self.data_path = data_path
        self.block_size = block_size

    def __iter__(self):
        data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        total_tokens = len(data)

        for i in range(0, total_tokens - self.block_size, self.block_size):
            chunk = data[i : i + self.block_size].astype(np.int64)
            t = torch.from_numpy(chunk)
            yield {"input_ids": t, "labels": t}

# 2. Cleanup and Optimizations
gc.collect()
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda"

# 3. Model & Config Setup
config = GPTConfig() 
model = GPT(config)

# HF compatibility patch
model.can_generate = lambda: True 
model.config.model_type = "gpt_moe"
model = torch.compile(model)

def to_json_string(self):
    import json
    return json.dumps(self.__dict__, indent=2)

GPTConfig.to_json_string = to_json_string

# 4. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# 5. Calculate Exact Steps for 2 Epochs
BIN_PATH = "data/train.bin"
raw_data = np.memmap(BIN_PATH, dtype=np.uint16, mode='r')

total_sequences = len(raw_data) // config.block_size 

eff_batch_size = config.batch_size * config.grad_acc

steps_per_epoch = total_sequences // eff_batch_size
total_max_steps = steps_per_epoch * config.num_train_epochs

print(f"Total Sequences: {total_sequences:,}")
print(f"Steps per Epoch: {steps_per_epoch:,}")
print(f"Total Steps: {total_max_steps:,}")

class MoETrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator, 
            num_workers=8, 
            pin_memory=True
        )
        
print("Batch-size: ", config.batch_size)

training_args = TrainingArguments(
    output_dir="./model_save",
    per_device_train_batch_size=config.batch_size, 
    gradient_accumulation_steps=config.grad_acc,

    max_steps=total_max_steps, 
    num_train_epochs=config.num_train_epochs,
    warmup_steps=config.warm_up,
    max_grad_norm=1.0, 
    
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine", 
    bf16=torch.cuda.is_bf16_supported(), 
    logging_steps=5,
    save_steps=1_000,
    eval_steps=1_000,
    eval_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    save_total_limit=5,
    weight_decay=config.weight_decay,
    remove_unused_columns=False,
    save_safetensors=False,
    gradient_checkpointing=False,
    prediction_loss_only=True,
    label_names=["labels"],

    # load_best_model_at_end=True, #<- avoided at pre training
    metric_for_best_model="loss",
    greater_is_better=False,
) 

def get_wsd_schedule(optimizer_or_list, num_warmup, num_stable, num_decay, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        if step < num_warmup + num_stable:
            return 1.0

        progress = float(step - num_warmup - num_stable) / float(max(1, num_decay))
        progress = min(1.0, progress)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    if isinstance(optimizer_or_list, list):
        return [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda) for opt in optimizer_or_list]
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer_or_list, lr_lambda)

num_warmup = config.warm_up
num_stable = int(total_max_steps * 0.2)
num_decay = total_max_steps - num_warmup - num_stable

optimizer=model.configure_optimizers(config.weight_decay, config.learning_rate, config.betas, "cuda")
scheduler = get_wsd_schedule(optimizer, num_warmup, num_stable, num_decay)
optimizers = optimizer, scheduler

# 8. Initialize and Run
trainer = MoETrainer(
    model=model,
    args=training_args,
    train_dataset=BinDataset(BIN_PATH, config.block_size),
    eval_dataset=BinDataset("data/test.bin", config.block_size),
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    optimizers=optimizers,
)

# trainer.train()
trainer.train(resume_from_checkpoint="model_save/checkpoint-22000/")