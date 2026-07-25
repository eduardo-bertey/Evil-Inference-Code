import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer
from LLM_2 import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# dataset
# -------------------------
ds = load_dataset("yahma/alpaca-cleaned", split="train")

def format_sample(x):
    if x["input"]:
        prompt = f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{x['instruction']}\n\n### Response:\n"

    full_text = prompt + x["output"] + tokenizer.eos_token
    tok = tokenizer(full_text)

    input_ids = tok["input_ids"]

    labels = list(input_ids)

    prompt_len = len(tokenizer(prompt)["input_ids"])
    for i in range(prompt_len - 1):
        labels[i] = -100

    return {"input_ids": input_ids, "labels": labels}


dataset = ds.map(format_sample, remove_columns=ds.column_names)

# -------------------------
# filter too-short answers
# -------------------------
MAX_LENGTH = 2048

def keep_valid(example):
    total_len = len(example["input_ids"])
    answer_len = len(example["labels"]) - example["labels"].count(-100)
    return total_len <= MAX_LENGTH and answer_len > 10

dataset = dataset.filter(keep_valid)

# -------------------------
# train / eval split
# -------------------------
eval_dataset = dataset.select(range(2000))
train_dataset = dataset.select(range(2000, len(dataset)))

# -------------------------
# collator: pad to batch max length
# -------------------------
def causal_lm_collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    labels    = [torch.tensor(f["labels"],    dtype=torch.long) for f in features]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels    = pad_sequence(labels,    batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# -------------------------
# model
# -------------------------
config = GPTConfig()
model = GPT(config).to(device)

# load pretrained checkpoint
state = torch.load("model_save/checkpoint-48833/pytorch_model.bin", map_location=device)
clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
model.load_state_dict(clean, strict=True)


def to_json_string(self):
    import json
    return json.dumps(self.__dict__, indent=2)

GPTConfig.to_json_string = to_json_string

# -------------------------
# training args (NOTE: name = training_args)
# -------------------------
training_args = TrainingArguments(
    output_dir="./ft_out",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    num_train_epochs=1,
    warmup_ratio=0.10,
    lr_scheduler_type="cosine", 
    logging_steps=5,
    eval_steps=50,
    save_steps=50,
    eval_strategy="steps",
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(),
    weight_decay=0.05,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    save_safetensors=False,
    max_grad_norm=1.0
)

import math

steps_per_epoch = math.ceil(
    len(train_dataset) /
    (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
)

total_steps = int(steps_per_epoch * training_args.num_train_epochs)

warmup_steps = int(0.2 * total_steps)
stable_steps = int(0.4 * total_steps)
decay_steps  = total_steps - warmup_steps - stable_steps

from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    weight_decay=0.0
)

from torch.optim.lr_scheduler import LambdaLR

def wsd_lambda(step):
    if step < warmup_steps:
        # warmup: 0 → 1
        return step / max(1, warmup_steps)

    elif step < warmup_steps + stable_steps:
        # stable: flat
        return 1.0

    else:
        # decay: 1 → 0
        progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
        return max(0.0, 1.0 - progress)

scheduler = LambdaLR(optimizer, wsd_lambda)

# -------------------------
# trainer
# -------------------------
class MyTrainer(Trainer):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator
        )

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=causal_lm_collator,
    optimizers=(optimizer, scheduler),
)

# -------------------------
# train
# -------------------------
trainer.train()