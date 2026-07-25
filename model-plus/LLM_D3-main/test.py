import torch
from transformers import AutoTokenizer
from LLM_2 import GPT, GPTConfig
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# model
config = GPTConfig() 
model = GPT(config).to(device)

# load checkpoint
state_dict = torch.load(
    # "model_save/checkpoint-48833/pytorch_model.bin",
    "ft_out/checkpoint-361/pytorch_model.bin",
    map_location=device
)

# remove torch.compile wrapper
clean_state_dict = {
    k.replace("_orig_mod.", ""): v
    for k, v in state_dict.items()
}

# load correctly
model.load_state_dict(clean_state_dict, strict=True)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

print("✅ Model loaded successfully")



@torch.no_grad()
def sample_generate(
    model,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=None,         
    eos_token_id=None,
    repetition_penalty=1.15  # Recommended: 1.1 to 1.2 for 350M models
):
    """
    input_ids: (B, T)
    returns:   (B, T + max_new_tokens)
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Forward pass (Full re-computation as requested)
        out = model(input_ids)
        logits = out.logits[:, -1, :]  # (B, vocab)

        # --- Repetition Penalty Logic ---
        if repetition_penalty != 1.0:
            for b in range(input_ids.shape[0]):
                for token_id in set(input_ids[b].tolist()):
                    # If logit is positive, reduce it; if negative, make it more negative
                    if logits[b, token_id] > 0:
                        logits[b, token_id] /= repetition_penalty
                    else:
                        logits[b, token_id] *= repetition_penalty

        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K
        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, indices, values)

        # Top-P (nucleus)
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)

            cutoff = cumulative_probs > top_p
            cutoff[:, 1:] = cutoff[:, :-1].clone()
            cutoff[:, 0] = False

            sorted_logits[cutoff] = float('-inf')
            logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Optional EOS stop
        if eos_token_id is not None:
            if (next_token == eos_token_id).all():
                break

    return input_ids

while True:
    prompt = input(">>> ")
    if prompt == "stop":
        break

    prompt = (

           # f"### Instruction:\nYou are Dummy-3.0, helpfull and knowlagable AI assistant. Answer clearly and briefly.\n\n"

            f"### Input:\n{prompt}\n\n"

            f"### Response:\n"

        )

    # prompt = prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    out = sample_generate(
        model,
        inputs["input_ids"],
        max_new_tokens=128,          # Slightly more room for the MoE to conclude
        temperature=0.7,             # Increased from 0.5 to give experts "room to breathe"
        top_k=40,                    # Adding top_k helps filter out noise before top_p kicks in
        top_p=0.85,                  # Increased from 0.50 to allow for more natural linguistic flow
        repetition_penalty=1.15,     # CRITICAL: Prevents MoE "ping-ponging" between two experts
        eos_token_id=tokenizer.eos_token_id
    )
    out = tokenizer.decode(out[0], skip_special_tokens=True)
    out = out[len(prompt):]
    
    print(out)
print("ez!")