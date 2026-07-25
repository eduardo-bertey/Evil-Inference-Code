import torch
import torch.nn.functional as F
import sys
from transformers import AutoTokenizer
from model import LLM, Config

# ─────────────────────────────────────────────────────────────────────────────
# Settings & Setup
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "ft_out/checkpoint-8356/pytorch_model.bin" # Update to your best checkpoint
TOKENIZER_PATH = "data_mixed/custom_multilingual_tokenizer"

SYS_PROMPT = "You are Dummy-5, a highly capable AI assistant. You communicate fluently in English, Russian, and Uzbek, always replying in the language the user asked a question. Provide clear, accurate, and concise answers."

# ─────────────────────────────────────────────────────────────────────────────
# Custom Autoregressive Streaming Generator
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def stream_generate(model, input_ids, max_new_tokens, temperature=0.7, top_k=50, repetition_penalty=1.15, eos_token_id=None):
    """
    Custom generator for the bare-metal PyTorch LLM class.
    Yields tokens one by one for a typewriter effect in the terminal.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Crop to max context window to prevent index out of bounds in RoPE
        idx_cond = input_ids if input_ids.size(1) <= model.config.block_size else input_ids[:, -model.config.block_size:]
        
        # Forward pass
        outputs = model(input_ids=idx_cond)
        logits = outputs.logits[:, -1, :].clone() # Grab logits for the very last token
        
        # ─────────────────────────────────────────────────────────────────────
        # Apply Repetition Penalty
        # ─────────────────────────────────────────────────────────────────────
        if repetition_penalty != 1.0:
            # Extract unique tokens seen in the current context window
            seen_tokens = torch.unique(idx_cond[0])
            
            # Penalize the logits of seen tokens
            for token_id in seen_tokens:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        # Apply Temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        yield next_token.item()
        
        if next_token.item() == eos_token_id:
            break
            
        input_ids = torch.cat((input_ids, next_token), dim=1)

# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model architecture and weights...")
    config = Config()
    model = LLM(config).to(DEVICE)
    
    try:
        state_dict = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        # Strip DDP/compile prefixes if they exist
        clean_state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=True)
        model.to(torch.bfloat16) # Run in bf16 to match your training and H200 capabilities
        print("✅ Model loaded successfully.\n")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    print("="*60)
    print(" Multilingual AI Assistant (English / Russian / Uzbek) ")
    print(" Type 'quit', 'exit', or 'stop' to terminate.")
    print(" Type '/history off' or '/history on' to toggle context.")
    print("="*60)

    # 1. Base Setup: Keep the system prompt separate from the rolling history
    system_text = f"System:\n{SYS_PROMPT}\n"
    system_ids = [tokenizer.bos_token_id] + tokenizer(system_text, add_special_tokens=False)["input_ids"]
    
    history_ids = system_ids.copy()
    use_history = False

    while True:
        try:
            # Native Python input() handles UTF-8 for Cyrillic/Uzbek characters automatically
            user_input = input("\nUser: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        cleaned_input = user_input.strip().lower()

        if cleaned_input in ["quit", "exit", "stop"]:
            print("Terminating chat.")
            break
            
        # 2. Add dynamic toggles so you don't have to restart the script
        if cleaned_input == "/history off":
            use_history = False
            print("[System: Conversation history is now OFF. Assistant will only see the current turn.]")
            continue
        elif cleaned_input == "/history on":
            use_history = True
            history_ids = system_ids.copy() # Reset history when turning back on to avoid gaps
            print("[System: Conversation history is now ON. Context has been reset.]")
            continue
            
        if not user_input.strip():
            continue

        # Format the current user turn
        turn_text = f"User:\n{user_input}\nAssistant:\n"
        turn_ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"]
        
        # 3. Route the input based on the history flag
        if use_history:
            history_ids.extend(turn_ids)
            input_ids_for_model = history_ids
        else:
            # Stateless mode: Only System Prompt + Current Turn
            input_ids_for_model = system_ids + turn_ids

        input_tensor = torch.tensor([input_ids_for_model], dtype=torch.long).to(DEVICE)
        
        print("Assistant: ", end="", flush=True)
        
        # Stream the response
        generated_tokens = []

        for token_id in stream_generate(
            model=model, 
            input_ids=input_tensor, 
            max_new_tokens=256, 
            temperature=0.8,   
            top_k=40,
            repetition_penalty=1.15, 
            eos_token_id=tokenizer.eos_token_id
        ):
            if token_id == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(token_id)
            
            # Decode and print dynamically
            chunk = tokenizer.decode([token_id], skip_special_tokens=True)
            print(chunk, end="", flush=True)
            
        print() # Newline after response

        # 4. Only append the model's response to the context window if history is ON
        if use_history:
            history_ids.extend(generated_tokens)
            history_ids.append(tokenizer.eos_token_id)
            
            # Context window management
            if len(history_ids) > (config.block_size - 600):
                print("\n[Warning: Reaching maximum context window. Truncating oldest messages...]")
                sys_len = len(system_ids)
                history_ids = history_ids[:sys_len] + history_ids[-(config.block_size - 1000):]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()