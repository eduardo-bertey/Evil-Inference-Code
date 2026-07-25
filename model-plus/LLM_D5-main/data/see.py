import os
import random
import numpy as np
from transformers import AutoTokenizer

# ── CONFIG ────────────────────────────────────────────────────────────────────
BIN_PATH = "data/train.bin"
TOKENIZER_PATH = "./custom_multilingual_tokenizer"

NUM_SAMPLES = 10       # How many different places to sample from
TOKENS_PER_SAMPLE = 200
BYTES_PER_TOKEN = 2    # np.uint16 uses 2 bytes per element
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(BIN_PATH):
        print(f"❌ Error: Could not find binary file at '{BIN_PATH}'")
        return

    # 1. Load your custom tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

    # 2. Find total tokens in the file without loading it all into memory
    file_size_bytes = os.path.getsize(BIN_PATH)
    total_tokens = file_size_bytes // BYTES_PER_TOKEN
    
    print(f"📊 Binary File Info:")
    print(f"   File size   : {file_size_bytes:,} bytes")
    print(f"   Total tokens: {total_tokens:,} tokens\n")

    if total_tokens <= TOKENS_PER_SAMPLE:
        print("⚠️ File is too small to sample from multiple distinct places.")
        return

    # 3. Read from random offsets
    with open(BIN_PATH, "rb") as f:
        for i in range(1, NUM_SAMPLES + 1):
            # Pick a random starting token index (leave room for TOKENS_PER_SAMPLE at the end)
            start_token_idx = random.randint(0, total_tokens - TOKENS_PER_SAMPLE)
            
            # Calculate the exact byte offset in the file
            byte_offset = start_token_idx * BYTES_PER_TOKEN
            f.seek(byte_offset)
            
            # Read the bytes and reconstruct the uint16 array
            raw_bytes = f.read(TOKENS_PER_SAMPLE * BYTES_PER_TOKEN)
            token_ids = np.frombuffer(raw_bytes, dtype=np.uint16).tolist()
            
            # Decode back to text
            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            
            # Print cleanly formatted block
            print(f"━" * 80)
            print(f"📍 SAMPLE #{i} | Starting at Token Index: {start_token_idx:,} (Byte: {byte_offset:,})")
            print(f"━" * 80)
            print(decoded_text)
            print("\n")

if __name__ == "__main__":
    main()