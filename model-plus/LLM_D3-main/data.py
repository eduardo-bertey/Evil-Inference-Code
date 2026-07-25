# import os
# import numpy as np
# import pyarrow.compute as pc
# from datasets import load_dataset
# from tqdm import tqdm

# # --- Configuration ---
# NUM_PROC = 32
# OUT_PATH = "data/data_19b.bin"
# CACHE_DIR = "../cache_dataset/"
# OWT_BIN_PATH = "train.bin"
# os.makedirs("data", exist_ok=True)

# # allenai/c4

# def main():
#     # 1. Load Pile
#     print("Loading Pile-10b...")
#     pile = load_dataset("NeelNanda/pile-tokenized-10b", split="train", cache_dir=CACHE_DIR)
    
#     # 2. Calculate exact total size
#     owt_token_count = os.path.getsize(OWT_BIN_PATH) // 2
    
#     print("Calculating Pile token count (using PyArrow compute)...")
#     # pc.list_value_length calculates lengths for every row; .sum() gives the total
#     # This is extremely fast and avoids the AttributeError
#     token_lengths = pc.list_value_length(pile.data['tokens'])
#     pile_token_count = pc.sum(token_lengths).as_py()
    
#     total_tokens = owt_token_count + pile_token_count
#     print(f"Total tokens: {total_tokens:,}")

#     # 3. Create the 40GB "Empty" File
#     print(f"Pre-allocating {total_tokens * 2 / 1024**3:.2f} GB...")
#     out_mmap = np.memmap(OUT_PATH, dtype=np.uint16, mode='w+', shape=(total_tokens,))

#     # 4. Copy OWT (Standard binary copy)
#     print("Writing OpenWebText portion...")
#     # Map the source file as read-only
#     owt_raw = np.memmap(OWT_BIN_PATH, dtype=np.uint16, mode='r')
    
#     # We'll copy in 128MB chunks
#     chunk_size = 128 * 1024 * 1024 // 2 # num of uint16 elements in 128MB
    
#     for i in tqdm(range(0, owt_token_count, chunk_size), desc="Copying OWT"):
#         end_i = min(i + chunk_size, owt_token_count)
#         # Write directly to the pre-allocated mmap
#         out_mmap[i:end_i] = owt_raw[i:end_i]
    
#     out_mmap.flush()
#     del owt_raw

#     # 5. Parallel Write the Pile (32 Processes)
#     print("Preparing offsets...")
#     # We calculate cumulative offsets so each row knows its exact slice
#     # np.cumsum converts lengths into absolute positions
#     pile_offsets = np.cumsum(token_lengths.to_numpy(), dtype=np.int64)
#     # Insert a 0 at the beginning for the first row's start position
#     pile_offsets = np.insert(pile_offsets, 0, 0)

#     def parallel_writer(examples, indices):
#         # Re-open mmap in each process
#         m = np.memmap(OUT_PATH, dtype=np.uint16, mode='r+', shape=(total_tokens,))
        
#         for i, row_idx in enumerate(indices):
#             tokens = examples['tokens'][i]
#             # Global Start = OWT end + this row's starting offset
#             start_pos = owt_token_count + pile_offsets[row_idx]
#             end_pos = start_pos + len(tokens)
#             m[start_pos:end_pos] = tokens
#         return {"done": [True] * len(indices)}

#     print(f"Launching {NUM_PROC} processes to write Pile...")
#     pile.map(
#         parallel_writer,
#         with_indices=True,
#         batched=True,
#         batch_size=2000, 
#         num_proc=NUM_PROC,
#         desc="Parallel Writing"
#     )

#     out_mmap.flush()
#     print(f"Success! Final file size: {os.path.getsize(OUT_PATH) / 1024**3:.2f} GB")

# if __name__ == "__main__":
#     main()

# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import multiprocessing as mp
from huggingface_hub import login

login("...") #<- Your Huggingface token

# Configuration
num_proc = 64 # Use all available cores
model_id = "EleutherAI/gpt-neox-20b"
# Use use_fast=True for the Rust-based tokenizer
enc = AutoTokenizer.from_pretrained(model_id, use_fast=True)
eos_id = enc.eos_token_id 

if __name__ == '__main__':
    # 1. Faster Loading: Load specific shards directly
    # 100M docs is roughly 25-30% of C4. We take the first 300 shards.
    base_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
    shard_pattern = [f"{base_url}c4-train.{i:05d}-of-01024.json.gz" for i in range(300)]
    
    print(f"🚀 Loading {len(shard_pattern)} shards from Hugging Face using {num_proc} cores...")

    # 2. Use the "json" loader with web URLs
    # This treats them as raw files and skips the C4 validation checks
    dataset = load_dataset(
        "json", 
        data_files=shard_pattern, 
        num_proc=num_proc,
        split="train"
    )

    # 2. Fast Split
    print("Splitting dataset...")
    split_dataset = dataset.train_test_split(test_size=0.000025, seed=2357, shuffle=True)
    
    # 3. Optimized Map
    # We use 'batched=True' to let the Rust tokenizer process blocks of text at once
    def process_fast(batch):
        # The Fast Tokenizer can process a whole list of strings extremely quickly
        tokenized = enc(batch['text'], add_special_tokens=False)
        out_ids = []
        out_lens = []
        for ids in tokenized['input_ids']:
            ids.append(eos_id)
            out_ids.append(ids)
            out_lens.append(len(ids))
        return {'ids': out_ids, 'len': out_lens}

    tokenized = split_dataset.map(
        process_fast,
        batched=True,
        batch_size=1000,
        remove_columns=['text'],
        desc="Tokenizing with Rust FastTokenizer",
        num_proc=num_proc,
    )

    # 4. Binary Writing (Data Directory)
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    for split, dset in tokenized.items():
        # Using dset.fast_column('len') if available, else sum
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(data_dir, f'{split}.bin')
        
        print(f"\n✍️ Writing {arr_len:,} tokens to {filename}...")
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        
        idx = 0
        # Larger batch size for disk writing to reduce IO overhead
        write_batch_size = 4096 
        for i in tqdm(range(0, len(dset), write_batch_size), desc=f"Flushing {split}"):
            batch = dset[i : i + write_batch_size]
            # Faster concatenation using list comprehension
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        
        arr.flush()
        del arr
        print(f"✅ Finished {split}")