import os
import sentencepiece as spm

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR = os.path.join(_PROJECT_ROOT, "tokenizer")
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle")

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3
TOOL_CALL_ID = 4
TOOLS_ID = 5

DEFAULT_MAX_ENC_LEN = 1024
DEFAULT_MAX_DEC_LEN = 512
DEFAULT_MAX_GEN_LEN = 512

_HF_MODEL_REPO = "Cactus-Compute/needle"


class NeedleTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    @property
    def pad_token_id(self): return PAD_ID
    @property
    def eos_token_id(self): return EOS_ID
    @property
    def bos_token_id(self): return BOS_ID
    @property
    def tool_call_token_id(self): return TOOL_CALL_ID
    @property
    def tools_token_id(self): return TOOLS_ID
    @property
    def vocab_size(self): return self.sp.GetPieceSize()

    def encode(self, text):
        return self.sp.Encode(text, out_type=int)

    def decode(self, ids):
        if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple)):
            return [self.sp.Decode(seq) for seq in ids]
        return self.sp.Decode(list(ids))


def _download_tokenizer_from_hf():
    from huggingface_hub import hf_hub_download
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    for fname in ["needle.model", "needle.vocab"]:
        hf_hub_download(
            repo_id=_HF_MODEL_REPO,
            filename=f"tokenizer/{fname}",
            repo_type="model",
            local_dir=TOKENIZER_DIR,
            force_download=True,
        )
        nested = os.path.join(TOKENIZER_DIR, "tokenizer", fname)
        dst = os.path.join(TOKENIZER_DIR, fname)
        if os.path.exists(nested) and not os.path.exists(dst):
            os.rename(nested, dst)


def get_tokenizer(max_samples=None):
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        print("Downloading pretrained tokenizer from HuggingFace...")
        _download_tokenizer_from_hf()
    return NeedleTokenizer(model_path)


def train_tokenizer(vocab_size=8192, corpus_path=None, force=False):
    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path) and not force:
        print(f"Tokenizer already exists at {model_path}")
        return model_path

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    if corpus_path is None:
        raise ValueError("corpus_path required to train tokenizer")

    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=PAD_ID, eos_id=EOS_ID, bos_id=BOS_ID, unk_id=UNK_ID,
        user_defined_symbols=["<tool_call>", "<tools>"],
        byte_fallback=True,
        normalization_rule_name="identity",
        num_threads=min(8, max(1, (os.cpu_count() or 1) // 2)),
        train_extremely_large_corpus=False,
        minloglevel=2,
    )
    print(f"Tokenizer saved to {model_path}")
    return model_path
