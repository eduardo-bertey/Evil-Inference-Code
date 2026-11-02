import argparse
import os
import re
import sys
import threading

HELP = """Check the readme"""


_ABSL_LOG_START = re.compile(rb"^[EIWF]\d{4} \d\d:\d\d:\d\d")
_NOISY_LOG_HEADER = re.compile(
    rb"\] (?:Fusion: .*gemm_fusion|Computation: .*_computation|Delay kernel timed out)"
)

_log_filter_installed = False


def _install_xla_log_filter():
    """Drop XLA Triton autotuner noise from stderr.

    XLA's Triton GEMM autotuner logs failed candidate fusions via LOG(ERROR)
    in xtile_compiler.cc and cuda_timer.cc. These are unconditional and do
    not respect TF_CPP_MIN_LOG_LEVEL, so we filter them at the file
    descriptor level.

    Strategy:
      - Rebind Python's sys.stderr to a fresh file object over the real
        terminal fd, so tqdm and print() writes go straight to the terminal
        and never enter our pipe. This keeps progress bars (which use \\r
        without trailing \\n) from stalling the filter's line parser.
      - Replace fd 2 with a pipe. Only C-level writes (absl / XLA LOG(...))
        now flow through the pipe, and they are always \\n-terminated and
        well-formed, so a simple line-based filter is reliable.
    """
    global _log_filter_installed
    if _log_filter_installed:
        return
    _log_filter_installed = True

    py_stderr_fd = os.dup(2)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    sys.stderr = os.fdopen(py_stderr_fd, "w", encoding="utf-8",
                           errors="replace", buffering=1)

    out_fd = os.dup(2) 

    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def pump():
        reader = os.fdopen(r_fd, "rb", buffering=0)
        out = os.fdopen(out_fd, "wb", buffering=0)
        buf = b""
        skipping = False
        try:
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    idx = buf.find(b"\n")
                    if idx == -1:
                        break
                    line = bytes(buf[:idx])
                    buf = buf[idx + 1:]
                    is_log_start = bool(_ABSL_LOG_START.match(line))
                    if skipping:
                        if is_log_start:
                            if _NOISY_LOG_HEADER.search(line):
                                continue
                            skipping = False
                            out.write(line + b"\n")
                        # else: continuation body of a skipped log block — drop
                    else:
                        if is_log_start and _NOISY_LOG_HEADER.search(line):
                            skipping = True
                            continue
                        out.write(line + b"\n")
        except Exception:
            pass

    t = threading.Thread(target=pump, daemon=True, name="xla-log-filter")
    t.start()


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
_install_xla_log_filter()



def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="needle", add_help=False)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("run", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--query", type=str, default=None, help="Query text for tool-call generation")
    p.add_argument("--tools", type=str, default=None, help="Tools JSON for tool-call generation")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-constrained", action="store_true",
                   help="Disable grammar-constrained decoding for tool names/arg keys")

    p = sub.add_parser("finetune", add_help=False)
    p.add_argument("jsonl_path", type=str, help="Path to JSONL training data")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Base model checkpoint (auto-downloads from HuggingFace if omitted)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA adapter rank (default: 16)")
    p.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling alpha (default: 32)")
    p.add_argument("--max-len", type=int, default=1024, help="Max training sequence length")
    p.add_argument("--generate", type=int, default=0,
                   help="Generate N extra examples via OpenRouter before training (0 = off)")
    p.add_argument("--model", type=str, default="deepseek/deepseek-v4-flash",
                   help="OpenRouter model for --generate")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent OpenRouter requests when generating (default: 8)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default=None, help="Output adapter path (.pkl)")

    p = sub.add_parser("generate-data", add_help=False)
    p.add_argument("--tools", type=str, default=None, help="Tool schemas JSON to seed generation")
    p.add_argument("--augment", type=str, default=None, help="Existing JSONL to expand")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent OpenRouter requests (default: 16)")
    p.add_argument("--model", type=str, default="deepseek/deepseek-v4-flash")
    p.add_argument("--output", type=str, default=None)

    p = sub.add_parser("build", add_help=False)
    p.add_argument("checkpoint", type=str, help="Base checkpoint (.pkl) to export")
    p.add_argument("--lora", type=str, default=None, help="LoRA adapter to merge before export")
    p.add_argument("--out", type=str, default=None, help="Output .cact path")
    p.add_argument("--upload", action="store_true", help="Push the .cact to $NEEDLE_HF_REPO")
    p.add_argument("--bits", type=str, default=None, choices=["2", "4"])

    p = sub.add_parser("playground", add_help=False)
    p.add_argument("--weights", type=str, default=None,
                   help="Tuned .cact to serve (defaults to the base model from HuggingFace)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")

    args = parser.parse_args()

    if not args.command:
        print(HELP)
        sys.exit(0)

    if args.command == "run":
        from .model.run import main as run_main
        run_main(args)
    elif args.command == "finetune":
        from .model.finetune import finetune_local
        finetune_local(args)
    elif args.command == "generate-data":
        from .model.finetune import generate_main
        generate_main(args)
    elif args.command == "build":
        from .model.finetune import build_main
        build_main(args)
    elif args.command == "playground":
        from .playground.server import main as playground_main
        playground_main(args)
