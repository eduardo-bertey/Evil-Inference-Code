import json
import os
import pickle
import types

import pytest

pytestmark = pytest.mark.slow

TOOLS = [{"name": "send_email", "parameters": {"type": "object", "properties": {
    "to": {"type": "string"}, "subject": {"type": "string"}}, "required": ["to"]}}]


def _write_data(path):
    rows = [
        {"tools": TOOLS, "query": "email a@b.com about lunch",
         "reasoning": "to from query", "answers": [
             {"name": "send_email", "arguments": {"to": "a@b.com", "subject": "lunch"}}]},
        {"tools": TOOLS, "query": "nothing actionable here",
         "reasoning": "off-topic", "answers": []},
    ]
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _finetune_args(data, checkpoint, out, ckpt_dir):
    return types.SimpleNamespace(
        jsonl_path=str(data), checkpoint=checkpoint, epochs=1, batch_size=2,
        lr=1e-3, lora_rank=4, lora_alpha=8.0, max_len=64, generate=0,
        model=None, checkpoint_dir=str(ckpt_dir), out=str(out))


def test_finetune_writes_adapter(tiny_checkpoint, tmp_path):
    from needle.model.finetune import finetune_local

    data = tmp_path / "data.jsonl"
    _write_data(data)
    out = tmp_path / "adapter.pkl"
    progress = []
    finetune_local(_finetune_args(data, tiny_checkpoint, out, tmp_path / "ck"),
                   progress=progress.append)

    assert any("loss" in m for m in progress)
    assert out.exists()
    with open(out, "rb") as handle:
        adapter = pickle.load(handle)
    assert adapter["rank"] == 4
    assert abs(adapter["scale"] - 2.0) < 1e-6
    assert adapter["base"] == tiny_checkpoint
    assert adapter["lora"]
    for value in adapter["lora"].values():
        assert "A" in value and "B" in value


def test_finetune_then_build_merges(tiny_checkpoint, tmp_path):
    from needle.model.finetune import finetune_local, build_main
    from needle.model.export import read_export

    data = tmp_path / "data.jsonl"
    _write_data(data)
    adapter = tmp_path / "adapter.pkl"
    finetune_local(_finetune_args(data, tiny_checkpoint, adapter, tmp_path / "ck"))

    out = str(tmp_path / "merged.cact")
    build_main(types.SimpleNamespace(checkpoint=tiny_checkpoint, lora=str(adapter),
                                     out=out, upload=False, bits="4"))
    assert os.path.exists(out)
    header, _ = read_export(out)
    assert header["num_tensors"] > 0
