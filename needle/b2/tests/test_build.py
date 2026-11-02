import os
import types

import pytest

pytestmark = pytest.mark.slow


def _build_args(checkpoint, out, bits="4", lora=None):
    return types.SimpleNamespace(checkpoint=checkpoint, lora=lora, out=out,
                                 upload=False, bits=bits)


def test_build_exports_loadable_cact(tiny_checkpoint, tmp_path):
    from needle.model.finetune import build_main
    from needle.model.export import read_export

    out = str(tmp_path / "tiny.cact")
    build_main(_build_args(tiny_checkpoint, out, bits="4"))

    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    header, tensors = read_export(out)
    assert header["num_tensors"] > 0
    assert len(tensors) == header["num_tensors"]
    assert any(isinstance(t, (bytes, bytearray)) for t in tensors)


def test_build_at_two_bits(tiny_checkpoint, tmp_path):
    from needle.model.finetune import build_main
    from needle.model.export import read_export

    out = str(tmp_path / "tiny_w2.cact")
    build_main(_build_args(tiny_checkpoint, out, bits="2"))
    header, _ = read_export(out)
    assert header["num_tensors"] > 0


def test_export_round_trips_a_projection(tiny_checkpoint, tmp_path):
    import pickle
    import numpy as np
    from needle.model.export import write_export, read_export
    from needle.model.architecture import TransformerConfig, effective_kv_window
    from needle.model.tokenizer import get_tokenizer

    with open(tiny_checkpoint, "rb") as handle:
        ckpt = pickle.load(handle)
    params, config = ckpt["params"], TransformerConfig(**ckpt["config"])

    out = str(tmp_path / "rt.cact")
    write_export(params, config, out, bits=4,
                 tokenizer=get_tokenizer(config.vocab_size),
                 kv_window=effective_kv_window(config))
    header, tensors = read_export(out)

    original = np.asarray(params["stack"]["layers"]["block"]["self_attn"]["q_proj"]["kernel"][0]).T
    dequant = tensors[2]
    assert dequant.shape == original.shape
    assert np.corrcoef(dequant.ravel(), original.ravel())[0, 1] > 0.9
