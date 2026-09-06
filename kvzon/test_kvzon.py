"""Tests kvzon: determinismo, caches, vectores Q-first.

Requiere kvzon/tokenizer.json + checkpoint.pt (ver download.py).
Corre en CPU con prompt corto. Skips si faltan pesos.
"""

import os
import sys
import unittest

_KVZON = os.path.dirname(os.path.abspath(__file__))
_LLMDIR = os.path.join(os.path.dirname(_KVZON), "laurelia-llm")
sys.path.insert(0, _KVZON)
sys.path.insert(0, _LLMDIR)

import torch

WEIGHTS = os.path.join(_KVZON, "checkpoint.pt")
TOK = os.path.join(_KVZON, "tokenizer.json")
HAS_WEIGHTS = os.path.exists(WEIGHTS) and os.path.exists(TOK)

from model import LLM, Config
from tokenizers import Tokenizer
from chat_compare import QFirst, generate_vanilla, generate_qfirst


def load():
    tok = Tokenizer.from_file(TOK)
    config = Config()
    config.emb_num = tok.get_vocab_size()
    model = LLM(config)
    ckpt = torch.load(ckpt_path := WEIGHTS, map_location="cpu")
    ckpt["model"].pop("head.emb_weight", None)
    model.load_state_dict(ckpt["model"], strict=False)
    del ckpt
    model.eval()
    return model, tok


@unittest.skipUnless(HAS_WEIGHTS, "faltan pesos (corre download.py)")
class TestKvzon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok = load()
        cls.eos = cls.tok.token_to_id("eos_token")
        ids = cls.tok.encode("hola mundo").ids
        cls.x = torch.tensor([ids], dtype=torch.long)

    def test_vanilla_determinista(self):
        torch.manual_seed(7)
        o1, _, s1, _, _ = generate_vanilla(self.model, self.x.clone(), 5, 0.0, 0, 1.0, 1.0, self.eos)
        torch.manual_seed(7)
        o2, _, s2, _, _ = generate_vanilla(self.model, self.x.clone(), 5, 0.0, 0, 1.0, 1.0, self.eos)
        self.assertEqual(o1[0].tolist(), o2[0].tolist())

    def test_qfirst_determinista(self):
        qf = QFirst(self.model)
        torch.manual_seed(7)
        o1, _, _, _, _ = generate_qfirst(qf, self.x.clone(), 5, 0.0, 0, 1.0, 1.0, self.eos)
        torch.manual_seed(7)
        o2, _, _, _, _ = generate_qfirst(qf, self.x.clone(), 5, 0.0, 0, 1.0, 1.0, self.eos)
        self.assertEqual(o1[0].tolist(), o2[0].tolist())

    def test_prefill_difiere_por_diseno(self):
        # Prefill Q-first (capa 1 sin residuo) DIVERGE del vanilla forward.
        with torch.no_grad():
            ref, _ = self.model.forward(self.x)
        qf = QFirst(self.model)
        with torch.no_grad():
            got, _ = qf.prefill(self.x)
        d = float((ref.float() - got.float()).abs().max())
        print(f"\n  prefill Q-first vs vanilla max|dlogits|={d:.3g} (difiere por diseno)")
        self.assertTrue(d < 1e6)  # solo anti-NaN/explosion

    def test_cache_shapes(self):
        qf = QFirst(self.model)
        with torch.no_grad():
            _, caches = qf.prefill(self.x)
        P = self.x.shape[1]
        self.assertEqual(len(caches), 16)
        for c in caches:
            # cache cruda: vectores (att+res) dim D, un token por posicion
            self.assertEqual(tuple(c.shape), (1, P, 768))

    def test_sin_vectores_colgados(self):
        qf = QFirst(self.model)
        with torch.no_grad():
            qf.prefill(self.x)
        self.assertFalse(hasattr(qf, "a_q"))
        self.assertFalse(hasattr(qf, "q_saved"))
        self.assertEqual(len(qf.diag["norm_a"]), 16)

    def test_decode_atencion_unica(self):
        # Decode Q-first: atencion congelada al entry stream -> DIVERGE del
        # vanilla por diseno. Solo verifica que corre, formas y caches.
        torch.manual_seed(7)
        qf = QFirst(self.model)
        o2, _, s2, _, _ = generate_qfirst(qf, self.x.clone(), 2, 0.0, 0, 1.0, 1.0, self.eos)
        self.assertEqual(len(s2), 2)
        self.assertEqual(o2.shape[1], self.x.shape[1] + 2)
        torch.manual_seed(7)
        _, _, s1, _, _ = generate_vanilla(self.model, self.x.clone(), 2, 0.0, 0, 1.0, 1.0, self.eos)
        d = float((s1[0] - s2[0]).abs().max())
        print(f"\n  decode atencion-unica vs vanilla max|dlogits|={d:.3g} (diverge por diseno)")
        self.assertTrue(d < 1e6)  # solo anti-NaN/explosion


if __name__ == "__main__":
    unittest.main(verbosity=2)
