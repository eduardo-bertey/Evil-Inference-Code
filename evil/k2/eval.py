"""Benchmarks K2 con harness estándar (lm-evaluation-harness, el del paper TST).
Uso Colab:
  !pip install -q lm-evaluation-harness
  !python eval.py [--model IFM/K2-Horizon-0.9B] [--tasks hellaswag,arc_easy,piqa,winogrande] [--limit 200]
Sin harness instalado, usa el mini multi-choice ES embebido.
Perplejidad siempre sobre bloque laurelia-plus.
"""
import math
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "Evil-Inference-Code", "laurelia-plus"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib
train_data = importlib.import_module("train-data")

BASE = "IFM/K2-Horizon-0.9B"
SEQ = 2048
TASKS = "hellaswag,arc_easy,piqa,winogrande"

MINI = [
    ("La capital de Argentina es", ["Buenos Aires", "Córdoba", "Rosario", "Mendoza"], 0),
    ("2 + 2 × 3 es igual a", ["8", "12", "6", "10"], 0),
    ("El agua hierve a", ["100 °C", "90 °C", "80 °C", "120 °C"], 0),
    ("Don Quijote fue escrito por", ["Cervantes", "Lope de Vega", "Góngora", "Quevedo"], 0),
    ("El planeta más cercano al Sol es", ["Mercurio", "Venus", "Marte", "Júpiter"], 0),
    ("La fotosíntesis ocurre en", ["los cloroplastos", "las mitocondrias", "el núcleo", "los ribosomas"], 0),
    ("El idioma oficial de Brasil es", ["portugués", "español", "inglés", "francés"], 0),
    ("Un siglo tiene", ["100 años", "10 años", "1000 años", "50 años"], 0),
    ("La raíz cuadrada de 144 es", ["12", "14", "16", "10"], 0),
    ("El océano más grande es", ["el Pacífico", "el Atlántico", "el Índico", "el Ártico"], 0),
    ("La moneda de Japón es", ["el yen", "el won", "el yuan", "el dólar"], 0),
    ("H2O es la fórmula de", ["el agua", "el oxígeno", "el hidrógeno", "el dióxido"], 0),
    ("El autor de Cien años de soledad es", ["García Márquez", "Vargas Llosa", "Borges", "Cortázar"], 0),
    ("Un triángulo con tres lados iguales es", ["equilátero", "isósceles", "escaleno", "rectángulo"], 0),
    ("La capital de España es", ["Madrid", "Barcelona", "Sevilla", "Valencia"], 0),
    ("El gas que respiramos mayormente es", ["nitrógeno", "oxígeno", "helio", "carbono"], 0),
    ("La Torre Eiffel está en", ["París", "Londres", "Roma", "Berlín"], 0),
    ("El año tiene", ["12 meses", "10 meses", "13 meses", "11 meses"], 0),
    ("El metal líquido a temperatura ambiente es", ["el mercurio", "el hierro", "el cobre", "el plomo"], 0),
    ("La independencia argentina fue en", ["1816", "1810", "1820", "1806"], 0),
    ("El río más largo del mundo es", ["el Amazonas", "el Nilo", "el Paraná", "el Misisipi"], 0),
    ("La velocidad de la luz es de", ["300.000 km/s", "150.000 km/s", "30.000 km/s", "3.000 km/s"], 0),
    ("El hueso más largo del cuerpo es", ["el fémur", "la tibia", "el húmero", "la columna"], 0),
    ("Buenos Aires es la capital de", ["Argentina", "Uruguay", "Chile", "Paraguay"], 0),
]


def get_arg(name, default):
    for a in sys.argv:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


@torch.no_grad()
def perplexity(model, tok, n_tokens):
    sd = train_data.TrainData(block_idx=0)

    class W:
        def encode(self, t):
            return tok.encode(t, add_special_tokens=False)
    sd.load_tokens(W())
    ids = sd.get_tokens()[:n_tokens]
    loss_fct = torch.nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    for i in range(0, len(ids) - SEQ - 1, SEQ):
        x = torch.tensor([ids[i:i + SEQ]], dtype=torch.long, device=model.device)
        y = torch.tensor([ids[i + 1:i + 1 + SEQ]], dtype=torch.long, device=model.device)
        logits = model(input_ids=x).logits.float()
        tot += loss_fct(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        n += 1
    return math.exp(tot / max(n, 1))


@torch.no_grad()
def mini_bench(model, tok):
    ok = 0
    for q, opts, gold in MINI:
        scores = []
        for o in opts:
            ids = tok(f"{q} {o}", return_tensors="pt").to(model.device)
            qids = tok(q, return_tensors="pt").to(model.device)
            logits = model(input_ids=ids["input_ids"]).logits.float()
            start = qids["input_ids"].shape[1] - 1
            lp = torch.log_softmax(logits[0, start:-1], dim=-1)
            tgt = ids["input_ids"][0, start + 1:]
            scores.append(lp.gather(1, tgt.unsqueeze(1)).squeeze(1).mean().item())
        if int(torch.tensor(scores).argmax()) == gold:
            ok += 1
    return ok, len(MINI)


def harness_bench(model_id, tasks, limit):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model_id, dtype="bfloat16", trust_remote_code=True,
              device="cuda" if torch.cuda.is_available() else "cpu")
    res = evaluator.simple_evaluate(model=lm, tasks=tasks.split(","),
                                    limit=limit, log_samples=False)
    for t, m in res["results"].items():
        acc = m.get("acc,none", m.get("acc", "?"))
        accn = m.get("acc_norm,none", "")
        print(f"  {t}: acc={acc} acc_norm={accn}")
    return res


def main():
    model_id = get_arg("--model", BASE)
    tasks = get_arg("--tasks", TASKS)
    limit = get_arg("--limit", "200")
    limit = None if limit == "all" else int(limit)
    n_tokens = int(get_arg("--ppl-tokens", 200000))

    try:
        import lm_eval  # noqa
        print(f"Harness: {tasks} (limit={limit})")
        harness_bench(model_id, tasks, limit)
    except ImportError:
        print("lm_eval no instalado -> mini multi-choice ES embebido "
              "(!pip install -q lm-evaluation-harness para el estándar)")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        model.eval()
        ok, n = mini_bench(model, tok)
        print(f"Mini-ES: {ok}/{n} = {100 * ok / n:.1f}%")
        del model

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    print(f"Perplejidad ({n_tokens} toks): {perplexity(model, tok, n_tokens):.2f}")


if __name__ == "__main__":
    main()
