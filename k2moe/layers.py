"""Capas del Transformer y freeze top-down por bloques (de arriba hacia abajo)."""


def get_layers(model):
    for getter in (lambda m: m.model.layers,
                   lambda m: m.language_model.layers,
                   lambda m: m.transformer.h,
                   lambda m: m.encoder.layer,
                   lambda m: m.decoder.layers):
        try:
            return list(getter(model))
        except AttributeError:
            pass
    raise ValueError("No encuentro las capas del Transformer")


def top_down_blocks(n_layers, per_block):
    """Indices de entrenamiento por bloque, desde la ULTIMA capa hacia la primera."""
    blocks, hi = [], n_layers - 1
    while hi >= 0:
        lo = max(0, hi - per_block + 1)
        blocks.append(list(range(lo, hi + 1)))
        hi = lo - 1
    return blocks


def apply_freeze(model, layers, n_active_blocks, blocks):
    """Deja entrenables los primeros n_active_blocks bloques (top-down) +
    todo lo que no sea una capa del transformer (embeddings/norms/lm_head)."""
    trainable = set()
    for i in range(min(n_active_blocks, len(blocks))):
        trainable.update(blocks[i])
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad = (i in trainable)
    return sorted(trainable)
