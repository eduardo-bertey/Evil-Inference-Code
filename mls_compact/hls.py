"""SimHash / LSH de 23 bits.

Genera un dato aleatorio de 24 bits y devuelve el par:
  a1 -> hash de 23 bits del dato
  b2 -> el dato de 24 bits (el que se hashea)

El hash es lineal (XOR + rotacion): el modelo puede aprenderlo y generalizar
a datos que nunca vio. Datos parecidos -> hashes parecidos (propiedad LSH).
"""

from __future__ import annotations

import random

BITS_IN = 24
BITS_WIN = 23
MASK_IN = (1 << BITS_IN) - 1
MASK_WIN = (1 << BITS_WIN) - 1


def _mix(v: int) -> int:
    """Mezcla lineal: XOR + rotacion (sin multiplicaciones, aprendible)."""
    v &= MASK_WIN
    v ^= (v << 7) & MASK_WIN
    v ^= (v >> 5)
    v ^= (v << 11) & MASK_WIN
    return v & MASK_WIN


def lsh_hash(rng: random.Random | None = None, salt: int = 0):
    """Genera un dato de 24 bits y devuelve (a1 su hash de 23 bits, b2 el dato)."""
    r = rng if rng is not None else random
    b2 = r.getrandbits(BITS_IN)
    window = b2 & MASK_WIN
    a1 = _mix(window ^ salt)
    return a1, b2


def main():
    r = random.Random(7)
    salt = r.getrandbits(BITS_IN)
    print(f"salt={salt} ({BITS_IN} bits)")
    base = r.getrandbits(BITS_IN)
    print("LSH: datos parecidos -> hashes parecidos")
    for i in range(4):
        b2 = base ^ (1 << i)
        a1, _ = lsh_hash(random.Random(b2), salt)
        print(f"  dato={b2:024b} -> hash={a1:023b}")
    a1, b2 = lsh_hash(r, salt)
    print(f"b2={b2:024b} ({b2}) -> a1={a1:023b} ({a1})")


if __name__ == "__main__":
    main()
