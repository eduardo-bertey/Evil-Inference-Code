"""SimHash / LSH de 23 bits.

Genera un dato aleatorio de 24 bits, desliza una ventana de 23 bits
(offset aleatorio) y devuelve el par:
  a1 -> hash de 23 bits del dato
  b2 -> el dato de 24 bits (el que se hashea)

LSH: dos datos que se parecen tienen mas chances de dar el mismo a1.
"""

from __future__ import annotations

import random

BITS_IN = 24
BITS_WIN = 23
MASK_IN = (1 << BITS_IN) - 1
MASK_WIN = (1 << BITS_WIN) - 1
N_SLIDES = BITS_IN - BITS_WIN + 1


def _mix(v: int) -> int:
    v = (v ^ (v >> 33)) * 0xFF51AFD7ED558CCD
    v = (v ^ (v >> 33)) * 0xC4CEB9FE1A85EC53
    return v ^ (v >> 33)


def lsh_hash(rng: random.Random | None = None, salt: int = 0):
    """Genera un dato de 24 bits y devuelve (a1 su hash de 23 bits, b2 el dato)."""
    r = rng if rng is not None else random
    b2 = r.getrandbits(BITS_IN)
    shift = r.randrange(N_SLIDES)
    window = (b2 >> shift) & MASK_WIN
    a1 = _mix(window ^ salt) & MASK_WIN
    return a1, b2


def main():
    r = random.Random(7)
    salt = r.getrandbits(BITS_IN)
    print(f"salt={salt} ({BITS_IN} bits)")
    for _ in range(3):
        a1, b2 = lsh_hash(r, salt)
        print(f"b2={b2:024b} ({b2}) -> a1={a1:023b} ({a1})")


if __name__ == "__main__":
    main()
