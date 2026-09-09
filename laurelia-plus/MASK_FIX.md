# Post-mortem: bug de máscara en `folded_bag_ce` (TST loss colapsado)

## Síntoma

- `s10 loss 1.29` con modelo **random** (esperado: `ln(32000) ≈ 10.4`).
- Piso estable `0.146` en fase TST, idéntico entre bloques.
- Fase CE honesta (14.5 → 7.7 en recovery). Generación basura en TST.

## Causa raíz

Commit `a5af82b`: cambió la máscara de

```python
ok = pos < T          # comparación LOCAL: vale para todas las filas
```

a

```python
ok = (row + pos) < T  # compara índices GLOBALES del batch aplanado contra largo de fila
```

`row = arange(B) * T` desplaza cada fila; al comparar contra `T`
(largo de **fila**, no del batch), las filas 1–7 quedan 100%
enmascaradas: aportan `CE = 0` y **gradiente cero**.

El código original solo tenía un crash de reshape
(`ok` era `[1,L,s]` y se pedía `.reshape(B*L, s)`); bastaba un
`.expand()`. El "fix" cambió la semántica y rompió el entrenamiento.

## Por qué el loss caía

Reportado = `tot / (B·L)` con 7/8 de los términos en cero:

- `s10`: `10.4 / 8 ≈ 1.3` ✓ (medido 1.2959).
- Piso: fila 0 memorizada (`≈1.17`) `/ 8 ≈ 0.146` ✓.
- Estable entre bloques: todas las filas 0 de todos los bloques aprenden un poco; el resto, nada.

Medición Y entrenamiento rotos a la vez: 87% del cómputo a la basura,
un solo renglón del batch entrenaba.

## Cómo se probó (DEBUG_TST, una vez por run)

- `valid/pos = 1.37` (sano: ~3.99) → 2/3 de las posiciones anuladas.
- `CE por futuro j ≈ 0.33` en los 4 = `(512 × 10.4 / 4) / 4096` →
  modelo random (CE≈10.4 real) visible solo en 1/8 de posiciones.
- `x[0,:8]` vs `y[0,:8]` confirmó shift +1 correcto (el targeting estaba bien).

## Fix (`a03176e`)

Máscara local pura, sin ningún término de fila:

```python
pos = arange(L)[:, None] * s + (s - 1) + arange(s)[None, :]  # [L,s]
ok1 = pos < T                                                # [L,s], compartida
idx = (row + pos.clamp_max(T - 1)).reshape(B * L, s)         # global solo aquí
```

Resultado: `valid/pos = 3.99`, `s10 = 10.4043 = ln(32000)`, per-j parejos
`≈2.62`. La Eq.3 del paper (media de CEs) quedó intacta; solo se restauró
que todas las filas la computen.

## Lecciones

1. Un crash de shapes (`.reshape`) se arregla con forma (`.expand`), nunca cambiando la semántica de la comparación.
2. `valid/pos` (términos válidos por posición) es el termómetro: debe dar `≈ s` menos cola; cualquier valor bajo = máscara rota.
3. Un loss "demasiado bueno" con modelo random es prueba de bug, no de aprendizaje.
