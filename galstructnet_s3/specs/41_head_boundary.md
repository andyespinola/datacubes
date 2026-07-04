# 41 — Cabeza de fronteras

> Módulo: `models/heads/boundary.py` · Hito: 4 · Depende de: cabezas de
> segmentación (40 v3).
>
> **Reframing v3 (revisión §2.5):** sin parámetros entrenables y derivada
> determinísticamente de `prob`, esto NO es una "cabeza" sino un
> **regularizador de gradiente espacial** (`L_boundary`, útil; conservado).
> Presentarlo así en la tesis evita la pregunta del comité "¿qué aprende esta
> cabeza?" (nada). El mapa `1−B` sigue siendo producto científico de
> inferencia. Opera sobre `prob_lum` (las fronteras científicas se definen en
> luz). El resto del spec v2 (definición B=exp(−‖∇p‖/τ), τ=0.1, MSE contra
> B(Y_lum_raw), tests, gradient-flow a prob) permanece vigente sin cambios.

## Responsabilidad

Produce un mapa continuo en `[0, 1]` que cuantifica **cuán ambigua es la
transición entre componentes** en cada spaxel. Valores altos indican
fronteras (transiciones físicamente reales entre estructuras). Valores bajos
indican interior estable de una componente.

A diferencia de muchos modelos de segmentación que tratan las fronteras como
artefactos a suavizar, aquí las fronteras son **objetos científicos por
derecho propio**. Las fronteras del bulbo, el final de la barra, los bordes
de los brazos espirales son cantidades de interés astronómico.

## Diseño general

La cabeza no es un "predictor independiente" — se deriva directamente del
output de la cabeza Dirichlet. Esto asegura coherencia: lo que decimos sobre
fronteras es consistente con lo que decimos sobre clases.

Definición operacional:

```
B(i,j) = exp(-||∇_xy prob(i,j)|| / τ)
```

Donde `||∇_xy prob||` es la magnitud del gradiente espacial multiclase, y
`τ` es un hiperparámetro de "anchura" de las fronteras.

Cuando dos clases vecinas son muy distintas (gran cambio en `prob`), el
gradiente es grande, B → 0. Cuando son iguales (interior), gradiente cero,
B → 1.

**Inversión semántica**: nota que B está en `[0, 1]` pero **alto = interior,
bajo = frontera**. Esto es contraintuitivo. Para visualizarlo como "mapa
de fronteras" típico, se muestra `1 - B`.

Decidimos llamarlo así porque la pérdida y el cálculo son más naturales en
esta forma. Documentamos claramente.

## Contrato de entrada

```python
prob: torch.Tensor  # (B, K, 69, 69) — probabilidades de la cabeza Dirichlet
```

## Contrato de salida

```python
B: torch.Tensor  # (B, 1, 69, 69) — mapa continuo en (0, 1]
```

## Algoritmo

```python
class BoundaryHead(nn.Module):
    """
    Mapa de fronteras derivado del gradiente espacial de las probabilidades.

    No tiene parámetros entrenables propios — es una transformación
    determinística sobre la salida de la cabeza Dirichlet.
    """

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, prob: Tensor) -> Tensor:
        # prob: (B, K, H, W)

        # Gradiente espacial por diferencias finitas
        # dy: diff a lo largo de eje H. Para spaxel (i,j), compara con (i-1,j).
        # En i=0 usamos copia (gradiente cero en bordes del cubo).
        dy = torch.diff(prob, dim=2, prepend=prob[:, :, :1, :])
        dx = torch.diff(prob, dim=3, prepend=prob[:, :, :, :1])

        # Magnitud del gradiente sobre todas las clases:
        # ||∇p||² = sum_c (dp_c/dx)² + (dp_c/dy)²
        grad_sq = (dy**2 + dx**2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        grad_mag = torch.sqrt(grad_sq + 1e-10)              # estabilidad

        # B = exp(-||∇|| / τ)
        boundary = torch.exp(-grad_mag / self.tau)
        return boundary  # (B, 1, H, W) ∈ (0, 1]
```

### Detalles importantes

**Diferencias finitas con `prepend`**: en los bordes del cubo (i=0 o j=0),
el `diff` clásico de PyTorch reduciría la dimensión. Usar `prepend` con la
primera columna copiada produce gradiente cero en el borde — equivalente
a asumir condiciones de frontera "no-flux".

**Gradiente sumado sobre clases**: una transición de "100% disco" a "100%
brazo" tiene gradiente alto en ambas clases (-1.0 en disco, +1.0 en brazo).
Sumar magnitudes al cuadrado captura esto correctamente.

**`τ = 0.1` como default**: significa que un cambio de magnitud 0.1 en el
gradiente reduce B a `e^(-1) ≈ 0.37`. Esto es un cambio "grande pero no
total". Ajustable según calidad observada.

## Pérdida asociada (importante)

La cabeza por sí sola es solo una transformación. La señal de aprendizaje
viene de `L_boundary` en `losses/boundary.py`:

```
L_boundary = MSE(B_predicho, B_target)
```

Donde `B_target` se calcula de la misma forma sobre las **etiquetas
suaves**:

```python
B_target = exp(-||∇ Y_int_light|| / τ)
```

El modelo aprende que sus probabilidades deben tener gradientes espaciales
consistentes con las etiquetas. Esto regulariza la cabeza Dirichlet
indirectamente — si las etiquetas tienen una transición suave entre disco y
brazo, las predicciones también deben tenerla.

## Validación

### Tests unitarios (`tests/unit/test_boundary_head.py`)

1. **Shape de salida**: input `(2, 5, 69, 69)` → output `(2, 1, 69, 69)`.
2. **Rango de valores**: `B ∈ (0, 1]`. Test sobre input aleatorio.
3. **Caso constante**: si `prob` es constante en el espacio (todos los
   spaxels con misma distribución), B debe ser ≈ 1 en todas partes (gradiente
   cero).
4. **Caso de transición clara**: construir un `prob` con una mitad disco
   (1.0) y otra mitad bulbo (1.0), B debe tener un valle alineado con la
   frontera.
5. **Determinismo**: forward 1 = forward 2 con mismo input.
6. **Sin parámetros entrenables**: `len(list(head.parameters()))` == 0.

### Test de gradiente

Aunque la cabeza no tiene parámetros, el gradiente debe fluir a través de
ella hacia `prob` (de la cabeza Dirichlet anterior).

```python
prob = torch.rand(2, 5, 69, 69, requires_grad=True)
B = boundary_head(prob)
B.sum().backward()
assert prob.grad is not None and prob.grad.abs().sum() > 0
```

## Criterios de aceptación

- [ ] Tests unitarios pasan.
- [ ] Output `(B, 1, 69, 69)` en `(0, 1]`.
- [ ] Tiempo de forward despreciable (es solo aritmética).
- [ ] Gradient flow hacia `prob` confirmado.

## Notas de implementación

- **Sin parámetros entrenables**: este es deliberado. La cabeza es una
  transformación determinística. Si quisiéramos parámetros (e.g., `τ`
  aprendible), podríamos hacerlo, pero `τ` global probablemente no es lo
  óptimo — distintas regiones de la galaxia pueden tener escalas de
  transición distintas.

- **Versión más sofisticada (futuro)**: predecir `B` directamente con una
  Conv 1×1 desde features del decoder, en paralelo a la cabeza Dirichlet.
  Esto permitiría que el modelo aprenda B más libremente, pero introduce el
  riesgo de inconsistencia con `prob`. Por ahora, derivado.

- **Visualización**: para mostrar fronteras como "mapa de calor caliente"
  (rojo = frontera fuerte), aplicar `1 - B` antes de mostrar.
