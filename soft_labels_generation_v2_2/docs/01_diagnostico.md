# Por qué un pipeline v2

> Diagnóstico arquitectónico que motiva el rediseño.

## Los seis problemas reportados en v1

Durante la inspección del caso piloto TNG50-87-141934-0-127 con el pipeline actual (v1), se reportaron seis problemas concretos:

| # | Problema | Manifestación |
|---|----------|---------------|
| 19.1 | Spaxels periféricos clasificados como disco | El spaxel (44, 18) tiene poca señal pero se clasifica como disco |
| 19.2 | Sin coherencia espacial entre vecinos | Spaxels contiguos pueden tener clases distintas |
| 19.3 | Centro no siempre es bulbo | El núcleo más brillante puede caer en disco o en incierto |
| 19.4 | hard_mass vs hard_light discrepan | Brazos: incierto en masa, disco en luz |
| 19.5 | Clase "incierto_otro" mezcla dos significados | Componente física residual + baja confianza |
| 19.6 | Umbrales duros demasiado permisivos | Subir el umbral mueve el problema, no lo resuelve |

## El patrón común

A primera vista los seis problemas parecen independientes y resolubles con parches locales. Pero al examinar las causas raíz aparece un patrón común:

> **Todos surgen de decisiones que se toman demasiado pronto, con información incompleta, o que mezclan dos preocupaciones en un solo paso.**

Esta no es coincidencia. Es consecuencia de la **arquitectura monolítica** del pipeline v1, donde cuatro tipos de decisiones conceptualmente ortogonales se entrelazan:

| # | Tipo de decisión | Información necesaria |
|---|------------------|------------------------|
| A | Física (qué es cada partícula) | 3D completo de la simulación |
| B | Calibración (¿coincide con catálogo?) | Fracciones globales por galaxia |
| C | Geometría (proyección y agregación) | Orientación de vista, geometría IFU |
| D | Observacional (PSF, máscara, ruido) | Características instrumentales |

En v1, estas decisiones se toman entrelazadas en el mismo flujo. El resultado:

- **19.3** (centro como disco): ocurre porque la decisión A (qué es cada partícula) se subordina a B (respeta fracciones globales del catálogo). El reescalado iterativo "distribuye" disco al centro para que los totales cuadren.
- **19.1** (periferia como disco): ocurre porque las decisiones C y D están mezcladas: el `valid_mask` observacional no garantiza la calidad estadística necesaria para A.
- **19.2** (sin coherencia espacial): cada spaxel toma su decisión final sin contexto, después de que A, C y D ya están entrelazadas y no se pueden revisar.
- **19.5** (incierto vs other): tres ejes ortogonales (clase física, confianza, validez) se almacenan en un solo tensor.

## Por qué los parches no son suficientes

El documento previo `analisis_etiquetas_aperturenet.docx` propone ocho mejoras que atacan los síntomas correctamente. Pero **no resuelven el acoplamiento subyacente**.

Por ejemplo:
- **Mejora #1** (ε como clasificador primario) cambia el algoritmo, pero la clasificación sigue mezclada con la proyección.
- **Mejora #5** (etiquetas intrínsecas sin PSF) requiere modificar dos pasos del pipeline simultáneamente porque la PSF está aplicada justo antes de la agregación.

Aplicar parches sin rediseñar la arquitectura produce un pipeline que **funciona pero se vuelve cada vez más difícil de extender**. Cada nueva mejora requiere coordinar cambios en múltiples lugares.

## Lo que pierde v1 al estar acoplado

1. **Reuso entre orientaciones**: las 4 vistas de una galaxia repiten la decisión física A cuatro veces. Esa decisión es invariante a la orientación; debería tomarse una sola vez.
2. **Validación independiente**: no se puede validar la clasificación física sin generar todo el pipeline de proyección y degradación.
3. **Iteración rápida**: experimentar con una nueva regla requiere correr todo el pipeline desde cero.
4. **Auditoría**: difícil saber si un problema en las etiquetas finales viene de la clasificación física o del proceso observacional.
5. **Composabilidad**: cada decisión está implícita en el código; no hay un contrato explícito que permita reemplazar un módulo por otro mejor.

## La hipótesis arquitectónica

> Si las cuatro decisiones se separan en módulos independientes con contratos claros de entrada/salida, cada problema reportado se vuelve atacable de forma aislada y aparecen oportunidades de validación cruzada que el pipeline monolítico no permite.

El pipeline v2 está construido sobre esta hipótesis. Los siguientes documentos detallan los principios de diseño (`02_principios.md`), la arquitectura resultante (`03_arquitectura.md`) y los contratos entre módulos (`04_contratos.md`).

## Ganancias esperadas

| Aspecto | v1 | v2 |
|---------|----|----|
| Tiempo total para 40K ejemplos | ~14 días (40K × 35s) | ~5.3 días (10K × 30s + 40K × 5s) |
| Re-experimentar con BarDetector | Re-correr todo | Re-correr solo BarDetector + Phase B |
| Detectar regresión en Classifier | End-to-end visual | Test unitario aislado |
| Añadir nueva clase | Refactorizar varias funciones | Añadir un detector + extender contratos |
| Validar reproducibilidad | Difícil | Cada producto intermedio es checkpoint |

## Lo que NO cambia

El v2 conserva todas las decisiones científicas validadas del v1:

- Vinculación MaNGIA ↔ TNG por snapshot + subhalo + vista
- Alineamiento face-on por momento angular estelar
- Ponderación por masa Y por luz en paralelo
- Uso del catálogo morfológico TNG (ahora como prior, no como constraint)
- Compatibilidad observacional con MaNGA mediante padding 69→74

El cambio es **arquitectónico**, no metodológico. Las mejoras científicas (Mejoras #1-#8 del análisis previo) se incorporan **dentro** de los nuevos módulos, no como capas adicionales.
