# AGENTS.md — Contexto Persistente del Proyecto

## Identidad del Proyecto

**Nombre:** Estudio de Aprobación Social y Dinámicas de Participación en Chats
**Objetivo:** Analizar, diagnosticar y proponer soluciones sobre búsqueda de aprobación social, sentido de pertenencia y dinámicas de rechazo/exclusión en canales de chat (WhatsApp, Telegram, Discord, Slack, Teams).

---

## Decisiones Arquitectónicas (registro vivo)

| Fecha | Decisión | Fundamento |
|-------|----------|------------|
| 2026-06-19 | **Enfoque incremental por vertical slices.** No abarcar todo el roadmap de golpe. Primer milestone: Diagnóstico Básico (clasificador de 5 perfiles + visualización scatter). | Evitar dispersión, tener algo funcional rápido. |
| 2026-06-19 | **Primero esquema abstracto, luego evaluar datos reales** — no recopilar sin estructura. | El marco teórico define las variables necesarias; los datos se ajustan al esquema, no al revés. |
| 2026-06-19 | **Pipeline en Python (pandas + networkx + scikit-learn + streamlit).** Notebook como entregable inicial. | Stack accesible, reproducible, escalable a producto. |
| 2026-06-19 | **Persistencia de contexto vía AGENTS.md + roadmap.md.** | Trazabilidad entre sesiones. |

---

## Estado Actual del Proyecto

**Fase activa:** Fase 1 (Pipeline Core) — completado
**Sesión actual:** Clasificador de perfiles funcionando sobre datos reales de Bluesky
**Próximo paso:** Visualización (scatter plot IA vs PA), sociograma, refinar umbrales de clasificación

### Resultados del Pipeline

| Métrica | Valor |
|---------|-------|
| Fuente | Bluesky (API pública) |
| Posts raíz procesados | 227 / 236 |
| Mensajes totales extraídos | 378 (227 roots + 151 replies) |
| Autores únicos | 252 |
| Interacciones (reply graph) | 191 |

**Distribución de perfiles:**
| Perfil | Usuarios | % | IA medio | PA medio |
|--------|----------|---|----------|----------|
| Núcleo (Líder) | 34 | 13.5% | 6.48 | 0.65 |
| Buscador de Validación | 0 | 0% | — | — |
| Integrado Silencioso | 66 | 26.2% | 0.63 | 0.51 |
| Periférico / Excluido | 152 | 60.3% | 2.32 | 0.00 |

> Nota: Buscador de Validación no detectado por esparsicidad de la fuente (Bluesky feed público, no chat cerrado). Usuarios con 1 post = 76.6% del total.

---

## Modelo de Datos (ER Refinado v1.0)

> Diagrama detallado completo en `docs/er_diagram.md` (incluye Mermaid ER, tipos, constraints, cardinalidades, reglas de negocio y ejemplo de población).

### Entidades (7)

| Entidad | Tipo | Descripción |
|---------|------|-------------|
| `Grupo` | Maestra | Canal de chat (grupo de WhatsApp, servidor de Discord, etc.) |
| `Usuario` | Maestra | Miembro del grupo (un registro por grupo en que participa) |
| `Mensaje` | Transaccional | Cada mensaje enviado al chat |
| `Reaccion` | Transaccional | Reacción (emoji) a un mensaje |
| `Thread` | Transaccional | Hilo o conversación anidada |
| `Interaccion` | Derivada | Tabla materializada de red social (origen → destino) para sociograma |

### Features Derivados

| Feature | Símbolo | Descripción |
|---------|---------|-------------|
| Índice de Aprobación | IA | reacciones recibidas / mensajes enviados (ajustado por línea base del grupo) |
| Latencia de Respuesta | LR | tiempo medio que tarda el grupo en responder a un usuario |
| Pertenencia Aproximada | PA | score compuesto (frecuencia de interacción + diversidad de destinatarios + uso de jerga) |
| Ratio de Respuesta Directa | RRD | respuestas directas recibidas / mensajes enviados |
| Ventana de Silencio | VS | días sin actividad |

---

## Matriz de Perfiles (Output del Clasificador)

| Perfil | IA | PA | Comportamiento Típico |
|--------|----|----|----------------------|
| Núcleo (Líder) | Alto | Alto | Recibe mayoría de reacciones. Marca tono del grupo. |
| Buscador de Validación | Inestable | Medio | Publica en exceso, abusa de memes. Frustración si lo ignoran. |
| Integrado Silencioso | Medio | Alto | Habla poco, reacciona mucho. Pertenece sin necesitar atención. |
| Periférico / Excluido | Bajo | Bajo | Mensajes ignorados. Alto riesgo de abandono. |
| Espectador Fantasma | Cero | Nulo | No interactúa ni reacciona. Genera desconfianza pasiva. |

---

## Estructura de Carpetas (actual)

```
aprobacion-social-chats/
├── AGENTS.md              ← Este archivo
├── roadmap.md             ← Hoja de ruta maestra (8 fases, 35+ módulos)
├── data/                  ← Datos crudos y procesados (pendiente)
├── notebooks/             ← Jupyter Notebooks (pendiente)
├── src/                   ← Código fuente modular (pendiente)
└── docs/                  ← Documentación adicional (pendiente)
```

---

## Convenciones

- **Idioma:** Español (código en español también, salvo keywords de Python)
- **Formato de timestamps:** ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)
- **Nombres de columnas:** snake_case
- **Perfiles:** solo los 5 definidos en la matriz; cualquier otro se etiqueta como `No_clasificado`

---

## Notas de Sesión

*(Aquí registramos decisiones, dudas abiertas y acuerdos de cada sesión)*

- **Sesión 1 (2026-06-19):** Activación del agente con el documento base. Definición del roadmap. Acuerdo de enfoque incremental. Creación de AGENTS.md y estructura de proyecto.
- **Sesión 2 (2026-06-19):** Diagrama ER detallado (docs/er_diagram.md). 7 entidades con tipos, constraints, cardinalidades, reglas de negocio y ejemplo poblado. Refinado el ER en AGENTS.md.
- **Sesión 3 (2026-06-19):** Extracción de datos reales de Bluesky vía API pública. Extractor en `src/extractors/bsky_extractor.py`. 378 mensajes de 252 autores. Pipeline de clasificación en `src/pipeline.py` con features IA, RRD, PA. Clasificación funcional (4 de 5 perfiles detectados).
