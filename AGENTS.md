# AGENTS.md — Contexto Persistente del Proyecto

## Identidad del Proyecto

**Nombre:** Dinámicas de Aprobación
**Repositorio:** https://github.com/juandelaf1/Dinamicas-De-Aprobacion
**Objetivo:** Analizar, diagnosticar y proponer soluciones sobre búsqueda de aprobación social, sentido de pertenencia y dinámicas de rechazo/exclusión en canales de chat (WhatsApp, Telegram, Discord, Slack, Teams).

---

## Decisiones Arquitectónicas (registro vivo)

| Fecha | Decisión | Fundamento |
|-------|----------|------------|
| 2026-06-19 | **Enfoque incremental por vertical slices.** No abarcar todo el roadmap de golpe. Primer milestone: Diagnóstico Básico (clasificador de 5 perfiles + visualización scatter). | Evitar dispersión, tener algo funcional rápido. |
| 2026-06-19 | **Primero esquema abstracto, luego evaluar datos reales** — no recopilar sin estructura. | El marco teórico define las variables necesarias; los datos se ajustan al esquema, no al revés. |
| 2026-06-19 | **Pipeline en Python (pandas + networkx + scikit-learn + streamlit).** Notebook como entregable inicial. | Stack accesible, reproducible, escalable a producto. |
| 2026-06-19 | **Persistencia de contexto vía AGENTS.md + roadmap.md.** | Trazabilidad entre sesiones. |
| 2026-06-20 | **Reddit (Pushshift) como fuente principal.** Reply chains + upvotes = mejor sennal de aprobacion. Sin auth. | Bluesky feed público es muy disperso. |
| 2026-06-20 | **Stop words bilingüe (ES+EN) en NLP.** Los datos Reddit mezclan español e inglés. | `stop_words="english"` dejaba pasar "que", "los", "del". |
| 2026-06-20 | **Hostilidad requiere >=2 patrones.** Eliminado patrón "generalizacion" por falsos positivos con palabras comunes. | 578/581 usuarios detectados como hostiles era inservible. |

---

## Estado Actual del Proyecto

**Fase activa:** Fase 2-3 (NLP + Sociograma + Validación) — completado
**Sesión actual:** Pipeline completo sobre datos Reddit (581 usuarios, 900 mensajes)
**Próximo paso:** Extraer más datos, refinar umbrales para Reddit, Fase 4

### Resultados del Pipeline (Reddit)

| Métrica | Valor |
|---------|-------|
| Fuente | Reddit (r/argentina, r/changemyview, r/askscience) |
| Mensajes totales | 900 |
| Autores únicos | 581 |
| Interacciones (reply graph) | 239 |
| Usuarios con 1 mensaje | 79.0% |

**Distribución de perfiles (4/5 detectados):**
| Perfil | Usuarios | % | IA medio | PA medio |
|--------|----------|---|----------|----------|
| Núcleo (Líder) | 63 | 10.8% | 125.76 | 0.60 |
| Buscador de Validación | 195 | 33.6% | 0.82 | 0.00 |
| Integrado Silencioso | 137 | 23.6% | 5.76 | 0.45 |
| Periférico / Excluido | 186 | 32.0% | 19.97 | 0.24 |

> Buscador de Validación detectado por primera vez con Reddit. Perfil más numeroso.

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
Dinamicas-De-Aprobacion/
├── AGENTS.md              ← Este archivo
├── roadmap.md             ← Hoja de ruta maestra (8 fases, 35+ módulos)
├── data/
│   ├── raw/               ← Datos crudos (JSONL)
│   │   ├── reddit_merged_20260620.jsonl      ← 900 msgs, 581 users (3 subreddits)
│   │   ├── bluesky_threads_*.jsonl           ← 378 msgs, 252 users
│   │   └── reddit_*.jsonl                    ← Extractores individuales
│   └── resumen_ejecutivo.txt                 ← Reporte completo
├── src/
│   ├── pipeline.py         ← Core: features (IA, PA, RRD) + clasificación
│   ├── sociograma.py       ← Grafo dirigido, centralidad, comunidades
│   ├── visualizar_sociograma.py ← PNG del sociograma
│   ├── manifold.py         ← t-SNE, clustering, Random Forest
│   ├── predictivo.py       ← Churn, escalada hostil, intervención
│   ├── subgrupos.py        ← Subgrupos excluyentes, puentes
│   ├── dashboard.py        ← Streamlit (4 tabs)
│   ├── resumen_final.py    ← Genera resumen ejecutivo
│   ├── extractors/
│   │   ├── bsky_extractor.py
│   │   └── reddit_extractor.py
│   └── nlp/
│       ├── stop_words.py       ← SW combinados ES+EN
│       ├── jerga_interna.py    ← TF-IDF + n-gramas
│       ├── sentimiento.py      ← Léxico bilingüe
│       ├── borrados.py         ← Detección de eliminados
│       ├── estilo_imitativo.py ← Similitud estilística
│       ├── topic_modeling.py   ← LDA + espiral del silencio
│       └── analisis_nlp.py     ← Orquestador
├── docs/                  ← Documentación
└── roadmap.md
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

- **Sesión 1 (2026-06-19):** Definición del roadmap. Enfoque incremental. AGENTS.md + estructura.
- **Sesión 2 (2026-06-19):** Diagrama ER detallado. 7 entidades con constraints y cardinalidades.
- **Sesión 3 (2026-06-19):** Extractor Bluesky + pipeline clasificación. 378 msgs, 252 users. 3 perfiles.
- **Sesión 4 (2026-06-20):** Fase 2 NLP completa (5 módulos). Fase 3 Sociograma, t-SNE, Predictivo, Subgrupos. Dashboard Streamlit.
- **Sesión 5 (2026-06-20):** Extractor Reddit (Pushshift). 900 msgs, 581 users, 4 perfiles (Buscador Validación aparece). Validación RF 99.5%.
- **Sesión 6 (2026-06-20):** Fixes calidad: stop words ES+EN en NLP, threshold hostilidad (578→5 users). Repo renombrado a `Dinamicas-De-Aprobacion`.
