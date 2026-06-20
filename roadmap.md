# 🗺️ HOJA DE RUTA: ESTUDIO DE APROBACIÓN SOCIAL Y DINÁMICAS DE PARTICIPACIÓN EN CHATS

> Documento maestro de todas las ramas, fases y módulos posibles para el desarrollo del proyecto.
> Cada fase es un bloque independiente; no requiere completar la anterior al 100% para empezar.

---

## FASE 0 — FUNDAMENTOS TRANSVERSALES (Previo a todo)

| # | Módulo | Descripción |
|---|--------|-------------|
| 0.1 | **Definición del modelo de datos** | Esquema unificado: tabla de mensajes (timestamp, autor, texto, reacciones, thread_id, deleted), tabla de usuarios (id, perfil, metadata), tabla de interacciones (de→a, tipo, timestamp). |
| 0.2 | **Generación de datos sintéticos** | Simulador de chats con perfiles, ruido controlado y ground truth conocido para validar pipelines. |
| 0.3 | **Extracción y limpieza de exports reales** | Parsers para WhatsApp, Telegram, Discord, Slack. Normalización de formato. |
| 0.4 | **Feature engineering base** | Features universales: frecuencia semanal/diaria, ratio reacciones recibidas/enviadas, ratio respuestas recibidas, longitud media de mensaje, hora típica de actividad, ventana de silencio. |

---

## FASE 1 — PIPELINE CORE (Diagnóstico del Grupo)

| # | Módulo | Descripción |
|---|--------|-------------|
| 1.1 | **Clasificador de Perfiles** | Asigna a cada usuario un perfil (Núcleo, Buscador de Validación, Integrado Silencioso, Periférico/Excluido, Espectador Fantasma). |
| 1.2 | **Índice de Aprobación (IA)** | Score individual: reacciones recibidas / mensajes enviados, ajustado por línea base del grupo. |
| 1.3 | **Latencia de Respuesta Diferencial** | Tiempo medio que tarda el grupo en responder a cada usuario, comparado intra-grupo. |
| 1.4 | **Sociograma de Red** | Grafo dirigido de interacciones (quién responde a quién). Centralidad, detección de subgrupos aislados. |
| 1.5 | **Dashboard Base (Streamlit)** | Vista general del grupo: tabla de perfiles, scatter plot IA vs Pertenencia, sociograma básico. |

---

## FASE 2 — NLP Y ANÁLISIS DE CONTENIDO

| # | Módulo | Descripción |
|---|--------|-------------|
| 2.1 | **Detección de jerga interna** | n-gramas, tf-idf, embedding cosine similarity para identificar léxico exclusivo del grupo. |
| 2.2 | **Análisis de sentimiento por perfil** | Polaridad y subjetividad comparada entre perfiles. ¿Los Periféricos escriben más neutro o más negativo? |
| 2.3 | **Detección de mensajes borrados** | Indicador proxy de ansiedad por validación: usuarios que borran mensajes sin respuesta. |
| 2.4 | **Estilo lingüístico imitativo** | Distancia coseno entre embeddings de un usuario y el "líder" del chat. ¿Quién lo imita más? |
| 2.5 | **Topic modeling de espiral del silencio** | Detectar temas donde ciertos perfiles dejan de participar abruptamente. |

---

## FASE 3 — MODELOS PREDICTIVOS (ML)

| # | Módulo | Descripción |
|---|--------|-------------|
| 3.1 | **Predicción de abandono (churn)** | Variables: IA bajo, latencia alta, 0 respuestas directas en N días, sentimiento decayendo. Modelo: XGBoost / Random Forest. |
| 3.2 | **Predicción de escalada hostil** | Detectar cuándo un Periférico pasa de pasivo a agresivo (aumento de negatividad, @menciones a líderes). |
| 3.3 | **Recomendación de intervención** | Sistema de reglas: "usuario X lleva Y días sin respuesta directa → sugerir mensaje de inclusión". |
| 3.4 | **Detección temprana de subgrupos excluyentes** | Formación de cliques que marginan a otros (análisis de densidad de grafo en ventanas de tiempo). |

---

## FASE 4 — EXPERIMENTACIÓN E INTERVENCIÓN

| # | Módulo | Descripción |
|---|--------|-------------|
| 4.1 | **Framework de experimentos A/B** | Diseño: intervención vs control. Ej: "saludo personalizado al Periférico" o "reacciones anónimas". |
| 4.2 | **Sistema de nudges** | Notificaciones automáticas suaves para Integrados Silenciosos y Fantasmas. |
| 4.3 | **Gamificación de participación** | Badges, rachas, puntuación de "calor grupal" — medir su efecto en los distintos perfiles. |
| 4.4 | **Evaluación de impacto** | Métrica pre-post: ¿cambió el IA, la latencia diferencial, la retención? |

---

## FASE 5 — VISUALIZACIÓN AVANZADA Y REPORTING

| # | Módulo | Descripción |
|---|--------|-------------|
| 5.1 | **Dashboard de Salud del Grupo** | Streamlit / Power BI: heatmap semanal, sociograma animado por período, alertas de riesgo. |
| 5.2 | **Reporte automático semanal** | PDF/MD generado: "Esta semana el Periférico X redujo su actividad un 40%". |
| 5.3 | **Vista longitudinal** | Evolución de perfiles en el tiempo: ¿cambia un Integrado Silencioso a Periférico? |
| 5.4 | **Manifold Learning (UMAP/t-SNE)** | Proyección 2D de alta dimensión: UMAP del sociograma (red), embeddings semánticos (temas), trayectorias temporales IA/PA. Detección de islas, puentes, burbujas, transiciones de perfil. Usable desde Fase 2. |

---

## FASE 6 — ESTUDIOS LONGITUDINALES Y PUBLICACIÓN

| # | Módulo | Descripción |
|---|--------|-------------|
| 6.1 | **Validación de hipótesis del marco** | Contrastar estadísticamente las afirmaciones del documento base (ej: "a mayor latencia, mayor abandono"). |
| 6.2 | **Estudio de cohortes** | Seguir a múltiples grupos durante semanas/meses. |
| 6.3 | **Publicación académica / white paper** | Documento formal con metodología, resultados, discusión. |
| 6.4 | **Framework de consultoría** | Metodología empaquetada para aplicar a cualquier grupo/cliente. |

---

## FASE 7 — ADYACENCIAS ESPECIALIZADAS (Opcional / Lateral)

| # | Módulo | Descripción |
|---|--------|-------------|
| 7.1 | **Chats laborales (Slack/Teams)** | Variables extra: fuera de horario, presión de respuesta, productividad vs pertenencia. |
| 7.2 | **Comunidades de aprendizaje** | Impacto en rendimiento académico, miedo a preguntar. |
| 7.3 | **Chats de gaming (Discord)** | Jerarquías por rol de juego, mayor tolerancia al caos, clanes. |
| 7.4 | **Análisis de desinformación** | Perfiles periféricos como vectores de contenido no verificado (buscan aprobación compartiendo sin filtro). |
| 7.5 | **Transcriptómica social (LLMs)** | Embeddings de mensajes para medir "distancia social" y detección de cámaras de eco. |
| 7.6 | **Salud mental y autoestima digital** | Correlación entre métricas de chat y escalas psicométricas externas. |

---

## FASE 8 — PRODUCTO / HERRAMIENTA

| # | Módulo | Descripción |
|---|--------|-------------|
| 8.1 | **API / Microservicio** | Endpoint que recibe export de chat y devuelve diagnóstico completo. |
| 8.2 | **App desktop o web** | Interfaz drag-and-drop para subir logs y obtener reporte. |
| 8.3 | **Bot de moderación** | Bot para Discord/Telegram que monitoriza y alerta en vivo. |

---

## 📊 ÁRBOL DE DEPENDENCIAS

```
Fase 0 ──► Fase 1 ──► Fase 2 ──► Fase 3 ──► Fase 4
                  │                     │
                  ├──► Fase 5           │
                  │                     │
                  └─────────────────────┴──► Fase 6
                                                 │
                                                 └──► Fase 8

Fase 7 (adyacencias) → cualquier punto desde Fase 2 en adelante.
```

---

## NOTAS

- **Ninguna fase requiere la anterior al 100%.** Puedes entrar por Fase 1 básica y saltar a 3.1 sin hacer NLP.
- **Fase 0** es el único prerrequisito duro: sin modelo de datos y features no hay pipeline.
- **Fase 7** son exploraciones laterales; se conectan desde diferentes puntos.
- **Fase 8** es la materialización como producto, al final del recorrido o como MVP temprano de una rama concreta.
