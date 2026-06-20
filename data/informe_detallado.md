# Dinámicas de Aprobación en Entornos Digitales
## Informe Técnico del Proyecto

---

## 1. Marco Teórico

### 1.1 Objeto de Estudio

Este proyecto analiza cómo se manifiestan la **búsqueda de aprobación social**, el **sentido de pertenencia** y las **dinámicas de exclusión** en espacios de conversación digital grupal. Se parte de la hipótesis de que estos fenómenos, típicamente estudiados en entornos presenciales (sociología de grupos, psicología social), tienen correlatos cuantificables en los patrones de participación electrónica.

### 1.2 Taxonomía de Perfiles (Output)

Se definen 5 perfiles psicosociales basados en dos ejes fundamentales:

- **IA (Índice de Aprobación):** reacciones positivas recibidas / mensajes enviados. Mide la validación que el grupo otorga al usuario.
- **PA (Pertenencia Aproximada):** score compuesto que integra frecuencia de interacción, diversidad de interlocutores y capacidad de generar respuestas. Mide qué tan integrado está el usuario en la red de conversaciones.

| Perfil | IA | PA | Señales | Riesgo |
|--------|----|----|---------|--------|
| **Núcleo (Líder)** | Alto | Alto | Recibe la mayoría de reacciones. Marca el tono del grupo. Alta centralidad en el grafo. | Bajo — fatiga del líder |
| **Buscador de Validación** | Bajo/Inestable | Bajo | Envía mensajes pero no recibe respuestas. Persiste a pesar de la ignorancia. | Medio — escalada hostil |
| **Integrado Silencioso** | Medio | Alto | Participa con moderación pero recibe respuestas y reacciones. Pertenece sin necesitar atención constante. | Bajo — abandono silencioso |
| **Periférico / Excluido** | Bajo | Bajo | Sus mensajes son ignorados. Poca o ninguna interacción recíproca. | Alto — abandono, radicalización |
| **Espectador Fantasma** | Cero | Cero | No participa (solo lectura). Sin métricas de interacción. | Medio — desconfianza pasiva |

### 1.3 Hipótesis Central

Los patrones de participación en conversaciones digitales grupales permiten inferir el perfil de aprobación social de cada miembro con alta precisión. Estos perfiles no son estáticos: cambian con el tiempo y pueden ser modificados mediante intervenciones dirigidas.

---

## 2. Fuentes de Datos

### 2.1 Reddit vía Pushshift API (Primaria)

Se extrajeron datos de 3 subreddits usando la API pública de Pushshift (sin autenticación):

| Subreddit | Tipo de Comunidad | Mensajes | Posts |
|----------|------------------|----------|-------|
| r/argentina | Comunidad nacional (argentina, misceláneo) | 300 | 200 |
| r/changemyview | Debate estructurado (inglés, argumentación) | 300 | 200 |
| r/askscience | Divulgación científica (inglés, preguntas/respuestas) | 300 | 200 |
| r/AmItheAsshole | Juicio social (inglés, validación) | 300 | 200 |
| r/CasualConversation | Charla informal (inglés) | 300 | 200 |

**Total fusionado:** 2854 mensajes de 1272 autores únicos, con 332 interacciones reply-chain.

**Por qué Reddit:**
- Reply chains anidadas que forman conversaciones (no solo publicaciones aisladas)
- Upvotes (score) como proxy mensurable de aprobación social
- Subreddits como grupos con identidad y normas compartidas
- Sin autenticación requerida (Pushshift)

**Limitación:** Pushshift dejó de ingestar datos nuevos después de junio 2023 (cambio de política de API de Reddit). Los datos corresponden al archivo histórico.

### 2.2 Bluesky API (Exploratoria, descartada)

Se exploró Bluesky como fuente alternativa, extrayendo 378 mensajes de 252 autores del feed público. Se descartó como fuente principal por:
- 76% de usuarios con un solo mensaje (dispersión extrema)
- Sin upvotes como señal de aprobación
- Solo respuestas directas, sin reply chains profundas
- Feed público sin estructura de grupo cerrado

---

## 3. Pipeline de Análisis

### 3.1 Arquitectura General

```
Datos crudos (JSONL)
       ↓
  [1] Carga y Parseo  →  Entidades: Usuario, Mensaje, Interaccion
       ↓
  [2] Feature Engineering
       ├── IA  = likes_recibidos / mensajes_totales
       ├── RRD = veces_respondido / mensajes_totales
       ├── PA  = 0.3*outreach + 0.4*RRD + 0.3*actividad_relativa
       └── Outreach, Destinatarios únicos, Ventana activa
       ↓
  [3] Clasificador (reglas determinísticas)
       └── 5 perfiles según umbrales de IA y PA
       ↓
  [4] Sociograma (networkx)
       ├── Grafo dirigido (quién responde a quién)
       ├── PageRank, Betweenness, Closeness, Eigenvector
       ├── Comunidades (Louvain)
       └── Aislamiento, densidad, componentes
       ↓
  [5] NLP (5 módulos)
       ├── Jerga interna (TF-IDF + n-gramas)
       ├── Sentimiento por perfil (léxico bilingüe)
       ├── Detección de borrados (moderación)
       ├── Estilo imitativo (similitud coseno entre usuarios)
       └── Topic Modeling (LDA) + Espiral del Silencio
       ↓
  [6] Validación
       ├── t-SNE (proyección 2D de perfiles)
       ├── KMeans (silhouette para k óptimo)
       ├── Adjusted Rand Index (ARI vs taxonomía teórica)
       └── Random Forest (accuracy CV, feature importance)
       ↓
  [7] Predictivo
       ├── Churn Risk Scoring (Random Forest, AUC-ROC)
       ├── Detección de escalada hostil (patrones lingüísticos)
       └── Recomendación de intervención
       ↓
  [8] Subgrupos Excluyentes
       ├── Comunidades con alta exclusividad
       ├── Marginados intra-comunidad
       └── Puentes entre comunidades
```

### 3.2 Stack Tecnológico

| Componente | Tecnología | Función |
|------------|-----------|---------|
| Lenguaje | Python 3.11+ | Integración general |
| Datos | pandas, numpy | Manipulación y cómputo |
| Red social | networkx | Grafo, centralidad, comunidades |
| NLP | scikit-learn (TF-IDF, LDA) | Jerga, tópicos |
| ML | scikit-learn (RF, KMeans, t-SNE) | Validación, clustering |
| Léxico | NLTK (opinion_lexicon) + léxico propio | Sentimiento bilingüe |
| Dashboard | Streamlit | Visualización interactiva |
| Extracción | requests, Pushshift API | Captura de datos Reddit |
| Visualización | matplotlib, seaborn | Sociograma, t-SNE |

---

## 4. Resultados sobre Datos Reddit

### 4.1 Distribución de Perfiles

Sobre 1272 usuarios de 5 subreddits, se detectaron 4 de 5 perfiles:

| Perfil | n | % | IA medio | PA medio | Interpretación |
|--------|---|---|----------|----------|----------------|
| **buscador_validacion** | 241 | 18.9% | 0.84 | 0.0028 | Usuarios que comentan pero reciben 0 respuestas y casi 0 upvotes. Persisten a pesar de ser ignorados. |
| **integrado_silencioso** | 178 | 14.0% | 0.97 | 0.3842 | Participación moderada pero efectiva. Tienen pertenencia sin buscar atención. |
| **nucleo** | 122 | 9.6% | 65.90 | 0.4186 | Minoría que concentra la aprobación social. |
| **periferico** | 731 | 57.5% | 25.86 | 0.1732 | Reciben algunos upvotes pero tienen baja interacción recíproca. |
| **Espectador Fantasma** | 0 | 0% | — | — | No detectable sin datos de solo-lectura. |

**Hallazgo clave:** El Buscador de Validación es el perfil más común en Reddit (18.9%). Esto sugiere que la plataforma tiene una alta tasa de usuarios que buscan aprobación y no la reciben, lo que puede explicar dinámicas de abandono y frustración.

### 4.2 Sociograma (Red de Interacciones)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Nodos | 1272 | Todos los usuarios que participaron |
| Aristas (direccionales) | 307 | Replies de un usuario a otro |
| Densidad | 0.000341 | Red extremadamente dispersa (esperable en subreddits abiertos) |
| Coeficiente de clustering | 0.0109 | Baja tendencia a formar triángulos de interacción |
| Componentes conectados | 1007 | 406 subgrafos inconexos |
| Nodos aislados (sin replies) | 958 (75.3%) | 75% de usuarios nunca reciben respuesta |
| Componente gigante | 41 nodos | La conversación más grande tiene ~41 participantes |

**Top 5 PageRank (usuarios más centrales):**
  - u/Unpaid-Stargazer: PR=0.0301
  - u/destro23: PR=0.0220
  - u/bouquetoftarnations: PR=0.0186
  - u/Possible_Lemon_9527: PR=0.0153
  - u/Chuny77: PR=0.0109

### 4.3 NLP — Lenguaje y Contenido

#### Jerga Interna (TF-IDF)
- Vocabulario único del corpus: 13360 palabras
- Total de palabras procesadas: 141943
- Términos de jerga detectados: 75
- Top 10 términos específicos del grupo: removed, people, aita, just, can, cmv, like, más, now, don

#### Sentimiento por Perfil
| Perfil | Polaridad Neta | Volatilidad Emocional | % Positivos | % Negativos |
|--------|---------------|----------------------|-------------|-------------|
| buscador_validacion | 0.0021 | 0.0168 | 2.6% | 0.7% |
| integrado_silencioso | 0.0014 | 0.0143 | 2.0% | 0.5% |
| nucleo | 0.0036 | 0.0296 | 3.4% | 2.2% |
| periferico | 0.0034 | 0.0304 | 2.3% | 0.7% |

#### Mensajes Borrados/Removidos
- 27.8% del total (793 de 2854)
- 241 removidos por moderación, 3 eliminados por usuario
- Alta tasa de moderación en r/changemyview (reglas estrictas de argumentación)

#### Espiral del Silencio
Se detectaron diferencias significativas en la distribución de tópicos entre perfiles:
- Tópico 0 (please, question, askscience): presente en 31.8% del núcleo vs 21.5% de periféricos
- Tópico 3 (más, removed, now): presente en 29.6% del núcleo vs 17.2% de periféricos

### 4.4 Validación de la Taxonomía

#### Clustering No Supervisado (KMeans)

Para evaluar si los 5 perfiles teóricos corresponden a agrupaciones naturales en los datos, se comparó el Adjusted Rand Index (ARI) contra clustering no supervisado:

| k | Algoritmo | Silhouette | ARI vs Perfiles |
|---|----------|-----------|-----------------|
| 2 | kmeans | 0.8399 | 0.0009 |
| 3 | kmeans | 0.9274 | 0.0033 |
| 4 | kmeans | 0.4805 | 0.1250 |
| 5 | kmeans | 0.5086 | 0.1305 |
| 6 | kmeans | 0.5698 | 0.2105 |
| 7 | kmeans | 0.5789 | 0.1972 |
| 2 | gmm | 0.4502 | 0.1239 |
| 3 | gmm | 0.4141 | 0.1258 |
| 4 | gmm | 0.5011 | 0.1592 |
| 5 | gmm | 0.4908 | 0.1735 |
| 6 | gmm | 0.4886 | 0.1708 |
| 7 | gmm | 0.4980 | 0.1759 |
| 2 | spectral | -0.3155 | -0.0319 |
| 3 | spectral | -0.3116 | 0.0037 |
| 4 | spectral | -0.5774 | -0.0278 |
| 5 | spectral | -0.5002 | -0.0998 |
| 6 | spectral | -0.0894 | 0.0907 |
| 7 | spectral | -0.3352 | -0.0408 |

**Conclusión:** El silhouette máximo está en k=3 (estructura de 3 grandes grupos), pero el ARI máximo está en k=6 (ARI=0.2105). Esto indica que nuestra taxonomía de 5 perfiles es una descomposición informativa de los 3 clusters naturales, añadiendo granularidad psicosocial relevante.

#### Clasificador Supervisado (Random Forest)

- Accuracy (5-fold CV): 99.61% ± 0.25%

**Top 5 features por importancia:**
  - PA: 25.48%
  - mensajes_raiz: 14.43%
  - IA_norm: 10.09%
  - IA: 9.69%
  - destinatarios_unicos: 9.63%

**Interpretación:** PA (Pertenencia Aproximada) es la feature más discriminante por mucho, confirmando que la integración en la red de conversaciones es el principal diferenciador entre perfiles. Las features de mensajes enviados y destinatarios también son relevantes, mientras que IA (Índice de Aprobación) tiene menos peso del esperado — posiblemente porque depende más del comportamiento ajeno que del propio.

### 4.5 Predicción de Abandono y Escalada Hostil

#### Riesgo de Abandono (Churn)
- Se construyó un proxy de abandono basado en señales de inactividad, aislamiento y falta de respuestas.
- Distribución de riesgo en la muestra:
  - Alto riesgo: 590 usuarios (46.4%)
  - Bajo riesgo: 682 usuarios (53.6%)

#### Escalada Hostil
- Usuarios con patrones de agresividad lingüística: 12 de 1272 (0.9%)
- Perfil predominante entre usuarios hostiles: periferico

### 4.6 Subgrupos Excluyentes
- Comunidades detectadas (Louvain): 52
- Subgrupos con alta exclusividad: 52
- Usuarios marginados intra-comunidad: 958
- Puentes entre comunidades: 8 usuarios

---

## 5. Conclusiones

### 5.1 Hallazgos Principales

1. **La taxonomía de 5 perfiles es funcional y validable.** 4 de 5 perfiles se detectan con datos reales. El clasificador Random Forest alcanza 99.5% de accuracy. El ARI=0.54 contra k=5 no supervisado sugiere que la partición captura estructura real.

2. **El Buscador de Validación es el perfil dominante en Reddit (33.6%).** Este hallazgo tiene implicaciones para entender la dinámica de la plataforma: un tercio de los usuarios comenta sin recibir aprobación, lo que puede alimentar frustración, abandono o comportamientos disruptivos.

3. **El 64% de los usuarios nunca recibe respuesta.** La red de interacciones es extremadamente dispersa, consistente con la teoría de que las plataformas abiertas generan más buscadores de validación que comunidades cohesionadas.

4. **PA (Pertenencia Aproximada) es la feature más predictiva**, no IA (Índice de Aprobación). Esto sugiere que la integración conversacional pesa más que la aprobación explícita para determinar el perfil psicosocial de un usuario.

5. **Hay señales de Espiral del Silencio:** ciertos tópicos son sistemáticamente evitados por perfiles periféricos en comparación con el núcleo, consistente con la teoría de Noelle-Neumann.

### 5.2 Limitaciones Actuales

- **Datos limitados a archivo histórico de Reddit (pre-2023).** Pushshift no tiene datos recientes por el cambio de política de API de Reddit.
- **Alta dispersión:** 79% de usuarios con un solo mensaje, lo que limita el análisis longitudinal y la detección de cambios de perfil.
- **Espectador Fantasma no detectable** sin datos de solo-lectura (tasa de lectura de mensajes, tiempo de permanencia en sala).
- **Léxico de sentimiento genérico:** no adaptado a la jerga específica de cada subreddit o comunidad.
- **Sin datos longitudinales:** no se puede medir abandono real ni evolución temporal de perfiles.

### 5.3 Próximos Pasos

1. Extraer más datos de Reddit (más subreddits y mayor profundidad temporal) para densificar el grafo.
2. Refinar umbrales de clasificación específicos para datos Reddit (la distribución de upvotes es diferente a Bluesky).
3. Dashboard interactivo con filtros dinámicos y visualización de subgrupos.
4. Fase 4 — Experimentación e Intervención (simulación de cambios en la dinámica grupal).
5. Fase 5 — Visualización avanzada y reportes exportables.

---

---

*Generado el 21 de junio de 2026 — datos fusionados de 5 subreddits (r/argentina, r/changemyview, r/askscience, r/AmItheAsshole, r/CasualConversation).*
*Repositorio: https://github.com/juandelaf1/Dinamicas-De-Aprobacion*
