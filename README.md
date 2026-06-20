# Estudio de Aprobación Social y Dinámicas de Participación en Chats

Pipeline de análisis psicosocial para diagnosticar dinámicas de aprobación, pertenencia y exclusión en canales de chat (WhatsApp, Telegram, Discord, Slack, Teams, redes sociales).

## Taxonomía de 5 Perfiles

| Perfil | IA | PA | Comportamiento |
|--------|----|----|---------------|
| **Núcleo (Líder)** | Alto | Alto | Recibe la mayoría de reacciones. Marca el tono del grupo. |
| **Buscador de Validación** | Inestable | Medio | Publica en exceso, abusa de memes. Frustración si lo ignoran. |
| **Integrado Silencioso** | Medio | Alto | Habla poco, reacciona mucho. Pertenece sin necesitar atención. |
| **Periférico / Excluido** | Bajo | Bajo | Mensajes ignorados. Alto riesgo de abandono. |
| **Espectador Fantasma** | Cero | Nulo | No interactúa ni reacciona. Genera desconfianza pasiva. |

**IA** = Índice de Aprobación (reacciones recibidas / mensajes enviados)
**PA** = Pertenencia Aproximada (score compuesto: diversidad de interacción + respuestas recibidas + replies)

## Arquitectura

```
aprobacion-social-chats/
├── data/raw/              ← Datos extraídos (JSONL)
├── src/
│   ├── pipeline.py        ← Pipeline: carga → features → clasificación
│   ├── sociograma.py      ← Grafo de red (networkx): centralidad, comunidades, aislamiento
│   ├── manifold.py        ← t-SNE, validación de clusters, Random Forest classifier
│   ├── visualizar_sociograma.py  ← Visualizaciones PNG del grafo
│   ├── dashboard.py       ← Streamlit app interactiva
│   ├── extractors/
│   │   └── bsky_extractor.py  ← Extracción desde Bluesky API
│   └── nlp/
│       ├── jerga_interna.py    ← Detección de léxico grupal (TF-IDF)
│       ├── sentimiento.py      ← Análisis de polaridad por perfil
│       ├── borrados.py         ← Detección de mensajes eliminados
│       ├── estilo_imitativo.py ← Similitud estilística entre usuarios
│       ├── topic_modeling.py   ← LDA + espiral del silencio
│       └── analisis_nlp.py     ← Orquestador NLP
├── docs/
│   └── er_diagram.md      ← Modelo entidad-relación detallado
├── roadmap.md             ← Hoja de ruta maestra (8 fases, 35+ módulos)
└── AGENTS.md              ← Contexto persistente del proyecto
```

## Funcionalidades

### Pipeline Core
- Carga de datos desde JSONL
- Cálculo de features: IA, PA, RRD, outreach, latencia
- Clasificación en 5 perfiles psicosociales
- Reporte textual de distribución

### NLP (Procesamiento de Lenguaje Natural)
- **Jerga interna**: TF-IDF + n-gramas para detectar vocabulario exclusivo del grupo
- **Sentimiento**: Léxico bilingüe (es/en) con análisis de polaridad por perfil
- **Borrados**: Detección automática de mensajes eliminados
- **Estilo imitativo**: Distancia coseno entre vectores de rasgos estilísticos
- **Topic modeling**: LDA para detectar espiral del silencio (temas que los periféricos evitan)

### Sociograma (Análisis de Red)
- Grafo dirigido de interacciones con pesos
- Métricas de centralidad: PageRank, betweenness, closeness, eigenvector
- Detección de comunidades (Louvain)
- Análisis de aislamiento y detección de subgrupos excluyentes
- Visualización coloreada por perfil

### Validación Científica
- **t-SNE / PCA**: Proyección 2D del espacio de features
- **KMeans clustering**: Validación de la taxonomía (silhouette score)
- **Random Forest**: Clasificador con validación cruzada (accuracy >98%)
- Importancia de features: PA (21%) > IA (13%) > RRD (9%)

### Dashboard
- Streamlit con 4 pestañas: Perfiles, Sociograma, NLP, Validación
- Filtros por perfil, IA mínima, PA mínima
- KPIs en vivo, scatter plots, tablas interactivas

## Instalación

```bash
# Clonar
git clone https://github.com/tuusuario/aprobacion-social-chats.git
cd aprobacion-social-chats

# Dependencias
pip install pandas numpy scikit-learn networkx matplotlib plotly streamlit
```

## Uso

```bash
# Pipeline completo
python src/pipeline.py

# Sociograma + visualizaciones
python src/test_sociograma.py

# NLP completo
python src/test_nlp_module.py

# Validación t-SNE + clustering + ML
python src/test_manifold.py

# Dashboard interactivo
streamlit run src/dashboard.py

# Test end-to-end
python src/test_e2e.py
```

## Datos

Los datos de ejemplo provienen de la API pública de Bluesky (378 mensajes, 252 autores). El pipeline es agnóstico a la fuente — cualquier export de chat en formato JSONL funciona.

## Roadmap

| Fase | Estado |
|------|--------|
| 0. Fundamentos | ✅ |
| 1. Pipeline Core | ✅ |
| 2. NLP | ✅ |
| 3. Sociograma + Validación | ✅ |
| 4. Dashboard | ✅ |
| 5. Modelos Predictivos | ⏳ Pendiente |
| 6. Experimentación | 📅 Futuro |
| 7. Visualización Avanzada | 📅 Futuro |
| 8. Estudios Longitudinales | 📅 Futuro |

## Marco Teórico

Este estudio se basa en teorías de psicología social:
- **Teoría de la Espiral del Silencio** (Noelle-Neumann): los periféricos se autocensuran
- **Teoría de la Identidad Social** (Tajfel): pertenencia como motor de participación
- **Refuerzo Social**: la aprobación (likes, replies) condiciona la conducta futura
- **Anomia Digital**: aislamiento en entornos virtuales y sus consecuencias

## Licencia

MIT
