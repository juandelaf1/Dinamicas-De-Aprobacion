import pandas as pd

from .jerga_interna import detectar_jerga_interna, obtener_terminos_grupo
from .sentimiento import analizar_sentimiento_perfil
from .borrados import detectar_borrados, resumen_borrados
from .estilo_imitativo import comparar_estilo_imitativo
from .topic_modeling import extraer_topicos_espiral


class AnalisisNLP:
    """
    Orquestador de todo el analisis NLP.
    Coordina los 5 modulos de Phase 2 y produce un reporte unificado.
    """

    def __init__(self, df_mensajes: pd.DataFrame, df_perfiles: pd.DataFrame):
        self.df_mensajes = df_mensajes
        self.df_perfiles = df_perfiles
        self.resultados = {}

    def ejecutar_jerga_interna(self, **kwargs) -> dict:
        """Module 2.1: Deteccion de jerga interna del grupo."""
        params = {"ngram_range": (1, 3), "max_features": 500, "umbral_exclusividad": 0.15}
        params.update(kwargs)
        self.resultados["jerga"] = detectar_jerga_interna(
            self.df_mensajes, **params
        )
        self.resultados["terminos_grupo"] = obtener_terminos_grupo(self.df_mensajes)
        return self.resultados["jerga"]

    def ejecutar_sentimiento(self, **kwargs) -> pd.DataFrame:
        """Module 2.2: Analisis de sentimiento por perfil."""
        result = analizar_sentimiento_perfil(
            self.df_mensajes, self.df_perfiles, **kwargs
        )
        self.resultados["sentimiento_agregado"], self.resultados["sentimiento_detalle"] = result
        return self.resultados["sentimiento_agregado"]

    def ejecutar_borrados(self, **kwargs) -> dict:
        """Module 2.3: Deteccion de mensajes borrados."""
        df_borrados = detectar_borrados(self.df_mensajes, **kwargs)
        self.resultados["borrados_df"] = df_borrados
        self.resultados["resumen_borrados"] = resumen_borrados(df_borrados, self.df_perfiles)
        return self.resultados["resumen_borrados"]

    def ejecutar_estilo(self, **kwargs) -> dict:
        """Module 2.4: Analisis de estilo linguistico imitativo."""
        self.resultados["estilo"] = comparar_estilo_imitativo(
            self.df_mensajes, self.df_perfiles, **kwargs
        )
        return self.resultados["estilo"]

    def ejecutar_topicos(self, **kwargs) -> dict:
        """Module 2.5: Topic modeling y espiral del silencio."""
        self.resultados["topicos"] = extraer_topicos_espiral(
            self.df_mensajes, self.df_perfiles, **kwargs
        )
        return self.resultados["topicos"]

    def ejecutar_todo(self, **kwargs) -> dict:
        """Ejecuta los 5 modulos NLP en secuencia."""
        self.ejecutar_jerga_interna(**kwargs.pop("jerga_kwargs", {}))
        self.ejecutar_sentimiento(**kwargs.pop("sentimiento_kwargs", {}))
        self.ejecutar_borrados(**kwargs.pop("borrados_kwargs", {}))
        self.ejecutar_estilo(**kwargs.pop("estilo_kwargs", {}))
        self.ejecutar_topicos(**kwargs.pop("topicos_kwargs", {}))
        return self.resultados

    def reporte_texto(self) -> str:
        """Genera reporte legible de todos los resultados."""
        lineas = []
        lineas.append("=" * 60)
        lineas.append("REPORTE DE ANALISIS NLP - Fase 2")
        lineas.append("=" * 60)

        # 2.1 Jerga
        if "jerga" in self.resultados:
            j = self.resultados["jerga"]
            n_jerga = len(j.get("terminos_jerga", []))
            lineas.append(f"\n[2.1] Jerga interna: {n_jerga} terminos candidatos")
            if n_jerga > 0:
                lineas.append(f"  Top: {', '.join(j['terminos_jerga'][:10])}")

        if "terminos_grupo" in self.resultados:
            tg = self.resultados["terminos_grupo"]
            lineas.append(f"  Vocabulario unico: {tg.get('vocabulario_unico', 0)} palabras")
            lineas.append(f"  Total palabras: {tg.get('total_palabras', 0)}")
            uni = tg.get("unigramas", {})
            if uni:
                top5 = list(uni.keys())[:5]
                lineas.append(f"  Top unigramas: {', '.join(top5)}")

        # 2.2 Sentimiento
        if "sentimiento_agregado" in self.resultados:
            sa = self.resultados["sentimiento_agregado"]
            lineas.append(f"\n[2.2] Sentimiento por perfil:")
            for _, row in sa.iterrows():
                lineas.append(
                    f"  {row['perfil']:<25s} sent={row['sentimiento_medio']:.4f}  "
                    f"volat={row['volatilidad_emocional']:.4f}  "
                    f"pos={row['proporcion_positivos']:.1%}  "
                    f"neg={row['proporcion_negativos']:.1%}"
                )

        # 2.3 Borrados
        if "resumen_borrados" in self.resultados:
            rb = self.resultados["resumen_borrados"]
            lineas.append(f"\n[2.3] Mensajes borrados:")
            lineas.append(f"  {rb['mensajes_borrados']} / {rb['total_mensajes']} ({rb['porcentaje_borrado']}%)")
            if rb.get("por_motivo"):
                lineas.append(f"  Motivos: {rb['por_motivo']}")
            if rb.get("por_perfil"):
                lineas.append(f"  Por perfil: {rb['por_perfil']}")

        # 2.4 Estilo imitativo
        if "estilo" in self.resultados:
            est = self.resultados["estilo"]
            n_pares = len(est.get("pares_similares", []))
            n_imitadores = len(est.get("imitadores_sospechosos", []))
            lineas.append(f"\n[2.4] Estilo imitativo:")
            lineas.append(f"  Pares con estilo similar: {n_pares}")
            lineas.append(f"  Posibles imitadores: {n_imitadores}")
            if n_pares > 0:
                top3 = est["pares_similares"][:3]
                for p in top3:
                    lineas.append(
                        f"  @{p['usuario_a'][-12:]} <-> @{p['usuario_b'][-12:]}  "
                        f"sim={p['similitud_estilo']:.3f}"
                    )

        # 2.5 Topic modeling
        if "topicos" in self.resultados:
            tp = self.resultados["topicos"]
            n_evitados = len(tp.get("topicos_evitados", []))
            lineas.append(f"\n[2.5] Topic modeling & Espiral del silencio:")
            lineas.append(f"  Topicos evitados por perifericos: {n_evitados}")
            for te in tp.get("topicos_evitados", []):
                bandera = " *** POSIBLE ESPIRAL ***" if te.get("posible_espiral_silencio") else ""
                lineas.append(
                    f"  Topico {te['topico']}: {', '.join(te['palabras_clave'][:3])}  "
                    f"nucleo={te['presencia_nucleo']:.1%} perif={te['presencia_periferico']:.1%}"
                    f"{bandera}"
                )
            if "distribucion_topicos_por_perfil" in tp:
                dist = tp["distribucion_topicos_por_perfil"]
                lineas.append("\n  Distribucion topicos por perfil:")
                lineas.append(f"  {dist.to_string()}")

        lineas.append("\n" + "=" * 60)
        return "\n".join(lineas)
