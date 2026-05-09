import pandas as pd
from pathlib import Path

from src.evaluation.query_token_tracker import QueryTokenTracker


def _estimate_tokens(text: str) -> int:
    """
    Estima tokens usando tiktoken (cl100k_base, compatible con GPT-4/Gemini).
    Fallback: len(text) // 4 si tiktoken no está disponible.

    NOTA: MSGraphRAG no expone el cliente LLM interno, así que los tokens
    son una estimación basada en el input (query + contextos) y el output
    (respuesta). El conteo real puede diferir según el system prompt interno
    de graphrag y el número de llamadas LLM que haga internamente.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(len(text) // 4, 1)


class MSGraphRAG:
    def __init__(self, workspace_dir: str = "./ms_graphrag_workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.output_dir = self.workspace_dir / "output"
        self._config = None
        self._df = None
        self._query_tracker: QueryTokenTracker | None = None

    def attach_token_tracker(self, tracker: QueryTokenTracker) -> None:
        """
        Engancha un QueryTokenTracker.
        Los tokens se estiman con tiktoken sobre (query + contextos) y respuesta,
        ya que graphrag no expone el cliente LLM directamente.
        """
        self._query_tracker = tracker

    def _load_config(self):
        from graphrag.config.load_config import load_config
        self._config = load_config(self.workspace_dir)
        return self._config

    def _load_dataframes(self):
        if self._df is not None:
            return self._df
        print("📂 Cargando índice MS-GraphRAG...")
        self._df = {
            "entities":          pd.read_parquet(self.output_dir / "entities.parquet"),
            "communities":       pd.read_parquet(self.output_dir / "communities.parquet"),
            "community_reports": pd.read_parquet(self.output_dir / "community_reports.parquet"),
            "text_units":        pd.read_parquet(self.output_dir / "text_units.parquet"),
            "relationships":     pd.read_parquet(self.output_dir / "relationships.parquet"),
        }
        print(f"   ✅ Entidades    : {len(self._df['entities']):,}")
        print(f"   ✅ Relaciones   : {len(self._df['relationships']):,}")
        print(f"   ✅ Comunidades  : {len(self._df['communities']):,}")
        print(f"   ✅ Reports      : {len(self._df['community_reports']):,}")
        print(f"   ✅ Text units   : {len(self._df['text_units']):,}")
        return self._df

    def load(self):
        self._load_config()
        self._load_dataframes()
        return self

    def _record_estimated_tokens(
        self, query: str, response: str, contexts: list[str]
    ) -> None:
        """Registra tokens estimados en el tracker si está activo."""
        if not self._query_tracker:
            return
        context_text = " ".join(contexts)
        prompt_est = _estimate_tokens(query + " " + context_text)
        completion_est = _estimate_tokens(response)
        self._query_tracker.record(
            prompt_tokens=prompt_est,
            completion_tokens=completion_est,
        )

    async def local_search(
        self,
        query: str,
        community_level: int = 2,
        response_type: str = "Single Paragraph",
    ) -> tuple[str, list[str]]:
        """
        Local search — busca entidades y relaciones específicas.
        Firma 2.7.1: (config, entities, communities, community_reports,
                       text_units, relationships, covariates, community_level,
                       response_type, query)
        """
        from graphrag.api import local_search

        df = self._load_dataframes()
        config = self._config

        response, context = await local_search(
            config=config,
            entities=df["entities"],
            communities=df["communities"],
            community_reports=df["community_reports"],
            text_units=df["text_units"],
            relationships=df["relationships"],
            covariates=None,
            community_level=community_level,
            response_type=response_type,
            query=query,
        )

        contexts = self._extract_contexts(context)
        self._record_estimated_tokens(query, str(response), contexts)
        return str(response), contexts

    async def global_search(
        self,
        query: str,
        community_level: int = 2,
        response_type: str = "Single Paragraph",
    ) -> tuple[str, list[str]]:
        """
        Global search — sintetiza sobre community reports del grafo completo.
        Firma 2.7.1: (config, entities, communities, community_reports,
                       community_level, dynamic_community_selection,
                       response_type, query)
        """
        from graphrag.api import global_search

        df = self._load_dataframes()
        config = self._config

        response, context = await global_search(
            config=config,
            entities=df["entities"],
            communities=df["communities"],
            community_reports=df["community_reports"],
            community_level=community_level,
            dynamic_community_selection=False,
            response_type=response_type,
            query=query,
        )

        contexts = self._extract_contexts(context)
        self._record_estimated_tokens(query, str(response), contexts)
        return str(response), contexts

    def _extract_contexts(self, context) -> list[str]:
        """Extrae texto del contexto devuelto por GraphRAG 2.7.1."""
        contexts = []
        try:
            if isinstance(context, dict):
                # 1. Sources primero (más relevantes para RAGAS)
                if "sources" in context and isinstance(context["sources"], pd.DataFrame):
                    df = context["sources"]
                    if "text" in df.columns:
                        texts = df["text"].dropna().astype(str).tolist()
                        # Filtrar chunks vacíos o demasiado cortos (basura)
                        texts = [t for t in texts if len(t.strip()) > 50]
                        contexts.extend(texts)

                # 2. Reports solo si no hay suficientes sources
                if len(contexts) < 3:
                    if "reports" in context and isinstance(context["reports"], pd.DataFrame):
                        df = context["reports"]
                        if "content" in df.columns:
                            reports = df["content"].dropna().astype(str).tolist()
                            reports = [r for r in reports if len(r.strip()) > 50]
                            contexts.extend(reports[:3])  # máximo 3 reports

                # 3. Entities: OMITIR — descripciones cortas dañan context_precision

            elif isinstance(context, str) and context.strip():
                contexts = [context]

        except Exception as e:
            print(f"⚠️ _extract_contexts error: {e}")

        # Limitar total de contextos para no diluir precision
        return contexts[:6] if contexts else [""]
