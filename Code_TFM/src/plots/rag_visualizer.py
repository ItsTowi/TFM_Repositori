"""
rag_visualizer.py
=================
Genera gráficas comparativas entre distintos RAGs a partir de los JSON
de resultados generados por el evaluador.

Uso en Jupyter:
    from rag_visualizer import plot_rag_comparison
    plot_rag_comparison(results_dir="./results", output_dir="./plots")
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paleta y estilo ────────────────────────────────────────────────────────────

RAG_COLORS = {
    "traditional":      "#4C9BE8",
    "advanced":         "#F4845F",
    "lightrag":         "#56C596",
    "msgraphrag_local": "#A67FF0",
    "msgraphrag_global":"#F0C040",
    "llamaindex":       "#E06C75",
}
DEFAULT_COLOR = "#AAAAAA"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}


def _color(rag_type: str) -> str:
    return RAG_COLORS.get(rag_type, DEFAULT_COLOR)


def _label(rag_type: str) -> str:
    return rag_type.replace("_", "\n")


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> list[dict]:
    """
    Carga todos los JSON de resultados de una carpeta.
    Si hay varios archivos del mismo rag_type, se queda con el más reciente.
    """
    folder = Path(results_dir)
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)

    seen = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            rag_type = data.get("rag_type", f.stem)
            seen[rag_type] = data          # el último (más reciente) gana
        except Exception as e:
            print(f"⚠️ No se pudo cargar {f.name}: {e}")

    results = list(seen.values())
    print(f"✅ {len(results)} RAGs cargados: {[r['rag_type'] for r in results]}")
    return results


def _safe_score(data: dict, metric: str) -> float | None:
    val = data.get("ragas_scores", {}).get(metric)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


# ── Gráfica 1: Barras agrupadas por métrica ────────────────────────────────────

def plot_grouped_bars(results: list[dict], ax: plt.Axes):
    n_metrics = len(METRICS)
    n_rags    = len(results)
    width     = 0.7 / n_rags
    x         = np.arange(n_metrics)

    for i, r in enumerate(results):
        scores = [_safe_score(r, m) for m in METRICS]
        offsets = x + (i - n_rags / 2 + 0.5) * width
        bars = ax.bar(
            offsets,
            [s if s is not None else 0 for s in scores],
            width=width * 0.9,
            color=_color(r["rag_type"]),
            label=r["rag_type"],
            zorder=3,
        )
        for bar, score in zip(bars, scores):
            if score is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{score:.2f}",
                    ha="center", va="bottom",
                    fontsize=7, color="#333333"
                )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RAGAS Metrics — Grouped by Metric", fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)


# ── Gráfica 2: Radar chart ─────────────────────────────────────────────────────

def plot_radar(results: list[dict], ax: plt.Axes):
    categories = [METRIC_LABELS[m] for m in METRICS]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7, color="grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="-", alpha=0.2)

    for r in results:
        values = [_safe_score(r, m) or 0 for m in METRICS]
        values += values[:1]
        color = _color(r["rag_type"])
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=r["rag_type"])
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_title("RAGAS Metrics — Radar", fontsize=13, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8, framealpha=0.85)


# ── Gráfica 3: Latencia media ──────────────────────────────────────────────────

def plot_latency(results: list[dict], ax: plt.Axes):
    labels   = [r["rag_type"] for r in results]
    latencies = [r.get("avg_latency_s", 0) for r in results]
    colors   = [_color(r["rag_type"]) for r in results]

    bars = ax.barh(labels, latencies, color=colors, zorder=3, height=0.5)
    for bar, val in zip(bars, latencies):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}s",
            va="center", fontsize=9, color="#333333"
        )

    ax.set_xlabel("Avg Latency (s)", fontsize=11)
    ax.set_title("Average Query Latency", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(latencies) * 1.25 + 1)


# ── Gráfica 4: Heatmap de métricas ────────────────────────────────────────────

def plot_heatmap(results: list[dict], ax: plt.Axes):
    rag_labels    = [r["rag_type"] for r in results]
    metric_labels = [METRIC_LABELS[m] for m in METRICS]

    matrix = np.array([
        [_safe_score(r, m) if _safe_score(r, m) is not None else np.nan
         for m in METRICS]
        for r in results
    ])

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(len(rag_labels)))
    ax.set_yticklabels(rag_labels, fontsize=9)
    ax.set_title("RAGAS Scores Heatmap", fontsize=13, fontweight="bold", pad=12)

    for i in range(len(rag_labels)):
        for j in range(len(metric_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="black" if 0.35 < val < 0.85 else "white",
                        fontweight="bold")


# ── Gráfica 5: Overall score (media de métricas) ──────────────────────────────

def plot_overall(results: list[dict], ax: plt.Axes):
    labels  = []
    overall = []
    colors  = []

    for r in results:
        scores = [_safe_score(r, m) for m in METRICS if _safe_score(r, m) is not None]
        if scores:
            labels.append(r["rag_type"])
            overall.append(np.mean(scores))
            colors.append(_color(r["rag_type"]))

    sorted_pairs = sorted(zip(overall, labels, colors), reverse=True)
    overall, labels, colors = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

    bars = ax.bar(labels, overall, color=colors, zorder=3, width=0.5)
    for bar, val in zip(bars, overall):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333"
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean RAGAS Score", fontsize=11)
    ax.set_title("Overall Performance (Mean of all metrics)", fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ── Función principal ──────────────────────────────────────────────────────────

def plot_rag_comparison(
    results_dir: str = "./results",
    output_dir: str  = "./plots",
    experiment_name: str = "",
    figsize: tuple = (18, 14),
    dpi: int = 150,
) -> Path:
    """
    Genera y guarda una figura con 5 subplots comparando los RAGs.

    Args:
        results_dir:      carpeta con los JSON de resultados
        output_dir:       carpeta donde guardar el PNG
        experiment_name:  nombre del experimento (para el título y filename)
        figsize:          tamaño de la figura
        dpi:              resolución del PNG

    Returns:
        Path del PNG generado
    """
    results = load_results(results_dir)
    if not results:
        print("❌ No se encontraron resultados")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#F8F9FA")

    # Layout: 2 filas × 3 columnas, el radar ocupa 1 celda, overall 1 celda
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                          left=0.07, right=0.96, top=0.90, bottom=0.07)

    ax_bars    = fig.add_subplot(gs[0, :2])          # fila 0, col 0-1
    ax_radar   = fig.add_subplot(gs[0, 2], polar=True) # fila 0, col 2
    ax_heatmap = fig.add_subplot(gs[1, :2])          # fila 1, col 0-1
    ax_latency = fig.add_subplot(gs[1, 2])            # fila 1, col 2

    plot_grouped_bars(results, ax_bars)
    plot_radar(results, ax_radar)
    plot_heatmap(results, ax_heatmap)
    plot_latency(results, ax_latency)

    # Título general
    title = f"RAG Comparison — {experiment_name}" if experiment_name else "RAG Comparison"
    n_q = max((len(r.get("qa_results", [])) for r in results), default=0)
    subtitle = f"{len(results)} systems · {n_q} question(s) · domain: {results[0].get('dominio', '')}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.96)
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=10, color="#666666")

    # Guardar
    safe_name = (experiment_name or "comparison").replace(" ", "_").lower()
    out_path = Path(output_dir) / f"rag_comparison_{safe_name}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"💾 Guardado en: {out_path}")
    return out_path


# ── Uso directo desde el notebook ─────────────────────────────────────────────
# from rag_visualizer import plot_rag_comparison
#
# plot_rag_comparison(
#     results_dir  = "./results",
#     output_dir   = "./plots",
#     experiment_name = "Test_Local_01",
# )
