import pandas as pd
import json
from pathlib import Path

def generate_rag_report():
    # 1. Configuración de rutas
    # 'base_path' apunta a 'CodeTFM/' (un nivel arriba de donde está el script)
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent
    
    # Definimos las carpetas donde están los JSONs
    # Puedes añadir aquí todas las que vayas creando
    results_folders = [
        "results_traditionalRAG",
        "results_RAGPlusPlus",
        "results_LightRAG",
        "results_PropertyRAG"
    ]
    
    all_data = []

    print("🔍 Iniciando escaneo de archivos...")

    for folder_name in results_folders:
        folder_path = base_path / folder_name
        
        if not folder_path.exists():
            print(f"⚠️ La carpeta {folder_name} no existe, saltando...")
            continue

        # Buscamos todos los archivos .json en la carpeta
        for json_file in folder_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extraemos solo lo que te interesa
                stats = {
                    "archivo": json_file.name,
                    "rag_type": data.get("rag_type", "unknown"),
                    "dominio": data.get("dominio", "n/a"),
                    "n_libros": data.get("n_libros", 0),
                    "n_preguntas": len(data.get("qa_results", [])),
                    "avg_latency_s": data.get("avg_latency_s", 0),
                    "n_errors": data.get("n_errors", 0),
                    # Scores de RAGAS
                    "faithfulness": data.get("ragas_scores", {}).get("faithfulness"),
                    "answer_relevancy": data.get("ragas_scores", {}).get("answer_relevancy"),
                    "context_precision": data.get("ragas_scores", {}).get("context_precision"),
                    "context_recall": data.get("ragas_scores", {}).get("context_recall")
                }
                all_data.append(stats)
            except Exception as e:
                print(f"❌ Error procesando {json_file.name}: {e}")

    if not all_data:
        print("No se encontraron datos. Revisa las rutas de las carpetas.")
        return

    # 2. Procesamiento con Pandas
    df = pd.DataFrame(all_data)

    # 3. Guardar CSV Detallado
    output_csv = base_path / "resumen_detallado_RAG.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ Reporte detallado guardado en: {output_csv}")

    # 4. Generar Resumen Estadístico por tipo de RAG
    summary = df.groupby("rag_type").agg({
        "n_preguntas": "sum",
        "n_errors": "sum",
        "avg_latency_s": "mean",
        "faithfulness": "mean",
        "answer_relevancy": "mean",
        "context_precision": "mean",
        "context_recall": "mean"
    }).round(4) # Redondeamos a 4 decimales para que sea legible

    # Guardar el resumen estadístico
    summary_csv = base_path / "estadisticas_comparativas_RAG.csv"
    summary.to_csv(summary_csv, encoding='utf-8')
    print(f"✅ Estadísticas comparativas guardadas en: {summary_csv}")
    
    print("\n--- RESUMEN FINAL ---")
    print(summary)

if __name__ == "__main__":
    generate_rag_report()