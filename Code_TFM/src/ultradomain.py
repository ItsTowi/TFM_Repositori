import json
import os
import random

DOMINIOS_VALIDOS = [
    "agriculture", "art", "biography", "biology", "cooking", "cs",
    "fiction", "health", "history", "literature", "mathematics",
    "music", "philosophy", "physics", "politics", "psychology", "technology"
]

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ultradomain")


def cargar_experimento(dominio, n_libros=1, shuffle=False):
    """
    Carga n_libros del dominio con sus Q&A asociadas.

    Args:
        dominio:  uno de DOMINIOS_VALIDOS
        n_libros: cuántos libros cargar (None = todos)
        shuffle:  si True, selecciona libros aleatoriamente

    Returns:
        libros: lista de {"context_id", "titulo", "autores", "texto"}
        qas:    lista de {"question", "answer", "context_id", "titulo"}
    """
    if dominio not in DOMINIOS_VALIDOS:
        raise ValueError(f"Dominio '{dominio}' no válido. Elige uno de: {DOMINIOS_VALIDOS}")

    libros_dir = os.path.join(PROCESSED_DIR, dominio)
    qa_path    = os.path.join(PROCESSED_DIR, "qa", f"{dominio}.json")

    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"No se encontró {qa_path}. ¿Has ejecutado exportar_dominio()?")

    # Cargar Q&A completas
    with open(qa_path, "r", encoding="utf-8") as f:
        todas_qas = json.load(f)

    # Obtener context_ids únicos
    context_ids = list({qa["context_id"] for qa in todas_qas})

    if shuffle:
        random.shuffle(context_ids)

    if n_libros is not None:
        context_ids = context_ids[:n_libros]

    # Cargar los .txt de esos libros
    libros = []
    for cid in context_ids:
        txt_path = os.path.join(libros_dir, f"{cid}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"No se encontró {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            texto = f.read()

        lineas  = texto.split("\n")
        titulo  = lineas[0].replace("Title: ", "")
        autores = lineas[1].replace("Authors: ", "")

        libros.append({
            "context_id": cid,
            "titulo":     titulo,
            "autores":    autores,
            "texto":      texto
        })

    # Filtrar Q&A de los libros seleccionados
    ids_seleccionados = {l["context_id"] for l in libros}
    qas = [qa for qa in todas_qas if qa["context_id"] in ids_seleccionados]

    # Resumen
    print(f"📚 Dominio: {dominio}")
    for libro in libros:
        n_preguntas = sum(1 for q in qas if q["context_id"] == libro["context_id"])
        print(f"   📖 {libro['titulo']} — {libro['autores']} ({n_preguntas} preguntas)")
    print(f"   Total Q&A: {len(qas)}")

    return libros, qas