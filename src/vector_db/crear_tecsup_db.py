import json
import logging
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def cargar_tecsup_default():
    base_dir = Path(__file__).resolve().parent.parent.parent
    json_path = base_dir / "data" / "raw" / "alumnos_vector_base.json"
    db_path = base_dir / "data" / "processed" / "chroma_tecsup"

    if not json_path.exists():
        logging.error(f"Archivo JSON no encontrado: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        alumnos = json.load(f)

    if not isinstance(alumnos, list) or len(alumnos) == 0:
        logging.error("El archivo JSON está vacío o malformado.")
        return

    ids = [str(doc["id"]) for doc in alumnos]
    textos = [doc["document"] for doc in alumnos]

    client = chromadb.PersistentClient(path=str(db_path))
    embedding_func = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(name="Tecsup", embedding_function=embedding_func)

    if collection.count() == 0:
        collection.add(documents=textos, ids=ids)
        logging.info(f"Documentos insertados en la colección 'Tecsup': {len(textos)}")
    else:
        logging.warning(f"La colección 'Tecsup' ya contiene {collection.count()} documentos. No se insertaron nuevos datos.")

if __name__ == "__main__":
    cargar_tecsup_default()
