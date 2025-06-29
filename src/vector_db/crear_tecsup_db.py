import os
import json
import logging
import chromadb
from chromadb.utils import embedding_functions

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Nuevo cliente persistente (actualizado 2025)
client = chromadb.PersistentClient(path="data/processed/chroma_tecsup")

# Leer datos
json_path = os.path.join("data", "raw", "alumnos_vector_base.json")
with open(json_path, "r", encoding="utf-8") as f:
    alumnos = json.load(f)

ids = [doc["id"] for doc in alumnos]
documents = [doc["document"] for doc in alumnos]

embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(name="Tecsup", embedding_function=embedding_function)

collection.add(documents=documents, ids=ids)

logging.info(f"Base de datos vectorial 'Tecsup' creada y persistida con {len(ids)} documentos.")
