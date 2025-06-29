import sys
import os
import json
import logging
import chromadb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.embeddings.embedding_engine import get_sentence_transformer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cliente con persistencia
client = chromadb.PersistentClient(path="data/processed/chroma_tecsup")

# Leer datos
json_path = os.path.join("data", "raw", "alumnos_vector_base.json")
with open(json_path, "r", encoding="utf-8") as f:
    alumnos = json.load(f)

ids = [doc["id"] for doc in alumnos]
documents = [doc["document"] for doc in alumnos]

# Usar embeddings con modelo SentenceTransformer
embedding_function = get_sentence_transformer()
collection = client.get_or_create_collection(name="Tecsup_Custom", embedding_function=embedding_function)

# Agregar documentos a la nueva colección
collection.add(documents=documents, ids=ids)

logging.info(f"Colección 'Tecsup_Custom' creada con {len(ids)} documentos usando SentenceTransformer.")
