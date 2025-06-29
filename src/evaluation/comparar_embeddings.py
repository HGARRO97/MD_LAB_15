import logging
import chromadb
from chromadb.utils import embedding_functions
import sys
import os

# Agregar src al path manualmente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.embeddings.embedding_engine import get_sentence_transformer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Prompt a comparar
prompt = "¿Quién estudia ciencia de datos o le interesa la inteligencia artificial?"
logging.info(f"Comparación de respuestas para: {prompt}")

# Cliente persistente
client = chromadb.PersistentClient(path="data/processed/chroma_tecsup")

# Función para consultar
def consultar(nombre_coleccion, embedding_func):
    col = client.get_or_create_collection(name=nombre_coleccion, embedding_function=embedding_func)
    return col.query(query_texts=[prompt], n_results=3)

# Ejecutar consultas
default_func = embedding_functions.DefaultEmbeddingFunction()
res_default = consultar("Tecsup", default_func)

custom_func = get_sentence_transformer()
res_custom = consultar("Tecsup_Custom", custom_func)

# Mostrar comparaciones
for i in range(3):
    logging.info(f"\n📊 Comparación #{i+1}")
    logging.info(f"[Default] Distancia: {res_default['distances'][0][i]:.4f}")
    logging.info(res_default['documents'][0][i])
    logging.info(f"[Custom]  Distancia: {res_custom['distances'][0][i]:.4f}")
    logging.info(res_custom['documents'][0][i])
