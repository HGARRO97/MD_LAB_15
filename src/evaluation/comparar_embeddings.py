import logging
import chromadb
from chromadb.utils import embedding_functions
from src.embeddings.embedding_engine import get_sentence_transformer

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Prompt de consulta
prompt = "¿Quién estudia ciencia de datos o le interesa la inteligencia artificial?"

# Conectar cliente persistente
client = chromadb.PersistentClient(path="data/processed/chroma_tecsup")

# --- Consulta con DefaultEmbeddingFunction ---
default_func = embedding_functions.DefaultEmbeddingFunction()
collection_default = client.get_or_create_collection(name="Tecsup", embedding_function=default_func)

res_default = collection_default.query(query_texts=[prompt], n_results=3)
logging.info("Resultados con DefaultEmbeddingFunction:")
for i, doc in enumerate(res_default["documents"][0]):
    dist = res_default["distances"][0][i]
    logging.info(f"[Default] Documento {i+1} (Distancia: {dist:.4f})")
    logging.info(f"   {doc}")

# --- Consulta con SentenceTransformerEmbeddingFunction ---
custom_func = get_sentence_transformer()
collection_custom = client.get_or_create_collection(name="Tecsup_Custom", embedding_function=custom_func)

res_custom = collection_custom.query(query_texts=[prompt], n_results=3)
logging.info("Resultados con SentenceTransformerEmbeddingFunction:")
for i, doc in enumerate(res_custom["documents"][0]):
    dist = res_custom["distances"][0][i]
    logging.info(f"[Custom] Documento {i+1} (Distancia: {dist:.4f})")
    logging.info(f"   {doc}")

