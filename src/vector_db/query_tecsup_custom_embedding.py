import logging
import chromadb
from chromadb.utils import embedding_functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.embeddings.embedding_engine import get_sentence_transformer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Usar cliente persistente con embedding personalizado
client = chromadb.PersistentClient(path="data/processed/chroma_tecsup")

# Reemplazar función de embedding
embedding_function = get_sentence_transformer()
collection = client.get_or_create_collection(name="Tecsup_Custom", embedding_function=embedding_function)

# Prompt
consulta = "¿Quién estudia ciencia de datos o le interesa la inteligencia artificial?"
logging.info(f"Consulta realizada (modelo personalizado): {consulta}")

# Consulta
resultados = collection.query(query_texts=[consulta], n_results=3)

# Resultados
logging.info("Resultados más similares (modelo personalizado):")
for i, doc in enumerate(resultados["documents"][0]):
    dist = resultados["distances"][0][i]
    logging.info(f"Documento {i+1} (Distancia: {dist:.4f})")
    for linea in doc.split('. '):
        logging.info(f"   {linea.strip()}")
