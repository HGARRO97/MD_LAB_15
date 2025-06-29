import logging
import chromadb
from chromadb.utils import embedding_functions

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Nuevo cliente persistente
client = chromadb.PersistentClient(path="data/processed/chroma_tecsup")

embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(name="Tecsup", embedding_function=embedding_function)

consulta = "¿Quién estudia ciencia de datos o le interesa la inteligencia artificial?"
logging.info(f"Consulta realizada: {consulta}")

resultados = collection.query(query_texts=[consulta], n_results=3)

logging.info("Resultados más similares:")
for i, doc in enumerate(resultados["documents"][0]):
    distancia = resultados["distances"][0][i]
    logging.info(f"Documento {i+1} (Distancia: {distancia:.4f})")
    for linea in doc.split('. '):
        logging.info(f"   {linea.strip()}")


