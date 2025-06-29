from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def get_sentence_transformer():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
