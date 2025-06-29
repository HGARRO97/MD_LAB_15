
# Resumen del Laboratorio GLAB-S15 – Minería de Textos

## Objetivo
Construir una base de datos vectorial llamada `Tecsup`, almacenar información de 20 alumnos, realizar consultas semánticas y comparar dos motores de embeddings distintos.

---

## Pasos Realizados

### 1. **Generación de datos simulados**
Se creó el archivo `alumnos_vector_base.json` con 20 alumnos, incluyendo campos como nombre, carrera e intereses.

### 2. **Creación de base vectorial con ChromaDB**
- Colección `Tecsup`: usando `DefaultEmbeddingFunction()`
- Colección `Tecsup_Custom`: usando `SentenceTransformerEmbeddingFunction()` con modelo `all-MiniLM-L6-v2`

### 3. **Consultas semánticas**
Se consultó la base con:
> "¿Quién estudia ciencia de datos o le interesa la inteligencia artificial?"

### 4. **Comparación de resultados**
Se compararon los resultados de ambos motores mostrando los 3 documentos más similares a la consulta.

---

## Motores usados

| Motor                   | Embedding usado                        | Resultado esperado        |
|------------------------|----------------------------------------|---------------------------|
| `DefaultEmbedding`     | Interno de `chromadb`                  | Aproximación simple       |
| `SentenceTransformer`  | `all-MiniLM-L6-v2`                     | Mayor precisión semántica |

---

## Observaciones

- `SentenceTransformer` generó mejores agrupamientos por significado, aunque con más tiempo de cómputo.
- El motor por defecto es suficiente para prototipos rápidos.
- La base vectorial permite realizar búsquedas tipo ChatGPT en documentos propios.

---

## Archivos clave

| Archivo                                      | Descripción                                   |
|---------------------------------------------|-----------------------------------------------|
| `data/raw/alumnos_vector_base.json`         | Datos simulados de 20 alumnos                 |
| `src/vector_db/crear_tecsup_db.py`          | Crea base con embeddings por defecto          |
| `src/vector_db/query_tecsup_db.py`          | Consulta base con prompt                      |
| `src/vector_db/crear_tecsup_custom_db.py`   | Crea base con SentenceTransformer             |
| `src/vector_db/query_tecsup_custom_embedding.py` | Consulta base con embedding contextual  |
| `src/evaluation/comparar_embeddings.py`     | Compara ambos motores                         |

---
