# vectorstore_utils.py
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from app.config import OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

def create_qdrant_index(texts: List[str]):
    # OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",   # or "text-embedding-3-large"
        api_key=OPENAI_API_KEY
    )

    # Connect to Qdrant Cloud
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    # Store embeddings inside Qdrant Cloud
    vectorstore = Qdrant.from_texts(
        texts,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    return vectorstore

def retrieve_relevant_docs(vectorstore, query: str, k: int = 3):
    return vectorstore.similarity_search(query, k=k)
 