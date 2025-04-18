from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import warnings
from typing import List, Dict
from pydantic import BaseModel
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="VedicSage Retrieval API",
              description="Retrieve relevant Vedic verses based on semantic similarity")


class RetrievalResult(BaseModel):
    verse: str
    source: str
    score: float


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]


def find_file(base_path: str, filename: str) -> str:
    search_dirs = [
        base_path,
        os.path.join(base_path, "output"),
        os.path.join(base_path, "../output"),
        os.path.join(base_path, "data"),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output"),
    ]
    logger.debug(f"Searching for {filename} in: {search_dirs}")
    for dir_path in search_dirs:
        file_path = os.path.join(dir_path, filename)
        if Path(file_path).exists():
            logger.info(f"Found {filename} at: {file_path}")
            return file_path
    raise FileNotFoundError(f"File '{filename}' not found in common directories: {search_dirs}")


class VedicRetriever:
    def __init__(self, index_path: str, metadata_path: str, model_name: str = "all-MiniLM-L6-v2"):
        try:
            if not Path(index_path).exists():
                logger.warning(f"FAISS index not found at {index_path}, searching common directories...")
                index_path = find_file(os.path.dirname(index_path), "verse_index.faiss")
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")

            if not Path(metadata_path).exists():
                logger.warning(f"Metadata CSV not found at {metadata_path}, searching common directories...")
                metadata_path = find_file(os.path.dirname(metadata_path), "verses_metadata.csv")
            self.df = pd.read_csv(metadata_path, encoding='utf-8')
            # old column format
            # required_columns = ['id', 'book', 'chapter', 'verse', 'text_en']
            required_columns = ['verse_id', 'text', 'source']
            if not all(col in self.df.columns for col in required_columns):
                logger.error(f"Metadata missing required columns: {required_columns}")
                raise ValueError(f"Metadata missing required columns")
            logger.info(f"Loaded {len(self.df)} verses from {metadata_path}")

            logger.info(f"Loading Sentence-BERT model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        try:
            query_emb = self.model.encode([query]).astype(np.float32)
            distances, indices = self.index.search(query_emb, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.df):
                    verse = self.df.iloc[idx]
                    score = 1 / (1 + dist)
                    source = f"{verse['book']} {verse['chapter']}.{verse['verse']}"
                    results.append({
                        "verse": verse['text_en'],
                        "source": source,
                        "score": float(score)
                    })
                else:
                    logger.warning(f"Invalid index {idx} returned by FAISS")
            return results
        except Exception as e:
            logger.error(f"Error retrieving for query '{query}': {e}")
            raise


retriever = VedicRetriever(
    index_path="../output/verse_index.faiss",
    metadata_path="../output/verses_metadata.csv",
    model_name="all-MiniLM-L6-v2"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VedicSage Retrieval API starting")
    logger.info(f"Working directory: {os.getcwd()}")
    yield
    logger.info("VedicSage Retrieval API shutting down")


app.lifespan = lifespan


@app.get("/retrieve", response_model=RetrievalResponse)
async def retrieve_verses(query: str, k: int = 5):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if k < 1 or k > 10:
        raise HTTPException(status_code=400, detail="k must be between 1 and 10")

    try:
        results = retriever.retrieve(query, k)
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"API error for query '{query}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# For testing API
@app.get("/")
async def root():
    return {"message": "Vedic Retrieval API is running"}

if __name__ == "__main__":
    import uvicorn
    import os

    logger.info(f"Working directory: {os.getcwd()}")



    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)