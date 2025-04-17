import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import logging
import argparse
import os
import warnings
from typing import List, Tuple

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_csv_file(base_path: str, filename: str = "verses.csv") -> str:
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
        csv_path = os.path.join(dir_path, filename)
        if Path(csv_path).exists():
            logger.info(f"Found CSV file at: {csv_path}")
            return csv_path
    raise FileNotFoundError(f"CSV file '{filename}' not found in common directories: {search_dirs}")

def load_verses(csv_path: str) -> pd.DataFrame:
    try:
        if not Path(csv_path).exists():
            logger.warning(f"CSV path {csv_path} not found, searching common directories...")
            csv_path = find_csv_file(os.path.dirname(csv_path))
        df = pd.read_csv(csv_path, encoding='utf-8')
        required_columns = ['id', 'book', 'chapter', 'verse', 'text_en']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"CSV missing required columns: {required_columns}")
            raise ValueError(f"CSV missing required columns")
        df = df[df['text_en'].str.len() > 5].copy()  # Relaxed filter
        logger.info(f"Loaded {len(df)} verses from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {csv_path}: {e}")
        raise

def generate_embeddings(verses: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    try:
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"Generating embeddings for {len(verses)} verses")
        embeddings = model.encode(verses, show_progress_bar=True, batch_size=32)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        logger.info(f"Built FAISS index with {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise

def save_index_and_metadata(index: faiss.Index, df: pd.DataFrame, index_path: str, metadata_path: str):
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        faiss.write_index(index, index_path)
        df.to_csv(metadata_path, index=False, encoding='utf-8')
        logger.info(f"Saved FAISS index to {index_path}")
        logger.info(f"Saved metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving index/metadata: {e}")
        raise

def test_retrieval(index: faiss.Index, df: pd.DataFrame, model: SentenceTransformer, queries: List[str], k: int = 5):
    logger.info("Testing retrieval with sample queries")
    for query in queries:
        try:
            query_emb = model.encode([query]).astype(np.float32)
            distances, indices = index.search(query_emb, k)
            logger.info(f"\nQuery: {query}")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                verse = df.iloc[idx]
                score = 1 / (1 + dist)
                logger.info(f"Result {i+1} (score: {score:.3f}):")
                logger.info(f"  {verse['book']} {verse['chapter']}.{verse['verse']}: {verse['text_en'][:100]}...")
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for Vedic verses and build FAISS index.")
    parser.add_argument("--csv_path", default="../output/verses.csv", help="Path to verses CSV")
    parser.add_argument("--index_path", default="../output/verse_embeddings.faiss", help="Path to save FAISS index")
    parser.add_argument("--metadata_path", default="../output/verse_metadata.csv", help="Path to save metadata CSV")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2", help="Sentence-BERT model name")
    args = parser.parse_args()

    logger.info(f"Current working directory: {os.getcwd()}")
    df = load_verses(args.csv_path)
    verses = df['text_en'].tolist()
    embeddings = generate_embeddings(verses, args.model_name)
    index = build_faiss_index(embeddings)
    save_index_and_metadata(index, df, args.index_path, args.metadata_path)
    model = SentenceTransformer(args.model_name)
    test_queries = [
        "What does the Upanishads say about the self?",
        "Vedic philosophy",
        "Karma in the Upanishads",
        "Peace and spirituality",
        "Brahman and Atman"
    ]
    test_retrieval(index, df, model, test_queries, k=5)

if __name__ == "__main__":
    main()