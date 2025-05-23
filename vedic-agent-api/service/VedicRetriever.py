import logging
import os
import warnings
from typing import List, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

from data.file_utils import find_file

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VedicRetriever:
    def __init__(self, index_path: str, metadata_path: str, model_path: str = "../output/fine_tuned_model"):
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
            required_columns = ['verse_id', 'text', 'source']
            if not all(col in self.df.columns for col in required_columns):
                logger.error(f"Metadata missing required columns: {required_columns}")
                raise ValueError(f"Metadata missing required columns")
            logger.info(f"Loaded {len(self.df)} verses from {metadata_path}")

            # LOAD MODEL for embedding
            self.model_path = model_path  # Store model_path for logging in unload_model
            if Path(model_path).exists():
                logger.info(f"Loading finetuned Model from: {model_path}")
                self.model = SentenceTransformer(model_path)  # Load fine-tuned model
            else:
                model_name: str = "all-MiniLM-L6-v2"
                logger.info(f"Loading Sentence-BERT model from: {model_name}")
                self.model = SentenceTransformer(model_name)

        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise

    def unload_model(self):
        """Unload the SentenceTransformer model to release any file locks."""
        if self.model is not None:
            logger.info(f"Unloading SentenceTransformer model from {self.model_path}")
            self.model = None  # Set to None to allow garbage collection
            import gc
            gc.collect()  # Force garbage collection to release resources

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        try:
            if self.model is None:
                raise RuntimeError("Model is not loaded. Please reinitialize VedicRetriever.")

            query_emb = self.model.encode([query]).astype(np.float32)
            distances, indices = self.index.search(query_emb, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.df):
                    verse = self.df.iloc[idx]
                    score = 1 / (1 + dist)

                    results.append({
                        "verse": verse["text"],
                        "source": verse["source"],
                        "score": float(score),
                        "verse_id": verse["verse_id"]
                    })
                else:
                    logger.warning(f"Invalid index {idx} returned by FAISS")
            return results
        except Exception as e:
            logger.error(f"Error retrieving for query '{query}': {e}")
            raise