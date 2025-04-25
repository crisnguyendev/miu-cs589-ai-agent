import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import os

from data.file_utils import find_file


def update_embeddings_and_index(model_path="../output/fine_tuned_model",
                                output_index_path="../output/verse_index.faiss"):
    # Load the fine-tuned model
    model = SentenceTransformer(model_path)

    # Load verse metadata using find_file
    start_dir = os.path.dirname(os.path.abspath(__file__))
    verses_file_path = find_file(start_dir, "verses_metadata.csv")
    print(f"Found verses_metadata.csv at: {verses_file_path}")
    df = pd.read_csv(verses_file_path)
    verses = df["text"].tolist()

    # Generate new embeddings
    embeddings = model.encode(verses, show_progress_bar=True)

    # Create and save new FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, output_index_path)
    print(f"Updated FAISS index saved to {output_index_path}")