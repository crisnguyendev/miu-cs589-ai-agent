import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

print("Started")
# Load dataset
csv_path = "../output/verses.csv"
if not Path(csv_path).exists():
    print(f"Error: {csv_path} not found")
    exit(1)

df = pd.read_csv(csv_path)
required_columns = ["verse_id", "text", "source"]
if not all(col in df.columns for col in required_columns):
    print(f"Error: CSV missing required columns: {required_columns}")
    exit(1)

verses = df["text"].tolist()

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(verses, show_progress_bar=True)

print("Started Encoding")
# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))

# Save index and metadata
faiss.write_index(index, "../output/verse_index.faiss")
df.to_csv("../output/verses_metadata.csv", index=False)

# Test nearest-neighbor search
sample_queries = [
    "What is Dharma?",
    "Vedic philosophy",
    "Peace in the Upanishads"
]

for query in sample_queries:
    print(f"\nQuery: {query}")
    query_emb = model.encode([query])
    D, I = index.search(query_emb.astype(np.float32), k=3)  # Top 3 results
    for i, idx in enumerate(I[0]):
        verse = df.iloc[idx]
        print(f"Result {i+1}:")
        print(f"Source: {verse['source']}")
        print(f"Text: {verse['text']}")
        print(f"Similarity Score: {1 / (1 + D[0][i]):.3f}")