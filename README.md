# VedicSage: Semantic Search for Upanishads

*VedicSage* is an AI-powered semantic search application designed to process and query verses from the Upanishads, ancient Vedic texts. By parsing raw text files, generating semantic embeddings, and serving a FastAPI-based search API, the project enables users to explore philosophical and spiritual teachings through natural language queries (e.g., "What does the Upanishads say about the self?"). Developed as part of the MIU CS589 AI Agent course, the project aims to parse \~1,000 verses (currently \~30, targeting \~276 from current texts).

## Project Overview

The project processes Upanishad texts (`upanishads-isa.txt`, `upanishads-kena.txt`, `upanishads-katha.txt`, `Upanishads-Translated-by-Swami-Paramananda-PLAINTEXT.txt`) to extract structured verses, converts them into embeddings using a SentenceTransformer model (`all-MiniLM-L6-v2`), and serves a FastAPI endpoint for semantic search. The pipeline is orchestrated by a Bash script (`run_vedicsage.sh`).

### Features

- **Text Parsing**: Extracts verses from Upanishad texts, handling complex formatting (e.g., Roman numerals, commentary).
- **Semantic Embeddings**: Generates vector representations for verses using NLP techniques.
- **FastAPI Server**: Provides a queryable API for retrieving relevant verses based on semantic similarity.
- **Scalable Design**: Supports adding more Upanishad texts to reach \~1,000 verses.

### Current Status

- Parses \~30 verses from four texts (expected \~276: Isa \~19, Kena \~36, Katha \~120, Paramananda \~100 estimated).
- Goal: Scale to \~1,000 verses by adding more texts (e.g., Svetasvatara, Prasna).

## Project Structure

```plaintext
miu-cs589-ai-agent-api/
├── run_vedicsage.sh               # Bash script to run the pipeline
├── data/
│   ├── vedic_parse_verses.py      # Parses Upanishad texts into verses.csv
│   ├── vedic_embed_verses.py      # Generates embeddings and FAISS index
├── service/
│   ├── vedic_retrieval_api.py     # FastAPI server for semantic search
├── resources/
│   ├── upanishads-isa.txt         # Isa Upanishad (~19 verses)
│   ├── upanishads-kena.txt        # Kena Upanishad (~36 verses)
│   ├── upanishads-katha.txt       # Katha Upanishad (~120 verses)
│   ├── Upanishads-Translated-by-Swami-Paramananda-PLAINTEXT.txt  # Collection (~100 verses estimated)
├── output/
│   ├── verses.csv                 # Parsed verses
│   ├── verse_embeddings.faiss      # FAISS index
│   ├── verse_metadata.csv         # Metadata for verses
├── .gitignore                     # Excludes .venv/, output/, cache files
├── README.md                      # This file
```

## Prerequisites

- **Python 3.8+**
- **Bash** (for running `run_vedicsage.sh`)
- **Dependencies** (installed via `pip`):
  - `pandas`
  - `numpy`
  - `sentence-transformers`
  - `faiss-cpu`
  - `fastapi`
  - `uvicorn`

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/crisnguyendev/miu-cs589-ai-agent-api.git
   cd miu-cs589-ai-agent-api
   ```

2. **Set Up Virtual Environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install pandas numpy sentence-transformers faiss-cpu fastapi uvicorn
   ```

4. **Verify Text Files**: Ensure `resources/` contains:

   - `upanishads-isa.txt`
   - `upanishads-kena.txt`
   - `upanishads-katha.txt`
   - `Upanishads-Translated-by-Swami-Paramananda-PLAINTEXT.txt`

5. **Make Script Executable**:

   ```bash
   chmod +x run_vedicsage.sh
   ```

## Running the Project

Run the pipeline to parse texts, generate embeddings, and start the FastAPI server:

```bash
./run_vedicsage.sh
```

The script will:

- Parse Upanishad texts into `output/verses.csv`.
- Generate embeddings and save to `output/verse_embeddings.faiss` and `output/verse_metadata.csv`.
- Start the FastAPI server at `http://localhost:8000`.

## Using the API

Query the API using a web browser or `curl`:

```bash
curl "http://localhost:8000/retrieve?query=What%20does%20the%20Upanishads%20say%20about%20the%20self?&k=3"
```

Example response:

```json
{
  "query": "What does the Upanishads say about the self?",
  "results": [
    {
      "verse": "The self is the source of all actions and duties in the world",
      "source": "Upanishads 1.2",
      "score": 0.85
    },
    {
      "verse": "That thou art the self is one with Brahman",
      "source": "Upanishads 2.1",
      "score": 0.82
    },
    {
      "verse": "Know the self to be eternal and unchanging",
      "source": "Upanishads 3.4",
      "score": 0.81
    }
  ]
}
```

Access the API documentation at `http://localhost:8000/docs`.

## Adding More Upanishads

To reach the \~1,000-verse goal (current: \~30, target: \~276):

1. Download additional Upanishad texts from Sacred-Texts.com:

   - Svetasvatara (\~100 verses)
   - Prasna (\~60 verses)
   - Taittiriya (\~50 verses)
   - Mundaka (\~60 verses)
   - Mandukya (\~12 verses)

2. Example download:

   ```bash
   curl https://www.sacred-texts.com/hin/svet/svet.txt -o resources/upanishads-svetasvatara.txt
   ```

3. Place files in `resources/` and rerun `run_vedicsage.sh`.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please focus on:

- Improving verse parsing to reach \~276+ verses.
- Adding more Upanishad texts.
- Enhancing API functionality or documentation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- MIU CS589-AI course for project inspiration.
- Sacred-Texts.com for Upanishad texts.
- SentenceTransformers and FAISS for NLP and search capabilities.

---

*Note*: The project is under development to increase the verse count from \~30 to \~276 and ultimately \~1,000. Contributions to improve parsing are highly encouraged!
