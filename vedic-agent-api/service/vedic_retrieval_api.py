from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import warnings
from openai import OpenAI
from dotenv import load_dotenv

from typing import List, Dict
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from service.VedicRetriever import VedicRetriever
from service.models import FeedbackRequest, RetrievalResponse

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Loading Environment variables
load_dotenv()

app = FastAPI(title="VedicSage Retrieval API",
              description="Retrieve relevant Vedic verses based on semantic similarity")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://miu-vedic-science.vercel.app"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


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



retriever = VedicRetriever(
    index_path="../output/verse_index.faiss",
    metadata_path="../output/verses_metadata.csv",
    model_path="../output/fine_tuned_model"
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

# Create OpenAI client once (outside of the endpoint)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/answer")
async def answer_question(query: str, k: int = 5):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Step 1: Retrieve top-k relevant verses
        verses = retriever.retrieve(query, k)
        context = "\n".join([f"- {v['verse']}" for v in verses])

        # Step 2: Compose prompt and call OpenAI
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant knowledgeable in Vedic texts. "
        #                                   "When the user prompts you, a semantic search is performed on our database, and suitable passages from Vedic texts are retreived. Include some of the content of these passages in your answer and state their source."},
        #     {"role": "user", "content": f"Please answer the question:\n\n{context}\n\nQuestion: {query}"}
        # ]

        messages = [
            {"role": "system", "content": "You are a helpful assistant knowledgeable in Vedic Science. Eventhough the Vedas talk about God, it is not a religion. Please do not name any religion, like Hinduism."},
            {"role": "user",
             "content": f"Question: {query}"
                        f"Please answer the question in the following way:"
                        f"If applicable, First give a short, maximum 2 sentences, general answer."
                        f"Then, without quoting them one to one, expand your answer in consideration of the found verses (=context): "
                        f"\n\n{context}\n\n. "
                        f"Mention the sources of the verses you consider in your answer (only the name "
                        # comment out line below, if other datasources outside of internal database should be considered and referenced:
                        f"and only if they were part of the provided context "
                        f"). "
                        f"Please do not go through the sources one by one, summarize amd synthesize. "
             }
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )

        answer = response.choices[0].message.content

        return {
            "query": query,
            "context": verses,
            "answer": answer.strip()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()  # This will print the full error stack trace to your terminal
        logger.error(f"Error generating answer for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))  # Show the actual error for debugging
        # logger.error(f"Error generating answer for query '{query}': {e}")
        # raise HTTPException(status_code=500, detail="Failed to generate answer")


### FEEDBACK
from pydantic import BaseModel


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        # Save feedback to a CSV file for later use
        feedback_data = {
            "query": feedback.query,
            "verse_id": feedback.verse_id,
            "is_relevant": feedback.is_relevant,
            "timestamp": pd.Timestamp.now()
        }
        feedback_df = pd.DataFrame([feedback_data])
        feedback_file = "../output/feedback.csv"
        if Path(feedback_file).exists():
            feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_file, index=False)
        logger.info(f"Feedback recorded for query: {feedback.query}")
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

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