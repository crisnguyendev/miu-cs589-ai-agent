import sys
import os

from finetuning.mongo_utils import mongo_connection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
import pandas as pd
import logging
import warnings
from openai import OpenAI
from datetime import datetime

from typing import List, Dict
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from finetuning.updateEmbeddingsAndIndex import update_embeddings_and_index
from service.VedicRetriever import VedicRetriever
from service.models import FeedbackRequest, RetrievalResponse
from finetuning.loadFeedback import load_feedback
from finetuning.fineTuneModel import fine_tune_model

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="VedicSage Retrieval API",
              description="Retrieve relevant Vedic verses based on semantic similarity")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://miu-vedic-science.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance (will be updated after fine-tuning)
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
    mongo_connection.close()  # Close MongoDB connection on shutdown

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
        messages = [
            {"role": "system", "content": "You are a helpful assistant knowledgeable in Vedic Science. Eventhough the Vedas talk about God, it is not a religion. Please do not name any religion, like Hinduism."},
            {"role": "user",
             "content": f"Question: {query}"
                        f"Please answer the question in the following way:"
                        f"If applicable, First give a short, maximum 2 sentences, general answer."
                        f"Then, without quoting them one to one, expand your answer in consideration of the found verses (=context): "
                        f"\n\n{context}\n\n. "
                        f"Mention the sources of the verses you consider in your answer (only the name "
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
        traceback.print_exc()
        logger.error(f"Error generating answer for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

### FEEDBACK
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        # Prepare feedback data for MongoDB
        feedback_data = {
            "query": feedback.query,
            "verse_id": feedback.verse_id,
            "is_relevant": feedback.is_relevant,
            "timestamp": datetime.utcnow()
        }

        # Insert feedback into MongoDB using the new connection utility
        mongo_connection.feedback_collection.insert_one(feedback_data)
        logger.info(f"Feedback recorded for query: {feedback.query}")
        logger.info(f"Feedback data: {feedback_data}")
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

### FINE-TUNE ENDPOINT
@app.post("/fine-tune")
async def fine_tune():
    global retriever
    try:
        positive_pairs, negative_pairs = load_feedback()
        if positive_pairs is None or negative_pairs is None or (len(positive_pairs) == 0 and len(negative_pairs) == 0):
            raise HTTPException(status_code=400, detail="No feedback data available for fine-tuning")

        fine_tune_model(positive_pairs, negative_pairs, model_name="all-MiniLM-L6-v2", output_path="../output/fine_tuned_model")
        update_embeddings_and_index(model_path="../output/fine_tuned_model", output_index_path="../output/verse_index.faiss")
        retriever = VedicRetriever(
            index_path="../output/verse_index.faiss",
            metadata_path="../output/verses_metadata.csv",
            model_path="../output/fine_tuned_model"
        )
        logger.info("VedicRetriever reloaded with fine-tuned model and updated index")
        return {"message": "Model fine-tuned, embeddings updated, and retriever reloaded successfully"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error during fine-tuning: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fine-tune model: {str(e)}")

# For testing API
@app.get("/")
async def root():
    return {"message": "Vedic Retrieval API is running"}

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Working directory: {os.getcwd()}")

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)