import sys
import os
import shutil
import tempfile
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks  # Added BackgroundTasks

from finetuning.mongo_utils import mongo_connection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# Resolve absolute paths for the retriever
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_dir = os.path.join(base_dir, "output")
index_path = os.path.join(output_dir, "verse_index.faiss")
metadata_path = os.path.join(output_dir, "verses_metadata.csv")
model_path = os.path.join(output_dir, "fine_tuned_model")

# Ensure the FAISS index exists at startup
if not os.path.exists(index_path):
    logger.info(f"FAISS index not found at {index_path}. Generating index...")
    update_embeddings_and_index(model_path=model_path, output_index_path=index_path)

# Global retriever instance (will be updated after fine-tuning)
retriever = VedicRetriever(
    index_path=index_path,
    metadata_path=metadata_path,
    model_path=model_path
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VedicSage Retrieval API starting")
    logger.info(f"Working directory: {os.getcwd()}")
    yield
    logger.info("VedicSage Retrieval API shutting down")
    mongo_connection.close()

app.lifespan = lifespan

@app.get("/retrieve", response_model=RetrievalResponse)
async def retrieve_verses(query: str, k: int = 5):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if k < 1 or k > 10:
        raise HTTPException(status_code=400, detail="k must be between 1 and 10")

    try:
        results = retriever.retrieve(query, k)
        logger.info(f"Retrieved {len(results)} verses for query: {query}")
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
        logger.info(f"Retrieved {len(verses)} verses for query: {query}")

        # Step 2: Compose prompt and call OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful assistant knowledgeable in Vedic Science. Even though the Vedas talk about God, it is not a religion. Please do not name any religion, like Hinduism."},
            {"role": "user",
             "content": f"Question: {query}"
                        f"Please answer the question in the following way:"
                        f"If applicable, first give a short, maximum 2 sentences, general answer."
                        f"Then, without quoting them one to one, expand your answer in consideration of the found verses (=context): "
                        f"\n\n{context}\n\n. "
                        f"Mention the sources of the verses you consider in your answer (only the name "
                        f"and only if they were part of the provided context). "
                        f"Please do not go through the sources one by one, summarize and synthesize."
             }
        ]

        logger.info("Calling OpenAI API for answer generation")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )

        answer = response.choices[0].message.content
        logger.info("Answer generated successfully")

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

        # Insert feedback into MongoDB
        logger.info("Inserting feedback into MongoDB")
        mongo_connection.feedback_collection.insert_one(feedback_data)
        logger.info(f"Feedback recorded for query: {feedback.query}")
        logger.debug(f"Feedback data: {feedback_data}")
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

### FINE-TUNE ENDPOINT
async def run_fine_tuning(positive_pairs, negative_pairs, temp_dir: str):
    """Run the fine-tuning process in the background."""
    global retriever
    try:
        logger.info("Calling fine_tune_model to fine-tune the model")
        fine_tune_model(positive_pairs, negative_pairs, model_name="all-MiniLM-L6-v2", output_path=temp_dir)
        logger.info("fine_tune_model completed successfully")

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Retry deleting the existing model directory with a delay
        if os.path.exists(model_path):
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Attempt {attempt + 1} to remove existing model directory: {model_path}")
                    shutil.rmtree(model_path)
                    logger.info("Existing model directory removed successfully")
                    break
                except Exception as e:
                    logger.warning(f"Failed to remove model directory on attempt {attempt + 1}: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(1)  # Wait 1 second before retrying
                    else:
                        logger.error(f"Failed to remove model directory after {max_attempts} attempts: {e}")
                        raise Exception(f"Failed to remove model directory after {max_attempts} attempts: {e}")
        logger.info(f"Moving fine-tuned model from {temp_dir} to {model_path}")
        shutil.move(temp_dir, model_path)
        logger.info("Fine-tuned model moved successfully")

        # Update embeddings and index with the new model
        logger.info("Updating embeddings and FAISS index with the new model")
        update_embeddings_and_index(model_path=model_path, output_index_path=index_path)
        logger.info("Embeddings and FAISS index updated successfully")

        # Reload the retriever with the new model
        logger.info("Reloading VedicRetriever with the new fine-tuned model")
        retriever = VedicRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            model_path=model_path
        )
        logger.info("VedicRetriever reloaded with fine-tuned model and updated index")
    except Exception as e:
        logger.error(f"Error during background fine-tuning: {e}")

@app.post("/fine-tune")
async def fine_tune(background_tasks: BackgroundTasks):  # Added BackgroundTasks parameter
    global retriever
    try:
        logger.info("Starting fine-tuning process")
        positive_pairs, negative_pairs = load_feedback()
        logger.info(f"Loaded feedback: {len(positive_pairs)} positive pairs, {len(negative_pairs)} negative pairs")
        if positive_pairs is None or negative_pairs is None or (len(positive_pairs) == 0 and len(negative_pairs) == 0):
            logger.warning("No feedback data available for fine-tuning")
            raise HTTPException(status_code=400, detail="No feedback data available for fine-tuning")

        # Unload the current model to release any file locks
        logger.info("Unloading current SentenceTransformer model to release file locks")
        retriever.unload_model()

        # Create a temporary directory to save the fine-tuned model
        temp_dir = tempfile.mkdtemp(dir=output_dir)  # Create a temp directory that persists for the background task
        logger.info(f"Using temporary directory for fine-tuning: {temp_dir}")

        # Schedule the fine-tuning process in the background
        background_tasks.add_task(run_fine_tuning, positive_pairs, negative_pairs, temp_dir)

        return {"message": "Fine-tuning process started in the background. Check logs for progress."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error initiating fine-tuning: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate fine-tuning: {str(e)}")

# For testing API
@app.get("/")
async def root():
    return {"message": "Vedic Retrieval API is running"}

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Working directory: {os.getcwd()}")

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)