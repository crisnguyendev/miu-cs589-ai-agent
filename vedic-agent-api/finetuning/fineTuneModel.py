from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
from dotenv import load_dotenv
import os
import logging

from data.file_utils import find_file
# Updated import path to match directory structure
from finetuning.loadFeedback import load_feedback
from finetuning.mongo_utils import mongo_connection

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fine_tune_model(positive_pairs, negative_pairs, model_name="all-MiniLM-L6-v2",
                    output_path="../output/fine_tuned_model"):
    # Log the output path
    logger.info(f"Saving fine-tuned model to: {output_path}")

    # Load the original model
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("SentenceTransformer model loaded successfully")

    # Load verse metadata to get verse text
    verses_file_path = find_file(os.path.dirname(os.path.abspath(__file__)), "verses_metadata.csv")
    logger.info(f"Found verses_metadata.csv at: {verses_file_path}")
    verses_df = pd.read_csv(verses_file_path)
    logger.info(f"Loaded verse metadata with {len(verses_df)} entries")

    # Prepare training examples
    train_examples = []

    # Positive pairs (label=1, similar)
    for _, row in positive_pairs.iterrows():
        verse_text = verses_df[verses_df["verse_id"] == row["verse_id"]]["text"].iloc[0]
        train_examples.append(InputExample(texts=[row["query"], verse_text], label=1.0))

    # Negative pairs (label=0, dissimilar)
    for _, row in negative_pairs.iterrows():
        verse_text = verses_df[verses_df["verse_id"] == row["verse_id"]]["text"].iloc[0]
        train_examples.append(InputExample(texts=[row["query"], verse_text], label=0.0))

    if not train_examples:
        logger.warning("No training examples available for fine-tuning.")
        return

    # Log training examples info
    logger.info(f"Number of training examples: {len(train_examples)}")
    if train_examples:
        sample = train_examples[0]
        logger.info(f"Sample training example: {sample.texts}, label: {sample.label}")

    # Define the training dataset and dataloader
    logger.info("Creating DataLoader for training examples")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
    logger.info("DataLoader created successfully")

    # Use Contrastive Loss
    # Put similar embeddings closer together and dissimilar further apart (in vector space)
    logger.info("Setting up Contrastive Loss")
    train_loss = losses.ContrastiveLoss(model=model)
    logger.info("Contrastive Loss set up successfully")

    # Fine-tune the model
    logger.info("Starting model fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        # Go through entire dataset once
        epochs=1,
        # Helps stabilize at beginning
        warmup_steps=int(len(train_dataloader) * 0.1),  # 10% of steps depending on training data size
        output_path=output_path,
        show_progress_bar=False  # Disable tqdm progress bar for Render
    )
    logger.info(f"Model fine-tuned and saved to {output_path}")

    # Delete feedback from MongoDB after fine-tuning
    try:
        logger.info("Deleting feedback records from MongoDB after fine-tuning")
        mongo_connection.feedback_collection.delete_many({})
        logger.info("Feedback records deleted from MongoDB after fine-tuning.")
    except Exception as e:
        logger.error(f"Error deleting feedback from MongoDB: {e}")
    finally:
        logger.info("Closing MongoDB connection")
        mongo_connection.close()

# Run fine-tuning
if __name__ == "__main__":
    logger.info("Starting fine-tuning script")
    positive_pairs, negative_pairs = load_feedback()
    if positive_pairs is not None and negative_pairs is not None:
        fine_tune_model(positive_pairs, negative_pairs)
    else:
        logger.error("Failed to load feedback data for fine-tuning")