from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

from pymongo import MongoClient  # Added for MongoDB
from dotenv import load_dotenv
import os

from finetuning.loadFeedback import load_feedback

# Load environment variables
load_dotenv()

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI")
if not mongo_uri:
    raise ValueError("MONGODB_URI not set in environment variables")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["vedic-agent"]
feedback_collection = db["feedback"]

def fine_tune_model(positive_pairs, negative_pairs, model_name="all-MiniLM-L6-v2",
                    output_path="../output/fine_tuned_model"):
    # Load the original model
    model = SentenceTransformer(model_name)

    # Load verse metadata to get verse text
    verses_df = pd.read_csv("../output/verses_metadata.csv")

    # Prepare training examples
    train_examples = []

    # Positive pairs (label=1)
    for _, row in positive_pairs.iterrows():
        verse_text = verses_df[verses_df["verse_id"] == row["verse_id"]]["text"].iloc[0]
        train_examples.append(InputExample(texts=[row["query"], verse_text], label=1.0))

    # Negative pairs (label=0)
    for _, row in negative_pairs.iterrows():
        verse_text = verses_df[verses_df["verse_id"] == row["verse_id"]]["text"].iloc[0]
        train_examples.append(InputExample(texts=[row["query"], verse_text], label=0.0))

    if not train_examples:
        print("No training examples available for finetuning.")
        return

    # Define the training dataset and dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Use Contrastive Loss
    train_loss = losses.ContrastiveLoss(model=model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,  # Keep it simple with 1 epoch for now
        warmup_steps=100,
        output_path=output_path
    )
    print(f"Model fine-tuned and saved to {output_path}")

    # Delete feedback from MongoDB after finetuning
    try:
        feedback_collection.delete_many({})  # Deletes all feedback records
        print("Feedback records deleted from MongoDB after finetuning.")
    except Exception as e:
        print(f"Error deleting feedback from MongoDB: {e}")
    finally:
        mongo_client.close()

# Run finetuning
if __name__ == "__main__":
    positive_pairs, negative_pairs = load_feedback()
    if positive_pairs is not None and negative_pairs is not None:
        fine_tune_model(positive_pairs, negative_pairs)