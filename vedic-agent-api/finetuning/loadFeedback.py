import pandas as pd
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB setup
mongo_uri = os.getenv("MONGODB_URI")
if not mongo_uri:
    raise ValueError("MONGODB_URI not set in environment variables")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["vedic-agent"]
feedback_collection = db["feedback"]

def load_feedback(max_entries=100):
    try:
        # Fetch the most recent feedback from MongoDB, limited to max_entries
        feedback_records = list(feedback_collection.find().sort("timestamp", -1).limit(max_entries))
        if not feedback_records:
            print("No feedback data found in MongoDB.")
            return None, None

        # Convert to DataFrame
        feedback_df = pd.DataFrame(feedback_records)

        # Select relevant columns (same as the original CSV format)
        feedback_df = feedback_df[["query", "verse_id", "is_relevant"]]

        # Split into positive and negative pairs
        positive_pairs = feedback_df[feedback_df["is_relevant"]][["query", "verse_id"]]
        negative_pairs = feedback_df[~feedback_df["is_relevant"]][["query", "verse_id"]]
        return positive_pairs, negative_pairs
    except Exception as e:
        print(f"Error loading feedback from MongoDB: {e}")
        return None, None
    finally:
        mongo_client.close()

# Example usage
if __name__ == "__main__":
    positive_pairs, negative_pairs = load_feedback()
    if positive_pairs is not None and negative_pairs is not None:
        print("Positive pairs:", len(positive_pairs))
        print("Negative pairs:", len(negative_pairs))