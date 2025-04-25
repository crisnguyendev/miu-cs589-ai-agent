import pandas as pd

from finetuning.mongo_utils import mongo_connection


def load_feedback(max_entries=100):
    try:
        # Fetch the most recent feedback from MongoDB, limited to max_entries
        feedback_records = list(mongo_connection.feedback_collection.find().sort("timestamp", -1).limit(max_entries))
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

# Example usage
if __name__ == "__main__":
    positive_pairs, negative_pairs = load_feedback()
    if positive_pairs is not None and negative_pairs is not None:
        print("Positive pairs:", len(positive_pairs))
        print("Negative pairs:", len(negative_pairs))