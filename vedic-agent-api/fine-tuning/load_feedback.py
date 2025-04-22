import pandas as pd
from pathlib import Path

def load_feedback(feedback_path="./output/feedback.csv"):
    if not Path(feedback_path).exists():
        print("No feedback data found.")
        return None
    feedback_df = pd.read_csv(feedback_path)
    positive_pairs = feedback_df[feedback_df["is_relevant"]][["query", "verse_id"]]
    negative_pairs = feedback_df[~feedback_df["is_relevant"]][["query", "verse_id"]]
    return positive_pairs, negative_pairs

# Example usage
positive_pairs, negative_pairs = load_feedback()
print("Positive pairs:", len(positive_pairs))
print("Negative pairs:", len(negative_pairs))