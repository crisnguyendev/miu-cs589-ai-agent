from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
from dotenv import load_dotenv
import os

from data.file_utils import find_file
# Updated import path to match directory structure
from finetuning.loadFeedback import load_feedback
from finetuning.mongo_utils import mongo_connection

# Load environment variables
load_dotenv()



def fine_tune_model(positive_pairs, negative_pairs, model_name="all-MiniLM-L6-v2",
                    output_path="../output/fine_tuned_model"):
    # Load the original model
    model = SentenceTransformer(model_name)

    # Load verse metadata to get verse text
    verses_file_path = find_file(os.path.dirname(os.path.abspath(__file__)), "verses_metadata.csv")
    verses_df = pd.read_csv(verses_file_path)

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
        print("No training examples available for fine-tuning.")
        return

    # Debug print to verify train_examples
    print(f"Number of training examples: {len(train_examples)}")
    print(f"Sample training example: {train_examples[0].texts}, label: {train_examples[0].label}")

    # Define the training dataset and dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

    # Use Contrastive Loss
    #  Put similar embeddings closer together and dissimilar further apart (in vector space)
    train_loss = losses.ContrastiveLoss(model=model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        # go through enitre dataset once:
        epochs=1,
        # helps stabilize at beginning:
        warmup_steps= int(len(train_dataloader) * 0.1),  # 10% of steps depending on trainingdata-size
        output_path=output_path
    )
    print(f"Model fine-tuned and saved to {output_path}")

    # Delete feedback from MongoDB after fine-tuning
    try:
        mongo_connection.feedback_collection.delete_many({})
        print("Feedback records deleted from MongoDB after fine-tuning.")
    except Exception as e:
        print(f"Error deleting feedback from MongoDB: {e}")
    finally:
        mongo_connection.close()

# Run fine-tuning
if __name__ == "__main__":
    positive_pairs, negative_pairs = load_feedback()
    if positive_pairs is not None and negative_pairs is not None:
        fine_tune_model(positive_pairs, negative_pairs)