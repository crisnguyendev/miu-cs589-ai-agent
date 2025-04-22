import logging
import os

import pdfplumber

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# extract text

def extract_text_from_pdf(pdf_path: str, output_name: str):
    logger.info(f"Started extracting")
    output_path = f"../resources/{output_name}.txt"
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Only add non-empty pages
                full_text += page_text + "\n"

    # Ensure resources folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to .txt file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    return full_text



def split_into_chapters(whole_text):
    # Dictionary to store chapter texts: { (kanda, chapter_num): chapter_text }
    chapter_texts = {}

    # Define kandas (focusing on Bala Kanda for now)
    kandas = ["Bala Kanda"]
    current_kanda = None
    current_chapter = None
    current_text = []

    # Split text into lines for easier processing
    lines = whole_text.splitlines()

    # Regex to detect chapter headers (e.g., "Chapter 1", "Bala Kanda Chapter 1")
    chapter_pattern = re.compile(r'^(Bala Kanda )?Chapter (\d+)', re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Check for kanda header
        if any(kanda in line for kanda in kandas):
            current_kanda = "Bala"  # Simplify to "Bala" for verse_id
            continue

        # Check for chapter header
        match = chapter_pattern.match(line)
        if match and current_kanda:  # Only process if we’re in a kanda
            # Save previous chapter’s text
            if current_chapter and current_text:
                chapter_texts[(current_kanda, current_chapter)] = " ".join(current_text)
                current_text = []

            # Start new chapter
            current_chapter = int(match.group(2))  # Extract chapter number
            continue

        # Add line to current chapter’s text
        if current_kanda and current_chapter:
            current_text.append(line)

    # Save the last chapter
    if current_kanda and current_chapter and current_text:
        chapter_texts[(current_kanda, current_chapter)] = " ".join(current_text)

    return chapter_texts

from nltk import sent_tokenize
import re

# Example: Split a chapter's text into pseudo-verses
def segment_chapter(chapter_text, kanda, chapter_num, max_words=150):
    sentences = sent_tokenize(chapter_text)
    pseudo_verses = []
    current_chunk = []
    word_count = 0
    verse_num = 1

    for sent in sentences:
        words = sent.split()
        word_count += len(words)
        current_chunk.append(sent)
        if word_count >= max_words:
            verse_id = f"Ramayana+{kanda}-{chapter_num}-{verse_num}"
            pseudo_verses.append({
                "verse_id": verse_id,
                "text": " ".join(current_chunk),
                "source": f"Ramayana {kanda}.{chapter_num}"
            })
            current_chunk = []
            word_count = 0
            verse_num += 1

    # Handle remaining sentences
    if current_chunk:
        verse_id = f"Ramayana+{kanda}-{chapter_num}-{verse_num}"
        pseudo_verses.append({
            "verse_id": verse_id,
            "text": " ".join(current_chunk),
            "source": f"Ramayana {kanda}.{chapter_num}"
        })
    return pseudo_verses

# Example usage
pdf_path = "../resources/The.Ramayana.of.Valmiki.by.Hari.Prasad.Shastri.pdf"
output_name = "ramayana_full_text_by_hari_prasad_shastri"
whole_text = extract_text_from_pdf(pdf_path, output_name)
# pseudo_verses = segment_chapter(chapter_text, "Bala", 1)