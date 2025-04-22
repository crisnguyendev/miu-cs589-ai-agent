import re
import pandas as pd
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def roman_to_int(roman):
    """
    Convert Roman numeral to integer (e.g., 'XVI' -> 16).
    Returns original string if not a valid Roman numeral (e.g., 'peace_chant').
    """
    if not roman or roman.lower() == "peace_chant":
        return roman
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    result = 0
    prev_value = 0
    for char in reversed(roman.upper()):
        curr_value = roman_values.get(char, 0)
        if curr_value >= prev_value:
            result += curr_value
        else:
            result -= curr_value
        prev_value = curr_value
    return str(result) if result > 0 else roman

def load_upanishad_verses(file_path, upanishad_name="Isa Upanishad"):
    """
    Parse verses from an Upanishad text into a list of dicts, excluding commentary.
    Args:
        file_path (str): Path to the text file (e.g., ../resources/upanishads-isa.txt).
        upanishad_name (str): Name of the Upanishad (e.g., "Isa Upanishad").
    Returns:
        List of dicts with verse_id, text, source.
    """
    # Read the file
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    current_part = "0"  # Default chapter if no parts
    current_verse = None
    verse_text = []
    in_verse = False
    book = "Upanishads"
    specification = upanishad_name.split()[0]  # E.g., "Isa" from "Isa Upanishad"

    # Regex patterns
    part_pattern = re.compile(r'^\s*Part\s+(?:First|Second|Third|Fourth|Fifth|Sixth|[IVXLC]+)\s*$', re.IGNORECASE)
    verse_pattern = re.compile(r'^\s*(?:[IVXLC]+|\d+)\s*$')  # Roman numerals or digits
    chant_pattern = re.compile(r'^\s*Peace Chant\s*$', re.IGNORECASE)

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            if in_verse and verse_text:
                # End verse at blank line to exclude commentary
                verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                source = f"{upanishad_name} {current_part}.{roman_to_int(current_verse)}"
                data.append({
                    "verse_id": verse_id,
                    "text": " ".join(verse_text).strip(),
                    "source": source
                })
                verse_text = []
                in_verse = False
            continue

        # Check for part header
        if part_pattern.match(line):
            if current_verse and verse_text:
                # Save previous verse
                verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                source = f"{upanishad_name} {current_part}.{roman_to_int(current_verse)}"
                data.append({
                    "verse_id": verse_id,
                    "text": " ".join(verse_text).strip(),
                    "source": source
                })
                verse_text = []
                in_verse = False
            # Extract part number
            part_name = line.lower().replace('part', '').strip()
            part_map = {
                'first': '1', 'second': '2', 'third': '3',
                'fourth': '4', 'fifth': '5', 'sixth': '6',
                'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6'
            }
            current_part = part_map.get(part_name, part_name)
            current_verse = None
            continue

        # Check for peace chant
        if chant_pattern.match(line):
            if current_verse and verse_text:
                # Save previous verse
                verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                source = f"{upanishad_name} {current_part}.{roman_to_int(current_verse)}"
                data.append({
                    "verse_id": verse_id,
                    "text": " ".join(verse_text).strip(),
                    "source": source
                })
                verse_text = []
            current_verse = "peace_chant"
            in_verse = True
            continue

        # Check for verse number
        if verse_pattern.match(line):
            if current_verse and verse_text:
                # Save previous verse
                verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                source = f"{upanishad_name} {current_part}.{roman_to_int(current_verse)}"
                data.append({
                    "verse_id": verse_id,
                    "text": " ".join(verse_text).strip(),
                    "source": source
                })
                verse_text = []
            current_verse = line.strip()
            in_verse = True
            continue

        # Collect verse text
        if in_verse:
            verse_text.append(line)

    # Save the last verse
    if current_verse and verse_text:
        verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
        source = f"{upanishad_name} {current_part}.{roman_to_int(current_verse)}"
        data.append({
            "verse_id": verse_id,
            "text": " ".join(verse_text).strip(),
            "source": source
        })

    logger.info(f"Parsed {len(data)} verses from {file_path}")
    return data

def clean_text(text):
    """
    Normalize text by removing extra whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()

def main():
    parser = argparse.ArgumentParser(description="Parse Upanishad texts into verses.")
    parser.add_argument("--file_path", default="../resources/upanishads-kena.txt", help="Path to text file")
    parser.add_argument("--upanishad_name", default="Kena Upanishad", help="Name of the Upanishad")
    parser.add_argument("--output_path", default="../output/verses.csv", help="Output CSV path")
    args = parser.parse_args()

    # Parse verses
    verses = load_upanishad_verses(args.file_path, args.upanishad_name)

    # Clean text
    for verse in verses:
        verse["text"] = clean_text(verse["text"])

    # Save to CSV
    if verses:
        df = pd.DataFrame(verses)
        output_path = Path(args.output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')
            else:
                df.to_csv(output_path, mode='w', header=True, index=False, encoding='utf-8')
            logger.info(f"Appended {len(verses)} verses to {output_path}")
        except Exception as e:
            logger.error(f"Error saving to {output_path}: {e}")
    else:
        logger.warning(f"No verses parsed for {args.file_path}")

if __name__ == "__main__":
    main()