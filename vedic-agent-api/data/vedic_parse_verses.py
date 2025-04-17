import re
import pandas as pd
from pathlib import Path
import logging
import argparse
import unicodedata
from typing import List, Dict

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def roman_to_int(roman: str) -> str:
    """
    Convert Roman numeral to integer (e.g., 'XVI' -> 16).
    Returns original string if not a valid Roman numeral (e.g., 'peace_chant').
    """
    if not roman or roman.lower() == "peace_chant":
        return roman
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    result = 0
    prev_value = 0
    try:
        for char in reversed(roman.upper()):
            curr_value = roman_values.get(char, 0)
            if not curr_value:
                logger.warning(f"Invalid Roman numeral character: {char} in {roman}")
                return roman
            if curr_value >= prev_value:
                result += curr_value
            else:
                result -= curr_value
            prev_value = curr_value
        return str(result) if result > 0 else roman
    except Exception as e:
        logger.error(f"Error converting Roman numeral {roman}: {e}")
        return roman

def clean_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and special characters.
    """
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_upanishad_verses(file_path: str, upanishad_name: str = "Isa Upanishad") -> List[Dict]:
    """
    Parse verses from an Upanishad text into a list of dicts, excluding commentary.
    Args:
        file_path (str): Path to the text file (e.g., upanishads-kena.txt).
        upanishad_name (str): Name of the Upanishad (e.g., "Isa Upanishad").
    Returns:
        List of dicts with id, book, chapter, verse, text_en.
    """
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return []

    data = []
    current_part = "1"  # Default chapter
    current_verse = None
    verse_text = []
    in_verse = False
    book = "Upanishads"
    specification = upanishad_name.split()[0]  # E.g., "Isa" from "Isa Upanishad"

    # Flexible regex patterns
    part_pattern = re.compile(r'^\s*(?:Part\s+)?(?:First|Second|Third|Fourth|Fifth|Sixth|[IVXLC]+)\s*(?:\.|\s|$)', re.IGNORECASE)
    verse_pattern = re.compile(r'^\s*(?:Verse\s+)?([IVXLC]+|\d+)\s*(?:\.|\s|$)', re.IGNORECASE)
    chant_pattern = re.compile(r'^\s*Peace\s+Chant\s*$', re.IGNORECASE)
    commentary_pattern = re.compile(r'^\s*(?:Note|Commentary|Explanation|Here\s+ends|This\s+Upanishad|Peace\s+Chant\s+.*|[-A-Za-z]+-Upanishad|Translated\s+by\s+.*|Among\s+the\s+Upanishads|The\s+[A-Za-z]+-Upanishad):?', re.IGNORECASE)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                logger.debug(f"Line {line_num}: {line[:50]}...")

                # Handle commentary
                if commentary_pattern.match(line):
                    if in_verse and verse_text:
                        cleaned_text = clean_text(" ".join(verse_text))
                        if len(cleaned_text) > 0:
                            verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                            data.append({
                                "id": verse_id,
                                "book": book,
                                "chapter": int(current_part) if current_part.isdigit() else 1,
                                "verse": roman_to_int(current_verse),
                                "text_en": cleaned_text
                            })
                            logger.debug(f"Verse saved: {verse_id}, text: {cleaned_text[:50]}...")
                        verse_text = []
                        in_verse = False
                    logger.debug(f"Detected commentary: {line[:50]}...")
                    continue

                # Handle empty lines
                if not line:
                    if in_verse and verse_text:
                        cleaned_text = clean_text(" ".join(verse_text))
                        if len(cleaned_text) > 0:
                            verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                            data.append({
                                "id": verse_id,
                                "book": book,
                                "chapter": int(current_part) if current_part.isdigit() else 1,
                                "verse": roman_to_int(current_verse),
                                "text_en": cleaned_text
                            })
                            logger.debug(f"Verse saved: {verse_id}, text: {cleaned_text[:50]}...")
                        verse_text = []
                        in_verse = False
                    continue

                # Handle part
                part_match = part_pattern.match(line)
                if part_match:
                    if in_verse and verse_text:
                        cleaned_text = clean_text(" ".join(verse_text))
                        if len(cleaned_text) > 0:
                            verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                            data.append({
                                "id": verse_id,
                                "book": book,
                                "chapter": int(current_part) if current_part.isdigit() else 1,
                                "verse": roman_to_int(current_verse),
                                "text_en": cleaned_text
                            })
                            logger.debug(f"Verse saved: {verse_id}, text: {cleaned_text[:50]}...")
                        verse_text = []
                        in_verse = False
                    part_name = line.lower().replace('part', '').strip().strip('.')
                    part_map = {
                        'first': '1', 'second': '2', 'third': '3',
                        'fourth': '4', 'fifth': '5', 'sixth': '6',
                        'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6'
                    }
                    current_part = part_map.get(part_name, roman_to_int(part_name))
                    current_verse = None
                    logger.debug(f"Detected part: {current_part}")
                    continue

                # Handle peace chant
                if chant_pattern.match(line):
                    if current_verse and verse_text:
                        cleaned_text = clean_text(" ".join(verse_text))
                        if len(cleaned_text) > 0:
                            verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                            data.append({
                                "id": verse_id,
                                "book": book,
                                "chapter": int(current_part) if current_part.isdigit() else 1,
                                "verse": roman_to_int(current_verse),
                                "text_en": cleaned_text
                            })
                            logger.debug(f"Verse saved: {verse_id}, text: {cleaned_text[:50]}...")
                        verse_text = []
                    current_verse = "peace_chant"
                    in_verse = True
                    logger.debug(f"Detected peace chant")
                    continue

                # Handle verse
                verse_match = verse_pattern.match(line)
                if verse_match:
                    if current_verse and verse_text:
                        cleaned_text = clean_text(" ".join(verse_text))
                        if len(cleaned_text) > 0:
                            verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                            data.append({
                                "id": verse_id,
                                "book": book,
                                "chapter": int(current_part) if current_part.isdigit() else 1,
                                "verse": roman_to_int(current_verse),
                                "text_en": cleaned_text
                            })
                            logger.debug(f"Verse saved: {verse_id}, text: {cleaned_text[:50]}...")
                        verse_text = []
                    current_verse = verse_match.group(1)
                    in_verse = True
                    logger.debug(f"Detected verse: {current_verse}")
                    verse_text.append(line[verse_match.end():].strip())
                    continue

                # Accumulate verse text
                if in_verse:
                    verse_text.append(line)
                    logger.debug(f"Appending to verse text: {line[:50]}...")

            # Save final verse
            if in_verse and verse_text:
                cleaned_text = clean_text(" ".join(verse_text))
                if len(cleaned_text) > 0:
                    verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                    data.append({
                        "id": verse_id,
                        "book": book,
                        "chapter": int(current_part) if current_part.isdigit() else 1,
                        "verse": roman_to_int(current_verse),
                        "text_en": cleaned_text
                    })
                    logger.debug(f"Verse saved: {verse_id}, text: {cleaned_text[:50]}...")

    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {e}")
        return data  # Return partial results

    logger.info(f"Parsed {len(data)} verses from {file_path}")
    return data

def save_verses(verses: List[Dict], output_path: str):
    if not verses:
        logger.warning(f"No verses to save for {output_path}")
        return
    df = pd.DataFrame(verses)
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(output_path, mode='w', header=True, index=False, encoding='utf-8')
        logger.info(f"Saved {len(verses)} verses to {output_path}")
        logger.info(f"Dataset stats: {df.groupby(['book', 'chapter']).size().to_dict()}")
    except Exception as e:
        logger.error(f"Error saving to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Parse Upanishad texts into verses.")
    parser.add_argument("--file_path", default="../resources/upanishads-kena.txt", help="Path to text file")
    parser.add_argument("--upanishad_name", default="Kena Upanishad", help="Name of the Upanishad")
    parser.add_argument("--output_path", default="../output/verses.csv", help="Output CSV path")
    args = parser.parse_args()

    logger.info(f"Processing {args.file_path} for {args.upanishad_name}")
    verses = load_upanishad_verses(args.file_path, args.upanishad_name)
    save_verses(verses, args.output_path)

if __name__ == "__main__":
    main()