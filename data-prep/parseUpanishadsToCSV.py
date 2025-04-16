import re
import pandas as pd
from pathlib import Path

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
        file_path (str): Path to the text file (e.g., pg3283.txt).
        upanishad_name (str): Name of the Upanishad (e.g., "Isa Upanishad").
    Returns:
        List of dicts with id, book, chapter, verse, text_en.
    """
    # Read the file
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
                data.append({
                    "id": verse_id,
                    "book": book,
                    "chapter": int(current_part),
                    "verse": roman_to_int(current_verse),
                    "text_en": " ".join(verse_text).strip()
                })
                verse_text = []
                in_verse = False
            continue

        # Check for part header
        if part_pattern.match(line):
            if current_verse and verse_text:
                # Save previous verse
                verse_id = f"{book}+{specification}-{current_part}-{roman_to_int(current_verse)}"
                data.append({
                    "id": verse_id,
                    "book": book,
                    "chapter": int(current_part),
                    "verse": roman_to_int(current_verse),
                    "text_en": " ".join(verse_text).strip()
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
                data.append({
                    "id": verse_id,
                    "book": book,
                    "chapter": int(current_part),
                    "verse": roman_to_int(current_verse),
                    "text_en": " ".join(verse_text).strip()
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
                data.append({
                    "id": verse_id,
                    "book": book,
                    "chapter": int(current_part),
                    "verse": roman_to_int(current_verse),
                    "text_en": " ".join(verse_text).strip()
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
        data.append({
            "id": verse_id,
            "book": book,
            "chapter": int(current_part),
            "verse": roman_to_int(current_verse),
            "text_en": " ".join(verse_text).strip()
        })

    return data

def clean_text(text):
    """
    Normalize text by removing extra whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()

def main():
    # File path to the Gutenberg text
    file_path = "data/upanishads-kena.txt"  # Update with actual path
    upanishad_name = "Kena Upanishad"  # Or "Katha Upanishad"

    # Parse verses
    verses = load_upanishad_verses(file_path, upanishad_name)

    # Clean text
    for verse in verses:
        verse["text_en"] = clean_text(verse["text_en"])

    # Save to CSV


    df = pd.DataFrame(verses)
    output_path = "verses.csv"
        # Check if file exists to avoid writing header multiple times
    if Path(output_path).exists():
        df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df.to_csv(output_path, mode='w', header=True, index=False, encoding='utf-8')
    print(f"Appended {len(verses)} verses to {output_path}")

if __name__ == "__main__":
    main()