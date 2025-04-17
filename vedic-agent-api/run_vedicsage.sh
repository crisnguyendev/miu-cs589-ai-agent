#!/bin/bash

# Simplified script to run VedicSage project from Step 1 with relative paths
# Uses all .txt files in resources/, excludes Bhagavad Gita
# No error checks or verse count validation, runs to completion
# Portable across local machines, compatible with older Bash versions

# Get script's directory for relative paths
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(realpath "${SCRIPT_DIR}")"
DATA_DIR="${PROJECT_DIR}/data"
SERVICE_DIR="${PROJECT_DIR}/service"
RESOURCES_DIR="${PROJECT_DIR}/resources"
OUTPUT_DIR="${PROJECT_DIR}/output"
VENV_DIR="${PROJECT_DIR}/.venv"
VENV_ACTIVATE="${VENV_DIR}/bin/activate"

# Function to log messages
log() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Step 0: Setup virtual environment
log "Setting up virtual environment..."
mkdir -p "${VENV_DIR}"
python3 -m venv "${VENV_DIR}" || true
source "${VENV_ACTIVATE}"

# Step 0.1: Install dependencies
log "Installing dependencies..."
pip install pandas numpy sentence-transformers faiss-cpu fastapi uvicorn --quiet || true
log "Dependencies installed"

# Step 0.2: Create directories and clear output
log "Creating directories and clearing output..."
mkdir -p "${RESOURCES_DIR}" "${OUTPUT_DIR}" "${DATA_DIR}" "${SERVICE_DIR}" || true
rm -f "${OUTPUT_DIR}"/* || true
log "Output directory cleared"

# Step 1: Parse Upanishads
log "Step 1: Parsing Upanishad texts..."
cd "${DATA_DIR}" || true
for file in "${RESOURCES_DIR}"/*.txt; do
  if [ -f "$file" ]; then
    relative_file="../resources/$(basename "${file}")"
    filename=$(basename "${file}")
    name="${filename%.txt}"
    name="${name#upanishads-}"
    if [[ "$name" == "Upanishads-Translated-by-Swami-Paramananda-PLAINTEXT" ]]; then
      name="Paramananda Collection Upanishad"
    else
      name="$(echo "$name" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}') Upanishad"
    fi
    log "Parsing ${name} from ${relative_file}..."
    python vedic_parse_verses.py --file_path "${relative_file}" --upanishad_name "${name}" --output_path "${OUTPUT_DIR}/verses.csv" || true
  fi
done

# Step 2: Embed verses
log "Step 2: Generating embeddings..."
python vedic_embed_verses.py \
  --csv_path "${OUTPUT_DIR}/verses.csv" \
  --index_path "${OUTPUT_DIR}/verse_embeddings.faiss" \
  --metadata_path "${OUTPUT_DIR}/verse_metadata.csv" || true

# Step 3: Run API
log "Step 3: Starting FastAPI server..."
cd "${SERVICE_DIR}" || true
python vedic_retrieval_api.py &

# Wait for API to start
sleep 5

log "Project running! API available at http://localhost:8000"
log "Press CTRL+C to stop the server"

# Keep script running to maintain API
wait