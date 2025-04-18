#!/bin/bash

# Script to run VedicSage FastAPI server
# Runs data generation first to ensure required files exist
# Compatible with older Bash versions

# Get script's directory for relative paths
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(realpath "${SCRIPT_DIR}")"
DATA_DIR="${PROJECT_DIR}/data"
SERVICE_DIR="${PROJECT_DIR}/service"
OUTPUT_DIR="${PROJECT_DIR}/output"
VENV_DIR="${PROJECT_DIR}/.venv"
VENV_ACTIVATE="${VENV_DIR}/bin/activate"

# Function to log messages
log() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Step 1: Verify required files
log "Verifying required files..."
if [ ! -f "${OUTPUT_DIR}/verse_index.faiss" ]; then
  log "Error: verse_index.faiss not found in ${OUTPUT_DIR}"
  exit 1
fi
if [ ! -f "${OUTPUT_DIR}/verses_metadata.csv" ]; then
  log "Error: verses_metadata.csv not found in ${OUTPUT_DIR}"
  exit 1
fi

# Step 2: Activate virtual environment
log "Activating virtual environment..."
if [ ! -f "${VENV_ACTIVATE}" ]; then
  log "Error: Virtual environment not found at ${VENV_ACTIVATE}"
  exit 1
fi
source "${VENV_ACTIVATE}"

# Step 3: Start FastAPI server
log "Starting FastAPI server..."
cd "${SERVICE_DIR}" || {
  log "Error: Could not change to ${SERVICE_DIR}"
  exit 1
}
python vedic_retrieval_api.py &

# Wait for API to start
sleep 5

log "API running at http://localhost:8000"
log "Press CTRL+C to stop the server"

# Keep script running to maintain API
wait