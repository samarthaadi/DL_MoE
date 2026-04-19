#!/usr/bin/env bash
# Launch the Flask API server for the web demo.
# Open website/index.html in a browser after starting.
set -e
cd "$(dirname "$0")/../.."

echo "Starting API server on http://localhost:5000 ..."
echo "Open website/index.html in your browser."
python api.py
