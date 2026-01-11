#!/bin/bash
# HYDRA Training Dashboard - Backend Server

cd "$(dirname "$0")"
source /home/tim/venvs/llm/bin/activate

echo "Starting HYDRA Training Dashboard API on http://0.0.0.0:8765"
uvicorn server:app --host 0.0.0.0 --port 8765 --reload
