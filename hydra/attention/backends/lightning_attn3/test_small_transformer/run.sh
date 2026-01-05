#!/bin/bash

# Activate venv
source /home/tim/venvs/llm/bin/activate

# Install requirements
pip install -r requirements.txt

# Run training
python train.py