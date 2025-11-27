#!/bin/bash
# Install dependencies if needed
pip install -q numpy pandas scikit-learn tensorflow matplotlib textblob

# Run the pipeline
python3 pipeline.py
