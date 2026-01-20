# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sentiment Analysis NLP project using PyTorch and Hugging Face Transformers.

## Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

## Key Dependencies

- PyTorch (torch)
- Hugging Face Transformers
- Hugging Face Datasets
- pandas, numpy

## Project Structure

- `src/data/` - Data loading and preprocessing
- `src/models/` - Model definitions
- `src/training/` - Training loops and utilities
- `src/inference/` - Inference/prediction code
- `configs/` - Configuration files
- `data/raw/` - Raw datasets
- `data/processed/` - Preprocessed data
- `data/labeled/` - Labeled datasets
- `notebooks/` - Jupyter notebooks for experimentation
- `tests/` - Unit tests
