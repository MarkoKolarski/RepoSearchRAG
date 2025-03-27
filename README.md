# CodeRAG: Repository Question Answering System

## Overview
CodeRAG is a Retrieval-Augmented Generation (RAG) system designed to answer questions over code repositories by retrieving and summarizing relevant files.

## Features
- GitHub repository cloning
- Semantic file retrieval
- Query expansion
- LLM-based file summarization
- Evaluation metrics (Recall@K)

## Requirements
- Python 3.8+
- Dependencies:
  * sentence-transformers
  * transformers
  * faiss-cpu
  * gitpython
  * nltk
  * chardet
  * numpy

## Installation
```bash
pip install sentence-transformers transformers faiss-cpu gitpython nltk chardet numpy
python -m nltk.downloader punkt wordnet
```

## Usage
```bash
python repository_utils.py \
    --repo_url https://github.com/viarotel-org/escrcpy \
    --repo_path repository \
    --top_k 10
```

## Customization
- Modify `test_queries` in `main()` for different evaluation scenarios
- Change embedding/summarization models in `CodeRAGSystem` initialization

## Evaluation Metrics
- Recall@K: Measures the fraction of relevant files retrieved
- Supports custom ground truth datasets

## Performance Considerations
- Model selection impacts retrieval quality
- Larger repositories may require more computational resources