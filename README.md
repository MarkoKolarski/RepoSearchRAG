# Advanced CodeRAG: Code Repository Question Answering System

## Overview
An advanced Retrieval-Augmented Generation (RAG) system designed to extract and answer questions from code repositories using state-of-the-art embedding, query expansion, and reranking techniques.

## Features
- 🔍 GitHub Repository Indexing
- 🧠 Query Expansion
- 🚀 Advanced Retrieval Strategies
- 🔬 Listwise Reranking
- 📊 Performance Evaluation

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- Sentence Transformers
- FAISS
- NLTK

## Installation
```bash
git clone https://github.com/your-repo/advanced-coderag.git
cd advanced-coderag
pip install -r requirements.txt
```

## Usage
```bash
python main.py --repo_url https://github.com/viarotel-org/escrcpy \
               --retrieval_strategy diverse \
               --top_k 10
```

### Retrieval Strategies
- `default`: Standard cosine similarity
- `probabilistic`: Probabilistic relevance retrieval
- `diverse`: Maximal Marginal Relevance (MMR)

## Evaluation Metrics
- Recall@K
- Relevance Scoring

## Advanced Techniques
1. Query Expansion
2. Cross-Encoder Reranking
3. Diverse Retrieval Strategies

## Future Work
- Support for more embedding providers
- Enhanced LLM integration
- Expanded evaluation metrics

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.

## License
[MIT](https://choosealicense.com/licenses/mit/)