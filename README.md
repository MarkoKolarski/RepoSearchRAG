# 🔍 LLM Listwise Reranker for CodeRAG

**Internship Project — Retrieval-Augmented Generation (RAG) for Code Repositories**  

---

## 🧠 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for **question-answering over a GitHub code repository**. It takes a GitHub URL as input, indexes the codebase, and allows users to ask natural language questions about the code. The system retrieves and **reranks relevant files** using advanced techniques such as **query expansion**, **diverse retrieval strategies**, and an **LLM-based listwise reranker**.

> ✅ This implementation fully satisfies all required, additional and optional (bonus) features outlined in the task specification.

---

## 🛠️ Clear Instructions for Running

### 🔧 Prerequisites
- Python 3.8+
- Git

### 📦 Installation
```bash
git clone https://github.com/MarkoKolarski/RepoSearchRAG.git
cd RepoSearchRAG
pip install -r requirements.txt
```

---

### ▶️ Run the System

You can run the system using the `main.py` script. Below are all the available command-line arguments with their descriptions and default values:

#### 🔧 Repository Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--repo_url` | `str` | `"https://github.com/viarotel-org/escrcpy"` | GitHub repository URL to clone |
| `--repo_path` | `str` | `"repository"` | Local directory where the repository will be cloned and processed |

#### 🔎 Retrieval Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--top_k` | `int` | `10` | Number of top search results (files) to return |
| `--retrieval_strategy` | `str` | `"diverse"` | Retrieval strategy to use. Options: `default`, `probabilistic`, `diverse` |

#### 🧠 Summarization Options *(optional)*

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--generate_summaries` | `flag` | `local model (flan-t5-small)` | Enable generation of file content summaries |
| `--use_large_summarizer` | `flag` | `False` | Use a larger local summarization model (flan-t5-base) for improved quality |
| `--use_gemini` | `flag` | `False` | Use Google Gemini API for summarization |
| `--gemini_api_key` | `str` | `None` | API key for Gemini (can also be set via `GOOGLE_API_KEY` environment variable) |
| `--gemini_model` | `str` | `"gemini-1.5-flash"` | Google Gemini model to use |

---

#### 📌 Example Usage

Basic usage with default settings:
```bash
python main.py
```

Specify different retrieval strategy and number of results:
```bash
python main.py --repo_url https://github.com/viarotel-org/escrcpy --retrieval_strategy probabilistic --top_k 5
```

Enable summarization using Google Gemini:
```bash
python main.py --generate_summaries --use_gemini --gemini_api_key YOUR_API_KEY
```

Use a larger local summarizer:
```bash
python main.py --generate_summaries --use_large_summarizer
```

---

### 🧪 Run Evaluation

You can evaluate the system using the `evaluate_coderag.py` script. The evaluation is based on the **Recall@K** metric and compares retrieved results with ground truth from a dataset.

#### 🛠️ Available Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--repo_url` | `str` | `"https://github.com/viarotel-org/escrcpy"` | GitHub repository URL to evaluate |
| `--repo_path` | `str` | `"repository"` | Local path where the repository will be cloned and processed |
| `--dataset` | `str` | `"escrcpy-commits-generated.json"` | Path to the evaluation dataset (JSON format) |
| `--retrieval_strategy` | `str` | `"diverse"` | Strategy used for retrieval: `default`, `probabilistic`, or `diverse` |
| `--top_k` | `int` | `10` | Number of top results to consider when calculating Recall@K |
| `--output` | `str` | `"evaluation_results.json"` | Path to save the evaluation output results |

---

#### 📌 Example Usage

Run evaluation using default settings:
```bash
python evaluate_coderag.py
```

Run evaluation with a custom dataset and probabilistic retrieval:
```bash
python evaluate_coderag.py --dataset evaluation_dataset.json --retrieval_strategy probabilistic --top_k 5
```

Save results to a specific file:
```bash
python evaluate_coderag.py --output my_results.json
```

---

#### 📁 Evaluation Dataset Format

The evaluation dataset should be a `.json` file with the following structure:

```json
{
  "strategy": "diverse",
  "recall": 0.9861,
  "average_query_time": 4.80809211730957,
  "total_time": 203.51548767089844,
  "retrieved_results": {
    "Question 1": [ "retrieved/file1", "retrieved/file2", ... ],
    "Question 2": [ "retrieved/file1", "retrieved/file2", ... ],
    "Question 3": [ "retrieved/file1", "retrieved/file2", ... ]
  }
}
```

Each entry contains a natural language question and a list of expected relevant file paths (ground truth). The system calculates **Recall@K** based on how many of the expected files are retrieved in the top K results.


---

## ✨ Features and Implementation Details

### 1️⃣ Complete Pipeline from Indexing to RAG

**✔️ Requirement:** System must provide a full pipeline from indexing to retrieval, requiring only a GitHub URL.

**💡 Implementation:**
- Repository is cloned via `clone_repository()` in `repository_utils.py`.
- Files are indexed using `prepare_repository_files()`, supporting `.py`, `.js`, `.md`, etc.
- `AdvancedCodeRAGSystem` class handles retrieval and reranking.

---

### 2️⃣ Scope Limited to a Single Repository

**✔️ Requirement:** System must work with `https://github.com/viarotel-org/escrcpy`.

**💡 Implementation:**
- The repo URL is passed as `--repo_url` argument.
- The system is developed and tested specifically on the `escrcpy` repository for optimal results.

---

### 3️⃣ Natural Language Question Answering

**✔️ Requirement:** Users can ask natural language questions and get relevant file paths.

**💡 Implementation:**
- `interactive_query_loop()` enables interactive Q&A.
- Queries are expanded via `QueryExpander` (adds synonyms and code-specific terms).
- Retrieval and reranking handled by `AdvancedCodeRAGSystem` and `ListwiseReranker`.

**📤 Output Sample:**
```
Your question: How does the device pairing work?

Top 10 relevant files for: "How does the device pairing work?"
• repository\src\utils\device\generateAdbPairingQR\index.js

• repository\src\dicts\device\index.js

• repository\electron\exposes\adb\helpers\scanner\index.js
• ...
```

---

### 4️⃣ Evaluation with Recall@10

**✔️ Requirement:** System should report Recall@10 scores.

**💡 Implementation:**
- Evaluation is done using `evaluate_coderag.py`.
- `RAGEvaluator` compares retrieved results with ground truth.


**📊 Output Example:**
```
Recall@10: 0.9861
```

---

## 🧠 Advanced Techniques

### 5.1 🏗️ Index Building Algorithm

- Efficiently indexes files using `prepare_repository_files()` with multiprocessing.
- Filters supported extensions only (e.g., `.py`, `.js`, `.md`).

---

### 5.2 🔍 Query Expansion

- Implemented in `QueryExpander`:
  - Uses **WordNet** for synonyms.
  - Adds **code-specific terms** like `function`, `class`, `component`.
  - Extracts **context-aware keywords** from source files.

---

### 5.3 🧠 LLM-Based Listwise Reranker

- Implemented in `ListwiseReranker`.
- Uses cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-12-v2`.
- Applies custom boosting based on:
  - File types (.py, .js)
  - Test file priority (e.g., `test_*.py`)
  - Token overlap & early keyword match

---

### 5.4 💡 Other Techniques

- Multiple retrieval strategies:
  - `default`: Cosine similarity
  - `probabilistic`: Relevance scoring
  - `diverse`: Maximal Marginal Relevance (MMR)

---

## 🏅 Optional Features


### ✅ 1. LLM-Generated Summaries

- Implemented via `AdvancedSummarizer`.
- Supports:
  - **Default local model** (`google/flan-t5-small`) – used if no other options are specified
  - **Larger local model** (`google/flan-t5-base`) – enabled with `--use_large_summarizer`
  - **API-based model** (e.g., **Google Gemini**) – enabled with `--use_gemini`
- Additional options:
  - `--generate_summaries` – enables summarization
  - `--use_large_summarizer` – uses a higher-quality local model
  - `--use_gemini` – uses the Gemini API for summarization
  - `--gemini_api_key` – required for Gemini API (can also be set via `GOOGLE_API_KEY` env variable)
  - `--gemini_model` – specifies the Gemini model (default: `gemini-1.5-flash`)
---

### 📊 Summarization Models – Visual Comparison Table

| Model Type       | Model Name              | How to Enable                                      | Speed ⏱️ | Quality 🧠 | Resource Usage 💻 |
|------------------|--------------------------|----------------------------------------------------|----------|------------|--------------------|
| ✅ **Default Local Model** | `google/flan-t5-small`     | *(default)* Just use `--generate_summaries`        | ⚡ Fast   | ⭐ Good     | 🟢 Low (CPU/GPU)    |
| 🔄 **Large Local Model**   | `google/flan-t5-base`      | `--generate_summaries --use_large_summarizer`      | 🐢 Slower | ⭐⭐⭐ Better | 🟡 Medium–High      |
| ☁️ **Cloud API (Gemini)**  | `gemini-1.5-flash` *(default)* | `--generate_summaries --use_gemini --gemini_api_key <KEY>` | ⚡ Fast   | ⭐⭐ Very Good | 🔵 External (API)   |
| ☁️ **Cloud API (Gemini Pro)** | `gemini-1.5-pro` *(custom)* | Same as above + `--gemini_model gemini-1.5-pro`    | ⚠️ Varies | ⭐⭐⭐⭐ Excellent | 🔵 External (API)   |

---

### ✅ 2. Well-Documented Repository

- README includes setup, usage, architecture, and evaluation.
- All classes and functions are documented with docstrings.

---

### ✅ 3. Latency/Quality Trade-Off Evaluation

- Retrieval strategies allow users to balance speed vs. accuracy.
- Evaluation script logs:
  - Average query time
  - Total evaluation time

---

### ✅ 4. Switching Between LLM & Embedding Models

- `AdvancedSummarizer` supports:
  - Default local model (`flan-t5-small`)
  - Larger local model (`flan-t5-base`) via `--use_large_summarizer`
  - API-based models (e.g., **Google Gemini**) via `--use_gemini`
- `AdvancedCodeRAGSystem` uses `SentenceTransformer` for embeddings, but can be extended to support:
  - OpenAI embeddings
  - Cohere embeddings
  - Custom local embedding models

---

## 🧪 Evaluation Criteria (Checklist)

| Criteria                         | Status | Notes |
|----------------------------------|--------|-------|
| ✅ Functional RAG System         | ✅     | End-to-end pipeline implemented |
| ✅ Accuracy: Recall@10           | ✅     | Achieves high scores |
| ✅ Efficiency                    | ✅     | Multiprocessing and MMR retrieval |
| ✅ Code Quality & Documentation  | ✅     | Modular, well-commented, and extensible |
| ✅ Clear Usage Instructions      | ✅     | Provided in README |

---

## 📁 Project Structure

```
.
├── main.py                          # Entry point for querying
├── evaluate_coderag.py              # Evaluation script (Recall@10)
├── repository_utils.py              # Cloning, indexing, querying logic
├── listwise_reranker.py             # Reranking using LLMs
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── escrcpy-commits-generated.json   # Input dataset for evaluation
└── evaluation_results.json          # Output results from evaluation (optional / generated)
```

---

## ✅ Conclusion

This project fully meets the goals of the internship task, offering a robust, modular, and extendable RAG system for question-answering over code repositories. It includes:

- ✅ A complete end-to-end pipeline  
- ✅ Strong retrieval performance (Recall@10)  
- ✅ Modular support for LLMs and embeddings  
- ✅ Advanced techniques: query expansion, reranking, summarization  
- ✅ Clear documentation and easy-to-use scripts  

> 🔧 If needed, the system can be easily extended to support multiple repositories, APIs, or UI components.

---

## 🙋‍♂️ Contact

For questions, suggestions, or contributions, feel free to open an issue or reach out directly.