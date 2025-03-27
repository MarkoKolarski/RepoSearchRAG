import os
import re
import chardet
import nltk
from typing import Dict, List, Optional
from git import Repo
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.corpus import wordnet
import numpy as np
import faiss

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def clone_repository(url: str, dest_folder: str) -> Optional[str]:
    """
    Clone a GitHub repository to a local directory.
    
    Args:
        url (str): GitHub repository URL
        dest_folder (str): Destination folder for cloning
    
    Returns:
        Optional[str]: Path to cloned repository or None if failed
    """
    if os.path.exists(dest_folder):
        print(f"Folder {dest_folder} already exists. Skipping cloning.")
        return dest_folder
    
    try:
        print(f"Cloning repository from {url}...")
        Repo.clone_from(url, dest_folder)
        print("Cloning completed successfully.")
        return dest_folder
    except Exception as e:
        print(f"Error during cloning: {e}")
        return None

def read_file_with_encoding(file_path: str) -> Optional[str]:
    """
    Read file with intelligent encoding detection.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        Optional[str]: File contents or None if reading fails
    """
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
        
        with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
            return f.read()
    except Exception as e:
        fallback_encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        for encoding in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    return f.read()
            except Exception:
                continue
        
        print(f"Failed to read {file_path} with all encoding attempts.")
        return None

def prepare_repository_files(repo_path: str) -> Dict[str, str]:
    """
    Collect and read files from repository.
    
    Args:
        repo_path (str): Path to repository
    
    Returns:
        Dict[str, str]: Dictionary of file paths and their contents
    """
    allowed_extensions = ['.py', '.js', '.md', '.txt', '.json']
    
    def collect_files(directory: str) -> List[str]:
        file_paths = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                
                if any(full_path.endswith(ext) for ext in allowed_extensions):
                    file_paths.append(full_path)
        
        return file_paths

    files = collect_files(repo_path)
    
    file_contents = {}
    for file_path in files:
        content = read_file_with_encoding(file_path)
        if content:
            file_contents[file_path] = content
    
    return file_contents

class CodeRAGSystem:
    def __init__(
        self, 
        embedding_model: str = 'all-MiniLM-L6-v2', 
        summarization_model: str = "facebook/bart-large-cnn"
    ):
        """
        Initialize CodeRAG system with embeddings and summarization.
        
        Args:
            embedding_model (str): Sentence transformer model
            summarization_model (str): Hugging Face summarization model
        """
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Summarization pipeline
        try:
            self.summarizer = pipeline("summarization", model=summarization_model)
        except Exception as e:
            print(f"Summarization model load error: {e}")
            self.summarizer = None

        # FAISS index
        self.faiss_index = None
        self.file_paths = []

    def create_embeddings(self, file_contents: Dict[str, str]):
        """
        Create embeddings for repository files.
        
        Args:
            file_contents (Dict[str, str]): Dictionary of file paths and contents
        """
        contents = list(file_contents.values())
        self.file_paths = list(file_contents.keys())
        
        embeddings = self.embedding_model.encode(contents)
        dimension = embeddings.shape[1]
        
        # L2 distance index
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)

    def retrieve_files(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve most relevant files for a query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
        
        Returns:
            List[str]: List of relevant file paths
        """
        if self.faiss_index is None:
            raise ValueError("Embeddings not created. Call create_embeddings first.")
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        return [self.file_paths[i] for i in indices[0]]

    def generate_summary(self, file_content: str, max_length: int = 150) -> str:
        """
        Generate summary for a file content with dynamic length adjustment.
        
        Args:
            file_content (str): File content to summarize
            max_length (int): Maximum summary length
        
        Returns:
            str: Summarized content
        """
        if not self.summarizer:
            return "Summarization unavailable"

        # Trim extremely long inputs
        if len(file_content) > 1000:
            file_content = file_content[:1000]

        try:
            # Dynamically adjust max_length
            adjusted_max_length = min(
                max_length, 
                max(30, int(len(file_content) * 0.3))  # Use 30% of input length, minimum 30
            )
            
            # Ensure min_length is less than max_length
            adjusted_min_length = max(10, int(adjusted_max_length * 0.5))

            summary = self.summarizer(
                file_content, 
                max_length=adjusted_max_length, 
                min_length=adjusted_min_length, 
                do_sample=False
            )
            
            return summary[0]['summary_text'] if summary else "Unable to generate summary"
        except Exception as e:
            print(f"Summary generation error: {e}")
            return "Unable to generate summary"

class QueryExpander:
    def __init__(self):
        """
        Initialize query expander with NLTK resources.
        """
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)

    def expand_query(self, query: str, context_words: Optional[List[str]] = None) -> List[str]:
        """
        Expand query with synonyms and context.
        
        Args:
            query (str): Original query
            context_words (Optional[List[str]]): Additional context words
        
        Returns:
            List[str]: List of expanded queries
        """
        try:
            tokens = nltk.word_tokenize(query.lower())
            
            # Get synonyms
            expanded_tokens = tokens + [
                syn for token in tokens 
                for syn in self._get_synonyms(token)[:2]
            ]
            
            # Add context words
            if context_words:
                expanded_tokens.extend(context_words)
            
            # Generate query variations
            variations = [
                ' '.join(expanded_tokens),
                query.lower(),
                ' '.join(reversed(tokens))
            ]
            
            return list(set(variations))
        except Exception as e:
            print(f"Query expansion error: {e}")
            return [query.lower()]

    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word.
        
        Args:
            word (str): Word to find synonyms for
        
        Returns:
            List[str]: List of synonyms
        """
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
        except Exception as e:
            print(f"Synonym retrieval error for {word}: {e}")
        return list(synonyms)

class RAGEvaluator:
    def __init__(self, ground_truth: Dict[str, List[str]]):
        """
        Initialize evaluator with ground truth data.
        
        Args:
            ground_truth (Dict[str, List[str]]): Reference file mappings
        """
        self.ground_truth = ground_truth

    def calculate_recall(self, retrieved_results: Dict[str, List[str]], k: int = 10) -> float:
        """
        Calculate Recall@K metric.
        
        Args:
            retrieved_results (Dict[str, List[str]]): Retrieved results for each query
            k (int): Top-K results to consider
        
        Returns:
            float: Average recall
        """
        recalls = []
        for query, expected_files in self.ground_truth.items():
            retrieved_set = set(retrieved_results.get(query, [])[:k])
            expected_set = set(expected_files)
            
            recall = len(retrieved_set.intersection(expected_set)) / len(expected_set) \
                if expected_set else (1.0 if not retrieved_set else 0.0)
            
            recalls.append(recall)
        
        return np.mean(recalls)

    def generate_report(self, retrieved_results: Dict[str, List[str]], k: int = 10) -> Dict:
        """
        Generate evaluation report.
        
        Args:
            retrieved_results (Dict[str, List[str]]): Retrieved results
            k (int): Top-K results to analyze
        
        Returns:
            Dict: Evaluation metrics
        """
        recall = self.calculate_recall(retrieved_results, k)
        print(f"Evaluation Report:\nRecall@{k}: {recall:.4f}")
        
        return {
            'recall': recall,
            'k': k
        }