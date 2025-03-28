import os
import re
import chardet
import nltk
from typing import Dict, List, Optional, Union
from git import Repo
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.corpus import wordnet
import numpy as np
import faiss
from listwise_reranker import ListwiseReranker


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

class AdvancedCodeRAGSystem:
    def __init__(
        self, 
        embedding_model: Union[str, object] = 'all-MiniLM-L6-v2',
        reranker_model: Optional[str] = None,
        retrieval_strategy: str = 'default',
        summarization_model: str = "facebook/bart-large-cnn"
    ):
        """
        Advanced CodeRAG system with flexible embedding and reranking.
        
        Args:
            embedding_model (Union[str, object]): Embedding model or custom implementation
            reranker_model (Optional[str]): Reranking model name
            retrieval_strategy (str): Retrieval approach
        """
        # Flexible embedding model loading
        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model

        # Initialize reranker
        self.reranker = ListwiseReranker(
            model_name=reranker_model or "cross-encoder/ms-marco-MiniLM-L-12-v2"
        )

        # Configurable retrieval strategy
        self.retrieval_strategy = self._get_retrieval_strategy(retrieval_strategy)

        self.summarizer = pipeline(
            "summarization", 
            model=summarization_model,
            max_length=150,
            min_length=30,
            do_sample=False
        )

    def generate_summary(self, file_content: str, max_tokens: int = 300) -> str:
        """
        Generate concise, context-aware summary with dynamic length adjustment.
        
        Args:
            file_content (str): Full file content
            max_tokens (int): Maximum tokens for summary
        
        Returns:
            str: Generated summary
        """
        # Truncate extremely long content
        content = file_content[:max_tokens * 4]  # Rough token estimation
        
        try:
            # Very short content - return as-is
            if len(content) < 100:
                return content
            
            # Dynamically calculate summary length
            input_length = len(content)
            summary_length = min(
                max(30, int(input_length * 0.3)),  # 30% of input, but at least 30 tokens
                150  # Cap at 150 tokens
            )
            
            # Minimum length should be less than max_length
            min_length = max(10, int(summary_length * 0.5))
            
            # Ensure we don't exceed model's input constraints
            summaries = self.summarizer(
                content, 
                max_length=summary_length, 
                min_length=min_length, 
                do_sample=False
            )
            
            return summaries[0]['summary_text'] if summaries else content[:300] + "..."
        
        except Exception as e:
            print(f"Summary generation error: {e}")
            return content[:300] + "..."  # Fallback to truncated content

    def _get_retrieval_strategy(self, strategy: str):
        """
        Select retrieval strategy dynamically.
        
        Args:
            strategy (str): Retrieval strategy name
        
        Returns:
            Callable retrieval function
        """
        strategies = {
            'default': self._default_retrieval,
            'probabilistic': self._probabilistic_retrieval,
            'diverse': self._diverse_retrieval
        }
        return strategies.get(strategy, self._default_retrieval)

    def _default_retrieval(self, embeddings, query_embedding, top_k):
        """Default cosine similarity retrieval."""
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return top_indices

    def _probabilistic_retrieval(self, embeddings, query_embedding, top_k):
        """Probabilistic relevance retrieval."""
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        probabilities = np.exp(similarities) / np.sum(np.exp(similarities))
        top_indices = probabilities.argsort()[-top_k:][::-1]
        return top_indices

    def _diverse_retrieval(self, embeddings, query_embedding, top_k):
        """Diverse retrieval with MMR (Maximal Marginal Relevance)."""
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        lambda_param = 0.5
        
        selected_indices = []
        candidates = list(range(len(embeddings)))
        
        while len(selected_indices) < top_k and candidates:
            if not selected_indices:
                best_index = similarities.argmax()
            else:
                # Compute diversity score
                diversity_scores = [
                    lambda_param * similarities[idx] - 
                    (1 - lambda_param) * np.max([
                        np.dot(embeddings[idx], embeddings[selected].T)
                        for selected in selected_indices
                    ])
                    for idx in candidates
                ]
                best_index = candidates[np.argmax(diversity_scores)]
            
            selected_indices.append(best_index)
            candidates.remove(best_index)
        
        return selected_indices

    def advanced_retrieve(
        self, 
        query: str, 
        file_contents: Dict[str, str], 
        top_k: int = 10
    ) -> List[str]:
        """
        Advanced retrieval with embedding, reranking, and diverse strategies.
        
        Args:
            query (str): Search query
            file_contents (Dict[str, str]): File contents
            top_k (int): Number of top results
        
        Returns:
            List[str]: Retrieved file paths
        """
        # Embedding
        content_list = list(file_contents.values())
        file_paths = list(file_contents.keys())
        
        embeddings = self.embedding_model.encode(content_list)
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Retrieval
        top_indices = self.retrieval_strategy(
            embeddings, query_embedding, top_k
        )
        
        retrieved_contents = [content_list[idx] for idx in top_indices]
        retrieved_paths = [file_paths[idx] for idx in top_indices]
        
        # Reranking
        reranked_contents = self.reranker.rerank(query, retrieved_contents, top_k)
        
        # Match reranked contents back to paths
        reranked_paths = [
            path for content, path in zip(retrieved_contents, retrieved_paths)
            if content in reranked_contents
        ]
        
        return reranked_paths

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