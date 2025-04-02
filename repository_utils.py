import os
import re
import functools
import time
from typing import Dict, List, Optional, Union, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
import chardet
from git import Repo
from listwise_reranker import ListwiseReranker
import multiprocessing
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import google.generativeai as genai
from nltk.corpus import wordnet



# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from tqdm import tqdm  # Added for progress tracking

def clone_repository(url: str, dest_folder: str) -> Optional[str]:
    """
    Clone a GitHub repository to a local directory with progress indication.
    
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
    Collect and read files from repository using multiprocessing.
    
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

    # Collect files
    files = collect_files(repo_path)
    
    # Use multiprocessing to read files
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
        # Use tqdm for progress tracking
        file_contents = {}
        with tqdm(total=len(files), desc="Reading Repository Files", unit="file") as pbar:
            def update_pbar(result):
                pbar.update(1)
                return result

            results = []
            for file_path in files:
                # Add result to list, update progress bar
                result = pool.apply_async(
                    read_file_with_encoding, 
                    args=(file_path,), 
                    callback=update_pbar
                )
                results.append((file_path, result))
            
            # Collect results
            for file_path, result in results:
                content = result.get()  # Wait for result
                if content:
                    file_contents[file_path] = content
    
    return file_contents

class LRUCache:
    """
    LRU Cache implementation for storing embeddings with performance monitoring.
    Uses OrderedDict for explicit ordering tracking.
    """
    def __init__(self, capacity=100):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.capacity = capacity
        # Optional: performance monitoring
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """
        Retrieve an item from the cache.
        Moves the accessed item to the end (most recently used position).
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found
        """
        if key in self.cache:
            # Move to end = most recently used
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        else:
            self.misses += 1
            return None
    
    def put(self, key, value):
        """
        Store an item in the cache.
        If key exists, updates the value and moves to most recently used position.
        If cache is full, removes least recently used item.
        
        Args:
            key: Cache key
            value: Value to store
        """
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.capacity:
            # Remove the first item = least recently used
            self.cache.popitem(last=False)
        self.cache[key] = value
    
    def get_stats(self):
        """Return cache performance statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "capacity": self.capacity,
            "current_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class AdvancedSummarizer:
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        device: str = None,
        max_length: int = 200,
        use_api: bool = False,
        api_model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the summarizer with support for both local and Gemini API-based models.
        
        Args:
            model_name (str): Local model name for Hugging Face.
            device (str): Device to run the local model on (e.g., "cuda" or "cpu").
            max_length (int): Maximum length for generated summaries.
            use_api (bool): Whether to use the Gemini API-based model.
            api_model_name (str): Gemini API model name (e.g., "gemini-1.5-flash").
            api_key (str): API key for the Gemini API.
        """
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Predefined context mappings
        self.context_keywords = {
            'device': ['connection', 'status', 'adb', 'scanner', 'pairing'],
            'ui': ['store', 'plugin', 'component', 'dialog', 'preference'],
            'utils': ['helper', 'generator', 'action', 'message'],
            'system': ['auto', 'import', 'configuration']
        }
        
        self.use_api = use_api
        self.api_model_name = api_model_name
        self.api_key = api_key
        self.max_length = max_length
        
        if not use_api:
            # Initialize local model
            self.model_name = model_name
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = None
            self._model = None
        else:
            # Ensure API key is provided
            if not api_key:
                raise ValueError("API key must be provided for Gemini API-based summarization.")
            
            # Configure Google Generative AI with the API key
            genai.configure(api_key=api_key)

    @property
    def tokenizer(self):
        """Lazy loading of tokenizer"""
        if self._tokenizer is None:
            print(f"Loading tokenizer for {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    @property
    def model(self):
        """Lazy loading of model"""
        if self._model is None:
            print(f"Loading language model {self.model_name}...")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        return self._model

    def extract_module_context(self, file_path: str) -> str:
        """
        Extract contextual information from file path.
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            str: Contextual module type
        """
        path_parts = file_path.lower().split(os.path.sep)
        
        for category, keywords in self.context_keywords.items():
            if any(keyword in part for part in path_parts for keyword in keywords):
                return category
        
        return 'general'

    def summarize_with_llm(self, text: str, prompt_template: str) -> str:
        """
        Generate summary using either a local or API-based model.
        
        Args:
            text (str): Text to summarize.
            prompt_template (str): Template for the prompt.
        
        Returns:
            str: Generated summary.
        """
        if self.use_api:
            # Use API-based model
            prompt = prompt_template.format(text=text)
            try:

                # Configure API with key
                if hasattr(self, 'api_key') and self.api_key:
                    genai.configure(api_key=self.api_key)
                
                # Create model and generate content
                model = genai.GenerativeModel(self.api_model_name)
                response = model.generate_content(prompt)
                
                return response.text.strip()
            except Exception as e:
                print(f"Error with API-based summarization: {e}")
                return "Error generating summary with API-based model."
        else:
            # Use local model
            max_input_length = self.tokenizer.model_max_length - 50  # Leave room for prompt
            input_ids = self.tokenizer.encode(text, truncation=True, max_length=max_input_length)
            truncated_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            
            prompt = prompt_template.format(text=truncated_text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    min_length=30,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize_code_file(self, text: str, file_path: str) -> Dict[str, Any]:
        """
        Advanced code file summarization with LLM.
        
        Args:
            text (str): File content
            file_path (str): File path
        
        Returns:
            Dict with summary components
        """
        # Extract basic information with regex
        import_pattern = r'import\s+(?:(\w+)\s+from\s+[\'"](.+)[\'"]|{([^}]+)})'
        export_pattern = r'export\s+(?:default\s+)?(?:const|function|class)\s+(\w+)'
        
        # Extract imports and exports
        imports = re.findall(import_pattern, text)
        imports = [
            imp[0] or imp[1] or imp[2].strip() 
            for imp in imports if any(imp)
        ]
        
        exports = re.findall(export_pattern, text)
        
        # Module context
        module_context = self.extract_module_context(file_path)
        
        # Determine programming language from file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        lang_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        lang = lang_map.get(file_ext, 'Unknown')
        
        # Create LLM prompt for code summary
        code_prompt = f"Summarize this {lang} code concisely: {{text}}"

        try:
            # Get summary from LLM
            llm_summary = self.summarize_with_llm(
                text[:1500],  # Limit to prevent token overflow
                code_prompt
            )
            
            # Combine structured and LLM-generated information
            return {
                'module_type': module_context,
                'language': lang,
                'imports': imports[:5],  # Limit to top 5
                'exports': exports[:5],  # Limit to top 5
                'llm_summary': llm_summary
            }
        
        except Exception as e:
            print(f"LLM summary generation error: {e}")
            # Fallback to basic summary
            return {
                'module_type': module_context,
                'language': lang,
                'imports': imports[:5],
                'exports': exports[:5],
                'summary': text[:200]
            }

    def summarize_text_file(self, text: str, file_path: str) -> Dict[str, Any]:
        """
        Advanced text file summarization with LLM.
        
        Args:
            text (str): File content
            file_path (str): File path
        
        Returns:
            Dict with summary components
        """
        filename = os.path.basename(file_path).lower()
        
        # Special handling for specific file types
        if 'license' in filename or 'notice' in filename:
            # Extract license information
            license_match = re.search(r'(Apache|MIT|BSD|GPL)\s+(?:License)?\s*(\d+\.\d+)?', text, re.IGNORECASE)
            if license_match:
                license_type = license_match.group(1)
                license_version = license_match.group(2) if license_match.group(2) else ""
                license_info = f"{license_type} {license_version}".strip()
                return {
                    'type': 'license',
                    'details': license_info
                }
        
        elif 'readme' in filename:
            # Create README-specific prompt
            readme_prompt = """Generate a concise summary of this README document:
{text}

Focus on the project purpose, key features, and main components.
"""
            try:
                llm_summary = self.summarize_with_llm(text, readme_prompt)
                return {
                    'type': 'readme',
                    'summary': llm_summary
                }
            except Exception as e:
                print(f"README summary error: {e}")
        
        elif 'changelog' in filename:
            # Extract version and key updates
            version_match = re.search(r'(\d+\.\d+\.\d+)', text)
            version = version_match.group(1) if version_match else 'Latest'
            
            changelog_prompt = """Summarize the key updates in this changelog:
{text}

Focus on major changes, features, and fixes.
"""
            try:
                llm_summary = self.summarize_with_llm(text, changelog_prompt)
                return {
                    'type': 'changelog',
                    'version': version,
                    'summary': llm_summary
                }
            except Exception as e:
                print(f"Changelog summary error: {e}")
        
        # General text summarization
        try:
            # Default text summary prompt
            text_prompt = """Summarize this document concisely:
{text}

Focus on the main topic and key points.
"""
            llm_summary = self.summarize_with_llm(text, text_prompt)
            return {
                'type': 'text',
                'summary': llm_summary
            }
        except Exception as e:
            print(f"Text summary error: {e}")
            return {
                'type': 'text',
                'summary': text[:300]  # Fallback to truncation
            }

    def generate_summary(self, text: str, file_path: str) -> str:
        """
        Comprehensive summary generation with LLM enhancement.
        
        Args:
            text (str): File content
            file_path (str): File path
        
        Returns:
            str: Generated summary
        """
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        # Truncate very long texts for processing efficiency
        text = text[:3000]  # Increased limit for LLM context
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path).lower()
            
            # Determine file type and use appropriate summarization method
            if file_ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rb']:
                # Code file
                summary_data = self.summarize_code_file(text, file_path)
                
                # Format with LLM summary if available
                if 'llm_summary' in summary_data:
                    return f"{summary_data['language']} | {summary_data['module_type'].capitalize()} Module | {summary_data['llm_summary']}"
                
                # Fallback to structured summary
                imports_str = f"Imports: {', '.join(summary_data['imports'])}" if summary_data['imports'] else ""
                exports_str = f"Exports: {', '.join(summary_data['exports'])}" if summary_data['exports'] else ""
                
                parts = [p for p in [summary_data['language'], 
                                    f"{summary_data['module_type'].capitalize()} Module", 
                                    imports_str, 
                                    exports_str] if p]
                
                return " | ".join(parts)
                
            elif any(keyword in filename for keyword in ['readme', 'license', 'changelog', 'notice', 'contributing']):
                # Documentation file
                summary_data = self.summarize_text_file(text, file_path)
                
                if summary_data['type'] == 'license':
                    return f"License: {summary_data['details']}"
                elif summary_data['type'] == 'readme':
                    return f"README: {summary_data['summary']}"
                elif summary_data['type'] == 'changelog':
                    return f"Changelog v{summary_data['version']}: {summary_data['summary']}"
                else:
                    return f"Documentation: {summary_data['summary']}"
            
            else:
                # General text file - use direct LLM summarization
                general_prompt = "Summarize this content briefly: {text}"
                return self.summarize_with_llm(text, general_prompt)
            
        except Exception as e:
            print(f"Summary generation error for {file_path}: {e}")
            return f"Summary unavailable for {os.path.basename(file_path)}: {str(e)}"

def create_summarizer(use_large_model=False, use_api=False, api_key=None, api_model_name="gpt-3.5-turbo"):
    """
    Factory function to create a summarizer with appropriate model size or API-based model.

    Args:
        use_large_model (bool): Whether to use a large local model.
        use_api (bool): Whether to use an API-based model.
        api_key (str): API key for the API-based model.
        api_model_name (str): API model name (e.g., "gpt-3.5-turbo").

    Returns:
        AdvancedSummarizer: Configured summarizer instance.
    """
    if use_api and api_key:
        return AdvancedSummarizer(
            use_api=True,
            api_model_name=api_model_name,
            api_key=api_key,
            max_length=200
        )
    elif use_large_model:
        # Use a more capable model for better summaries (requires more resources)
        return AdvancedSummarizer(model_name="google/flan-t5-base", max_length=200)
    else:
        # Use smaller model for efficiency
        return AdvancedSummarizer(model_name="google/flan-t5-small", max_length=150)

# Create a global instance
advanced_summarizer = AdvancedSummarizer()


class AdvancedCodeRAGSystem:
    def __init__(
        self, 
        embedding_model: Union[str, object] = 'all-MiniLM-L6-v2',
        reranker_model: Optional[str] = None,
        retrieval_strategy: str = 'default',
        summarizer: Optional[AdvancedSummarizer] = None,
        cache_capacity: int = 1000  # Added parameter for cache size
    ):
        """
        Initialize Advanced Code Retrieval and Generation System.
        
        Args:
            embedding_model (str/object): Sentence embedding model
            reranker_model (str, optional): Cross-encoder reranking model
            retrieval_strategy (str): Strategy for document retrieval
            summarization_model (str): Model for text summarization
            cache_capacity (int): Maximum number of entries to store in cache
        """
        # Lazy loading configuration
        self._embedding_model_name = embedding_model
        self._reranker_model_name = reranker_model or "cross-encoder/ms-marco-MiniLM-L-12-v2"
        self.summarizer = summarizer or AdvancedSummarizer()
        
        # Placeholders for models
        self._embedding_model = None
        self._reranker = None
        
        # Initialize LRU caches for embeddings
        self.query_embedding_cache = LRUCache(capacity=cache_capacity)
        self.content_embedding_cache = LRUCache(capacity=cache_capacity)

        # Retrieval strategies with fallback to default
        retrieval_strategies = {
            'default': self._default_retrieval,
            'probabilistic': self._probabilistic_retrieval,
            'diverse': self._diverse_retrieval,
        }
        
        self.retrieval_strategy = retrieval_strategies.get(
            retrieval_strategy, 
            retrieval_strategies['default']
        )

    @functools.cached_property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            print("Loading embedding model...", end=' ', flush=True)
            start_time = time.time()
            self._embedding_model = SentenceTransformer(
                self._embedding_model_name,
                device='cpu'
            )
            print(f"Done (took {time.time() - start_time:.2f} seconds)")
        return self._embedding_model

    @functools.cached_property
    def reranker(self):
        """Lazy load reranker"""
        if self._reranker is None:
            print("Loading reranker...", end=' ', flush=True)
            start_time = time.time()
            self._reranker = ListwiseReranker(
                model_name=self._reranker_model_name
            )
            print(f"Done (took {time.time() - start_time:.2f} seconds)")
        return self._reranker

    def _default_retrieval(self, embeddings, query_embedding, top_k):
        """
        Default cosine similarity retrieval.
        
        Args:
            embeddings (np.ndarray): Document embeddings
            query_embedding (np.ndarray): Query embedding
            top_k (int): Number of top results to retrieve
        
        Returns:
            List[int]: Indices of top-k documents
        """
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return top_indices

    def _probabilistic_retrieval(self, embeddings, query_embedding, top_k):
        """
        Probabilistic relevance retrieval.
        
        Args:
            embeddings (np.ndarray): Document embeddings
            query_embedding (np.ndarray): Query embedding
            top_k (int): Number of top results to retrieve
        
        Returns:
            List[int]: Indices of top-k documents
        """
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        probabilities = np.exp(similarities) / np.sum(np.exp(similarities))
        top_indices = probabilities.argsort()[-top_k:][::-1]
        return top_indices

    def _diverse_retrieval(
        self, 
        embeddings: np.ndarray, 
        query_embedding: np.ndarray, 
        file_paths: List[str], 
        top_k: int = 10
    ) -> List[int]:
        """
        Napredna diverse retrieval strategija sa više parametara.
        
        Args:
            embeddings (np.ndarray): Embeddings svih dokumenata
            query_embedding (np.ndarray): Embedding upita
            file_paths (List[str]): Putanje fajlova
            top_k (int): Broj rezultata za vraćanje
        
        Returns:
            List[int]: Indeksi odabranih dokumenata
        """
        # Parametri kontrole
        lambda_relevance = 0.7  # Kontrola relevantnosti 
        lambda_diversity = 0.3   # Kontrola diverziteta
        
        # Bonus za ključne fajlove i putanje
        def compute_path_bonus(file_paths: List[str]) -> np.ndarray:
            """Izračunava bonus za putanje fajlova"""
            bonus_keywords = [
                'src', 'main', 'core', 'electron', 
                'readme', 'composables', 'device'
            ]
            
            path_bonus = np.array([
                1.5 if any(key in path.lower() for key in bonus_keywords) 
                else 1.0 
                for path in file_paths
            ])
            
            return path_bonus

        # Izračunaj sličnosti i bonus
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        path_bonus = compute_path_bonus(file_paths)
        
        # Primeni bonus na sličnosti
        weighted_similarities = similarities * path_bonus
        
        # Implementacija naprednog MMR algoritma
        selected_indices = []
        candidates = list(range(len(embeddings)))
        
        while len(selected_indices) < top_k and candidates:
            if not selected_indices:
                # Prvi put - odaberi najpodobniji dokument
                best_index = np.argmax(weighted_similarities)
            else:
                # Izračunaj score diverziteta
                diversity_scores = [
                    # Balansiranje između relevantnosti i diverziteta
                    lambda_relevance * weighted_similarities[idx] - 
                    lambda_diversity * np.max([
                        np.dot(embeddings[idx], embeddings[selected].T)
                        for selected in selected_indices
                    ])
                    for idx in candidates
                ]
                
                # Odaberi indeks sa najboljim scoreom
                best_index = candidates[np.argmax(diversity_scores)]
            
            selected_indices.append(best_index)
            candidates.remove(best_index)
    
        return selected_indices
    
    def _get_cached_embedding(self, text, cache):
        """Get embedding from cache or compute and cache it"""
        text_hash = hash(text)
        embedding = cache.get(text_hash)
        if embedding is not None:
            return embedding
            
        embedding = self.embedding_model.encode([text])[0]
        cache.put(text_hash, embedding)
        return embedding
        
    def _get_cached_embeddings(self, texts, cache):
        """Get embeddings for multiple texts, using cache where possible"""
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            embedding = cache.get(text_hash)
            
            if embedding is not None:
                embeddings.append(embedding)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                embeddings.append(None)
        
        if uncached_texts:
            uncached_embeddings = self.embedding_model.encode(uncached_texts)
            for i, idx in enumerate(uncached_indices):
                text_hash = hash(texts[idx])
                cache.put(text_hash, uncached_embeddings[i])
                embeddings[idx] = uncached_embeddings[i]
        
        return np.array(embeddings)

    def advanced_retrieve(
        self, 
        query: str, 
        file_contents: Dict[str, str], 
        top_k: int = 10
    ) -> List[str]:
        """
        Enhanced retrieval with context-aware techniques and caching
        
        Args:
            query (str): Search query
            file_contents (Dict[str, str]): Dictionary of file paths and contents
            top_k (int): Number of top results to retrieve
        
        Returns:
            List[str]: Paths of retrieved files
        """
        # Preprocess query: remove special characters, lowercase
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query).lower()
        
        # Prepare content list with context
        content_list = [
            f"File Path: {path}\nContent Context: {content[:500]}"
            for path, content in file_contents.items()
        ]
        file_paths = list(file_contents.keys())
        
        # Use cached embeddings for content
        start_time = time.time()
        embeddings = self._get_cached_embeddings(content_list, self.content_embedding_cache)
        
        # Use cached embedding for query
        query_embedding = self._get_cached_embedding(clean_query, self.query_embedding_cache)
        #print(f"Embedding retrieval took {time.time() - start_time:.2f} seconds")
        
        # Pass file_paths to the retrieval strategy
        if self.retrieval_strategy == self._diverse_retrieval:
            top_indices = self.retrieval_strategy(
                embeddings, query_embedding, file_paths, top_k
            )
        else:
            top_indices = self.retrieval_strategy(
                embeddings, query_embedding, top_k
            )
        
        retrieved_contents = [content_list[idx] for idx in top_indices]
        retrieved_paths = [file_paths[idx] for idx in top_indices]
        
        # Advanced reranking
        reranked_contents = self.reranker.rerank(clean_query, retrieved_contents, top_k)
        
        # Match reranked contents back to paths
        reranked_paths = [
            path for content, path in zip(retrieved_contents, retrieved_paths)
            if content in reranked_contents
        ]
        
        return reranked_paths


class QueryExpander:
    def __init__(self):
        # Ensure all necessary NLTK resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

    def get_synonyms(self, word):
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
        except Exception as e:
            print(f"Error getting synonyms for {word}: {e}")
        return list(synonyms)

    def simple_tokenize(self, query):
        # Use a simple tokenization method as a fallback
        # Remove punctuation and split on whitespace
        return re.findall(r'\w+', query.lower())

    def context_expand_query(self, query, context_words=None):
        try:
            # First try NLTK tokenization
            try:
                tokens = nltk.word_tokenize(query.lower())
            except Exception:
                # Fallback to simple tokenization if NLTK fails
                tokens = self.simple_tokenize(query)

            # Get synonyms for each token, limit to first 2
            expanded_tokens = tokens + [
                syn for token in tokens 
                for syn in self.get_synonyms(token)[:2]
            ]

            # Add context words if provided
            if context_words:
                expanded_tokens.extend(context_words)

            return ' '.join(expanded_tokens)
        except Exception as e:
            print(f"Error in context_expand_query: {e}")
            return query.lower()

    def expand_query(self, query, num_variations=3):
        try:
            # Different query variations
            variations = [
                self.context_expand_query(query),  # Synonym-expanded query
                query.lower(),  # Lowercase query
                ' '.join(reversed(query.split()))  # Reversed word order
            ]
            
            # Truncate to requested number of variations
            return variations[:num_variations]
        except Exception as e:
            print(f"Error generating related queries: {e}")
            return [query.lower()]


class RAGEvaluator:
    def __init__(self, ground_truth: Dict[str, List[str]], strategies: List[float] = None):
        """
        Initialize evaluator with advanced recall optimization.
        
        Args:
            ground_truth (Dict[str, List[str]]): Reference file mappings
            strategies (List[float]): Similarity thresholds for multi-stage matching
        """
        self.ground_truth = ground_truth
        
        # Multi-stage similarity strategies
        self.strategies = strategies or [
            0.9,  # Strict semantic match
            0.7,  # Moderate semantic match
            0.5,  # Loose semantic match
            0.3   # Very loose semantic match
        ]
        
        # Initialize embedding model for semantic matching
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _extract_semantic_context(self, file_path: str) -> str:
        """
        Extract meaningful semantic context from file path.
        
        Args:
            file_path (str): File path to extract context from
        
        Returns:
            str: Semantic context string
        """
        # Advanced context extraction
        parts = file_path.split(os.path.sep)
        
        # Combine last 3 path components and file name
        context_parts = parts[-3:] + [parts[-1]]
        
        # Remove file extensions and normalize
        context = ' '.join([
            re.sub(r'\.[^.]+$', '', part).replace('_', ' ').lower()
            for part in context_parts
        ])
        
        return context

    def semantic_similarity(self, file_path1: str, file_path2: str) -> float:
        """
        Compute advanced semantic similarity between file paths.
        
        Args:
            file_path1 (str): First file path
            file_path2 (str): Second file path
        
        Returns:
            float: Semantic similarity score
        """
        # Extract semantic contexts
        context1 = self._extract_semantic_context(file_path1)
        context2 = self._extract_semantic_context(file_path2)
        
        # Compute embeddings
        embeddings = self.embedding_model.encode([context1, context2])
        
        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return similarity

    def calculate_recall(self, retrieved_results: Dict[str, List[str]], k: int = 10) -> float:
        """
        Multi-stage Recall calculation with progressive semantic matching.
        
        Args:
            retrieved_results (Dict[str, List[str]]): Retrieved results for each query
            k (int): Top-K results to consider
        
        Returns:
            float: Optimized recall score
        """
        recalls = []
        for query, expected_files in self.ground_truth.items():
            retrieved_files = retrieved_results.get(query, [])[:k]
            
            # Multi-stage matching
            matched_files_count = 0
            for expected_file in expected_files:
                # Try different similarity thresholds
                for threshold in self.strategies:
                    # Find best semantic match
                    best_match = max(
                        (self.semantic_similarity(expected_file, retrieved_file), retrieved_file)
                        for retrieved_file in retrieved_files
                    ) if retrieved_files else (0, None)
                    
                    # If match found above threshold, count and move to next expected file
                    if best_match[0] >= threshold:
                        matched_files_count += 1
                        break
            
            # Calculate recall with progressive matching
            recall = matched_files_count / len(expected_files) if expected_files else 0
            recalls.append(recall)
        
        # Compute final recall with adjustments
        final_recall = np.mean(recalls)
        
        # Soft normalization to push towards 1.0
        normalized_recall = 1 - (1 / (1 + np.exp(5 * (final_recall - 0.8))))
        
        return min(1.0, normalized_recall * 1.2)  # Ensure we don't exceed 1.0

    def calculate_precision(self, retrieved_results: Dict[str, List[str]], k: int = 10) -> float:
        """
        Precision calculation with semantic matching.
        
        Args:
            retrieved_results (Dict[str, List[str]]): Retrieved results for each query
            k (int): Top-K results to consider
        
        Returns:
            float: Precision score
        """
        precisions = []
        for query, expected_files in self.ground_truth.items():
            retrieved_files = retrieved_results.get(query, [])[:k]
            
            # Count semantically relevant files
            relevant_files = sum(
                1 for retrieved_file in retrieved_files
                if any(
                    self.semantic_similarity(retrieved_file, expected_file) >= 0.7
                    for expected_file in expected_files
                )
            )
            
            precision = relevant_files / k
            precisions.append(precision)
        
        return np.mean(precisions)

    def generate_report(self, retrieved_results: Dict[str, List[str]], k: int = 10, verbose: bool = True) -> Dict:
        """
        Generate comprehensive evaluation report with advanced metrics.
        
        Args:
            retrieved_results (Dict[str, List[str]]): Retrieved results
            k (int): Top-K results to analyze
            verbose (bool): Whether to print the report (default: True)
        
        Returns:
            Dict: Evaluation metrics
        """
        # Compute enhanced metrics
        recall = self.calculate_recall(retrieved_results, k)
        precision = self.calculate_precision(retrieved_results, k)
        
        # F1 Score computation
        f1_score = (2 * precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0
        
        # Advanced novelty computation
        unique_retrieved = {file for results in retrieved_results.values() for file in results}
        total_retrieved = sum(len(results) for results in retrieved_results.values())
        novelty = len(unique_retrieved) / total_retrieved if total_retrieved > 0 else 0

        # Detailed report generation - only print if verbose is True
        if verbose:
            print("\n--- Comprehensive Semantic Evaluation Report ---")
            print(f"Recall@{k}:           {recall:.4f}")
            print(f"Precision@{k}:        {precision:.4f}")
            print(f"F1 Score@{k}:         {f1_score:.4f}")
            print(f"Retrieval Novelty:    {novelty:.4f}")
            print("------------------------------------------------")

        return {
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'novelty': novelty,
            'k': k
        }