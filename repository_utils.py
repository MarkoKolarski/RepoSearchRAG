# Standard Library Imports
import os
import re
import time
import functools
import multiprocessing
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import OrderedDict
from argparse import Namespace
import nltk
import chardet
import numpy as np
import torch
from git import Repo
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import google.generativeai as genai
from listwise_reranker import ListwiseReranker

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def clone_repository(url: str, dest_folder: str) -> Optional[str]:
    """
    Clones a GitHub repository to a local directory if it doesn't already exist.

    Args:
        url (str): URL of the GitHub repository.
        dest_folder (str): Local destination folder where the repo will be cloned.

    Returns:
        Optional[str]: The path to the cloned repository, or None if cloning failed.
    """
    if os.path.exists(dest_folder):
        print(f"Folder '{dest_folder}' already exists. Skipping cloning.")
        return dest_folder

    try:
        print(f"Cloning repository from {url}...")
        Repo.clone_from(url, dest_folder)
        print("[SUCCESS] Cloning completed successfully.")
        return dest_folder
    except Exception as e:
        print(f"[ERROR] Failed to clone repository: {e}")
        return None

def read_file_with_encoding(file_path: str) -> Optional[str]:
    """
    Reads a file with automatic encoding detection. Falls back to common encodings if detection fails.

    Args:
        file_path (str): Path to the file.

    Returns:
        Optional[str]: File contents as a string, or None if reading fails.
    """
    try:
        # Try detecting encoding from raw bytes
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            detected_encoding = chardet.detect(raw_data)['encoding']

        # Try reading with detected encoding
        with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
            return f.read()

    except Exception:
        # Fallback to common encodings
        fallback_encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        for encoding in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    return f.read()
            except Exception:
                continue

        print(f"[ERROR] Failed to read file: {file_path} with all encoding attempts.")
        return None


def prepare_repository_files(repo_path: str) -> Dict[str, str]:
    """
    Collects and reads allowed files from a repository directory using multiprocessing.

    Args:
        repo_path (str): Path to the repository directory.

    Returns:
        Dict[str, str]: Dictionary mapping file paths to their contents.
    """
    allowed_extensions = ['.py', '.js', '.md', '.txt', '.json']

    def collect_files(directory: str) -> List[str]:
        """
        Recursively collects files with allowed extensions.

        Args:
            directory (str): Root directory.

        Returns:
            List[str]: List of file paths.
        """
        file_paths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                full_path = os.path.join(root, filename)
                if any(full_path.endswith(ext) for ext in allowed_extensions):
                    file_paths.append(full_path)
        return file_paths

    # Step 1: Collect all relevant files
    files = collect_files(repo_path)

    # Step 2: Read all files in parallel using multiprocessing
    file_contents = {}
    start_time = time.time()
    print("Reading repository files...", end="", flush=True)

    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
        results = [
            (file_path, pool.apply_async(read_file_with_encoding, args=(file_path,)))
            for file_path in files
        ]

        # Step 3: Collect results
        for file_path, async_result in results:
            content = async_result.get()
            if content:
                file_contents[file_path] = content

    end_time = time.time()
    elapsed = end_time - start_time
    print(f" Done (took {elapsed:.2f} seconds)")

    return file_contents

def read_file_content(file_path: str) -> str:
    """
    Safely read file content with error handling.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"[File Error] Failed to read {file_path}: {e}")
        return ""


def initialize_summarizer(args: Namespace):
    """
    Create a summarizer instance based on CLI arguments.
    """
    if not args.generate_summaries:
        return None

    api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")

    if args.use_gemini:
        if not api_key:
            print(
                "Error: Google Gemini API key must be provided via "
                "--gemini_api_key or GOOGLE_API_KEY environment variable."
            )
            return None

        print(f"Using Google Gemini API with model: {args.gemini_model}")
        return create_summarizer(
            use_api=True,
            api_key=api_key,
            api_model_name=args.gemini_model
        )

    model_type = "large" if args.use_large_summarizer else "default small"
    print(f"Using {model_type} summarization model...")

    return create_summarizer(use_large_model=args.use_large_summarizer)


def interactive_query_loop(rag_system, repo_files: list, args: Namespace):
    """
    Continuously prompt user for queries and return relevant files.
    """
    query_expander = QueryExpander()

    print("\nEnter your questions about the codebase (type 'exit' to quit):\n")
    first_iteration = True
    while True:
        user_query = input("Your question: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not first_iteration:
            print("Processing queries... please wait.")
        first_iteration = False

        expanded_queries = query_expander.expand_query(user_query)
        all_results = []

        for expanded_query in expanded_queries:
            results = rag_system.advanced_retrieve(
                expanded_query,
                repo_files,
                args.top_k
            )
            all_results.extend(results)

        # Deduplicate results and keep top_k
        unique_results = list(dict.fromkeys(all_results))[:args.top_k]

        print(f"\nTop {args.top_k} relevant files for: \"{user_query}\"")
        for file_path in unique_results:
            print(f"• {file_path}")

            content = read_file_content(file_path)

            if args.generate_summaries and rag_system.summarizer:
                summary = rag_system.summarizer.generate_summary(content, file_path)
                print(f"Summary: {summary}\n")
            else:
                print()

        print("-" * 60)

class LRUCache:
    """
    Least Recently Used (LRU) Cache implementation using OrderedDict.
    Tracks cache hits and misses for performance monitoring.
    """

    def __init__(self, capacity: int = 100):
        """
        Initialize the LRU Cache.

        Args:
            capacity (int): Maximum number of items the cache can hold.
        """
        self.cache = OrderedDict()
        self.capacity = capacity

        # Performance monitoring
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """
        Retrieve a value from the cache by key.

        Moves the accessed item to the end to mark it as recently used.

        Args:
            key: The key to look up in the cache.

        Returns:
            The cached value if found, otherwise None.
        """
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value

        self.misses += 1
        return None

    def put(self, key, value):
        """
        Add or update a value in the cache.

        If the cache is at capacity, evicts the least recently used item.

        Args:
            key: The key to store.
            value: The value to associate with the key.
        """
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)

        self.cache[key] = value


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
            return self._summarize_with_api(text, prompt_template)
        else:
            return self._summarize_with_local_model(text, prompt_template)
    
    def _summarize_with_api(self, text: str, prompt_template: str) -> str:
        """Generate summary using API-based model."""
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
    
    def _summarize_with_local_model(self, text: str, prompt_template: str) -> str:
        """Generate summary using local model."""
        max_input_length = self.tokenizer.model_max_length - 50
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
        imports = self._extract_imports(text)
        exports = self._extract_exports(text)
        
        # Module context
        module_context = self.extract_module_context(file_path)
        
        # Determine programming language from file extension
        lang = self._determine_language(file_path)
        
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
                'imports': imports[:5],
                'exports': exports[:5],
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
    
    def _extract_imports(self, text: str) -> list:
        """Extract imports from code."""
        import_pattern = r'import\s+(?:(\w+)\s+from\s+[\'"](.+)[\'"]|{([^}]+)})'
        imports = re.findall(import_pattern, text)
        return [
            imp[0] or imp[1] or imp[2].strip() 
            for imp in imports if any(imp)
        ]
    
    def _extract_exports(self, text: str) -> list:
        """Extract exports from code."""
        export_pattern = r'export\s+(?:default\s+)?(?:const|function|class)\s+(\w+)'
        return re.findall(export_pattern, text)
    
    def _determine_language(self, file_path: str) -> str:
        """Determine programming language from file extension."""
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
        return lang_map.get(file_ext, 'Unknown')

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
            return self._summarize_license(text)
        elif 'readme' in filename:
            return self._summarize_readme(text)
        elif 'changelog' in filename:
            return self._summarize_changelog(text)
        else:
            return self._summarize_general_text(text)
    
    def _summarize_license(self, text: str) -> Dict[str, Any]:
        """Extract and summarize license information."""
        license_match = re.search(r'(Apache|MIT|BSD|GPL)\s+(?:License)?\s*(\d+\.\d+)?', text, re.IGNORECASE)
        if license_match:
            license_type = license_match.group(1)
            license_version = license_match.group(2) if license_match.group(2) else ""
            license_info = f"{license_type} {license_version}".strip()
            return {
                'type': 'license',
                'details': license_info
            }
        return {
            'type': 'license',
            'details': 'Unknown License'
        }
    
    def _summarize_readme(self, text: str) -> Dict[str, Any]:
        """Summarize README file."""
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
            return {
                'type': 'readme',
                'summary': text[:300]
            }
    
    def _summarize_changelog(self, text: str) -> Dict[str, Any]:
        """Summarize changelog file."""
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
            return {
                'type': 'changelog',
                'version': version,
                'summary': text[:300]
            }
    
    def _summarize_general_text(self, text: str) -> Dict[str, Any]:
        """Summarize general text file."""
        try:
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
                'summary': text[:300]
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
        text = text[:3000]
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path).lower()
            
            # Determine file type and use appropriate summarization method
            if self._is_code_file(file_ext):
                return self._format_code_summary(self.summarize_code_file(text, file_path))
            elif self._is_documentation_file(filename):
                return self._format_doc_summary(self.summarize_text_file(text, file_path))
            else:
                # General text file - use direct LLM summarization
                general_prompt = "Summarize this content briefly: {text}"
                return self.summarize_with_llm(text, general_prompt)
            
        except Exception as e:
            print(f"Summary generation error for {file_path}: {e}")
            return f"Summary unavailable for {os.path.basename(file_path)}: {str(e)}"
    
    def _is_code_file(self, file_ext: str) -> bool:
        """Check if the file is a code file based on extension."""
        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rb']
        return file_ext in code_extensions
    
    def _is_documentation_file(self, filename: str) -> bool:
        """Check if the file is a documentation file based on name."""
        doc_keywords = ['readme', 'license', 'changelog', 'notice', 'contributing']
        return any(keyword in filename for keyword in doc_keywords)
    
    def _format_code_summary(self, summary_data: Dict[str, Any]) -> str:
        """Format code summary into readable string."""
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
    
    def _format_doc_summary(self, summary_data: Dict[str, Any]) -> str:
        """Format documentation summary into readable string."""
        if summary_data['type'] == 'license':
            return f"License: {summary_data['details']}"
        elif summary_data['type'] == 'readme':
            return f"README: {summary_data['summary']}"
        elif summary_data['type'] == 'changelog':
            return f"Changelog v{summary_data['version']}: {summary_data['summary']}"
        else:
            return f"Documentation: {summary_data['summary']}"

def create_summarizer(
    use_large_model: bool = False,
    use_api: bool = False,
    api_key: str = None,
    api_model_name: str = "google/flan-t5-small"
):
    """
    Factory function to create a summarizer instance based on configuration.

    Args:
        use_large_model (bool): If True, use a larger local model for better quality.
        use_api (bool): If True, use an API-based model.
        api_key (str): API key required for using the API-based model.
        api_model_name (str): Name of the API model to use.

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

    if use_large_model:
        return AdvancedSummarizer(
            model_name="google/flan-t5-base",
            max_length=200
        )

    return AdvancedSummarizer(
        model_name="google/flan-t5-small",
        max_length=150
    )


class AdvancedCodeRAGSystem:
    def __init__(
        self, 
        embedding_model: Union[str, object] = 'all-MiniLM-L6-v2',
        reranker_model: Optional[str] = None,
        retrieval_strategy: str = 'default',
        summarizer: Optional[AdvancedSummarizer] = None,
        cache_capacity: int = 1000
    ):
        """
        Initialize Advanced Code Retrieval and Generation System.
        
        Args:
            embedding_model (str/object): Sentence embedding model
            reranker_model (str, optional): Cross-encoder reranking model
            retrieval_strategy (str): Strategy for document retrieval
            summarizer (AdvancedSummarizer, optional): Summarizer instance
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

        self._setup_retrieval_strategy(retrieval_strategy)

        self._processing_message_shown = False

    def _setup_retrieval_strategy(self, retrieval_strategy: str) -> None:
        """
        Set up the retrieval strategy based on the provided name.
        
        Args:
            retrieval_strategy (str): Name of the retrieval strategy
        """
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

    def _default_retrieval(self, embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int) -> List[int]:
        """
        Default cosine similarity retrieval.
        
        Args:
            embeddings: Document embeddings
            query_embedding: Query embedding
            top_k: Number of top results to retrieve
        
        Returns:
            Indices of top-k documents
        """
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return top_indices

    def _probabilistic_retrieval(self, embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int) -> List[int]:
        """
        Probabilistic relevance retrieval.
        
        Args:
            embeddings: Document embeddings
            query_embedding: Query embedding
            top_k: Number of top results to retrieve
        
        Returns:
            Indices of top-k documents
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
        top_k: int = 15
    ) -> List[int]:
        """
        Optimized diverse retrieval strategy with refined parameters.
        
        Args:
            embeddings: Embeddings of all documents
            query_embedding: Embedding of query
            file_paths: File paths
            top_k: Number of results to return
        
        Returns:
            List of selected document indices
        """
        # Control parameters
        lambda_relevance = 0.75
        lambda_diversity = 0.25
        
        # Calculate similarities and apply bonus
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        path_bonus = self._compute_path_bonus(file_paths)
        weighted_similarities = similarities * path_bonus
        
        return self._apply_mmr_selection(
            embeddings, 
            weighted_similarities, 
            lambda_relevance, 
            lambda_diversity, 
            top_k
        )
    
    def _compute_path_bonus(self, file_paths: List[str]) -> np.ndarray:
        """
        Compute bonus for file paths with enhanced weighting.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Array of bonus values for each path
        """
        bonus_keywords = [
            'src', 'main', 'core', 'electron', 'readme', 'composables', 'device',
            'utils', 'helper', 'component', 'api', 'service', 'controller', 'model',
            'interface', 'type', 'const', 'config', 'index', 'hook'
        ]
        
        language_bonus = {
            'py': 1.2, 'js': 1.2, 'ts': 1.25, 'jsx': 1.2, 'tsx': 1.2,
            'vue': 1.2, 'go': 1.1, 'java': 1.1, 'rb': 1.1, 'php': 1.1
        }
        
        path_bonuses = []
        for path in file_paths:
            # Start with base bonus
            bonus = 1.0
            
            # Add keyword bonus
            if any(key in path.lower() for key in bonus_keywords):
                bonus *= 1.6
            
            # Add language/extension bonus
            ext = path.split('.')[-1] if '.' in path else ''
            if ext in language_bonus:
                bonus *= language_bonus[ext]
                
            # Add bonus for directory depth
            depth = path.count('/') + path.count('\\')
            if depth <= 2:
                bonus *= 1.2
                
            path_bonuses.append(bonus)
            
        return np.array(path_bonuses)
    
    def _apply_mmr_selection(
        self, 
        embeddings: np.ndarray, 
        weighted_similarities: np.ndarray, 
        lambda_relevance: float, 
        lambda_diversity: float, 
        top_k: int
    ) -> List[int]:
        """
        Apply Maximal Marginal Relevance for diverse document selection.
        
        Args:
            embeddings: Document embeddings
            weighted_similarities: Similarity scores with applied weights
            lambda_relevance: Weight for relevance component
            lambda_diversity: Weight for diversity component
            top_k: Number of documents to select
            
        Returns:
            List of selected document indices
        """
        selected_indices = []
        candidates = list(range(len(embeddings)))
        
        while len(selected_indices) < top_k and candidates:
            if not selected_indices:
                # First document - select highest relevance
                best_index = np.argmax(weighted_similarities)
            else:
                # Calculate MMR scores with dynamic lambda adjustment
                # As we select more documents, gradually favor diversity more
                current_lambda_rel = lambda_relevance * (1 - (len(selected_indices) / (top_k * 2)))
                current_lambda_div = lambda_diversity * (1 + (len(selected_indices) / (top_k * 2)))
                
                mmr_scores = []
                for idx in candidates:
                    # Relevance component
                    relevance = weighted_similarities[idx]
                    
                    # Diversity component - maximum similarity to any selected document
                    max_similarity = max([
                        np.dot(embeddings[idx], embeddings[selected].T)
                        for selected in selected_indices
                    ])
                    
                    # Calculate MMR score
                    mmr_score = current_lambda_rel * relevance - current_lambda_div * max_similarity
                    mmr_scores.append(mmr_score)
                    
                # Select best index
                best_index = candidates[np.argmax(mmr_scores)]
                
            selected_indices.append(best_index)
            candidates.remove(best_index)
        
        return selected_indices
    
    def _get_cached_embedding(self, text: str, cache: LRUCache) -> np.ndarray:
        """
        Get embedding from cache or compute and cache it.
        
        Args:
            text: Text to embed
            cache: Cache object
            
        Returns:
            Embedding vector
        """
        text_hash = hash(text)
        embedding = cache.get(text_hash)
        if embedding is not None:
            return embedding
            
        embedding = self.embedding_model.encode([text])[0]
        cache.put(text_hash, embedding)
        return embedding
        
    def _get_cached_embeddings(self, texts: List[str], cache: LRUCache) -> np.ndarray:
        """
        Get embeddings for multiple texts, using cache where possible.
        
        Args:
            texts: List of texts to embed
            cache: Cache object
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = hash(text)
            embedding = cache.get(text_hash)
            
            if embedding is not None:
                embeddings.append(embedding)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                embeddings.append(None)
        
        # Calculate embeddings for uncached texts
        if uncached_texts:
            uncached_embeddings = self.embedding_model.encode(uncached_texts)
            for i, idx in enumerate(uncached_indices):
                text_hash = hash(texts[idx])
                cache.put(text_hash, uncached_embeddings[i])
                embeddings[idx] = uncached_embeddings[i]
        
        return np.array(embeddings)

    def _prepare_contents(self, file_contents: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """
        Prepare content list with context and extract file paths.
        
        Args:
            file_contents: Dictionary of file paths and contents
            
        Returns:
            Tuple of (content_list, file_paths)
        """
        content_list = [
            f"File Path: {path}\nContent Context: {content[:500]}"
            for path, content in file_contents.items()
        ]
        file_paths = list(file_contents.keys())
        return content_list, file_paths

    def advanced_retrieve(
        self, 
        query: str, 
        file_contents: Dict[str, str], 
        top_k: int = 10
    ) -> List[str]:
        """
        Enhanced retrieval with context-aware techniques and caching.
        
        Args:
            query: Search query
            file_contents: Dictionary of file paths and contents
            top_k: Number of top results to retrieve
        
        Returns:
            Paths of retrieved files
        """
        # Preprocess query: remove special characters, lowercase
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query).lower()
        
        # Prepare content list with context
        content_list, file_paths = self._prepare_contents(file_contents)
        
        # Use cached embeddings for content and query
        embeddings = self._get_cached_embeddings(content_list, self.content_embedding_cache)
        query_embedding = self._get_cached_embedding(clean_query, self.query_embedding_cache)
        
        # Execute retrieval strategy
        top_indices = self._execute_retrieval_strategy(
            embeddings, query_embedding, file_paths, top_k
        )
        
        # Get retrieved contents and paths
        retrieved_contents = [content_list[idx] for idx in top_indices]
        retrieved_paths = [file_paths[idx] for idx in top_indices]
        
        # Perform reranking
        return self._rerank_results(clean_query, retrieved_contents, retrieved_paths, top_k)
    
    def _execute_retrieval_strategy(
        self, 
        embeddings: np.ndarray, 
        query_embedding: np.ndarray, 
        file_paths: List[str], 
        top_k: int
    ) -> List[int]:
        """
        Execute the appropriate retrieval strategy.
        
        Args:
            embeddings: Document embeddings
            query_embedding: Query embedding
            file_paths: File paths
            top_k: Number of results to retrieve
            
        Returns:
            Indices of top documents
        """
        if self.retrieval_strategy == self._diverse_retrieval:
            return self.retrieval_strategy(
                embeddings, query_embedding, file_paths, top_k
            )
        else:
            return self.retrieval_strategy(
                embeddings, query_embedding, top_k
            )
    
    def _rerank_results(
        self, 
        query: str, 
        retrieved_contents: List[str], 
        retrieved_paths: List[str], 
        top_k: int
    ) -> List[str]:
        """
        Rerank retrieved results.
        
        Args:
            query: Search query
            retrieved_contents: Retrieved content texts
            retrieved_paths: Retrieved file paths
            top_k: Number of results to return
            
        Returns:
            Reranked file paths
        """
        # Advanced reranking
        reranked_contents = self.reranker.rerank(query, retrieved_contents, top_k)

        # Ensure the processing message is printed only once
        if not self._processing_message_shown:
            print("Processing queries... please wait.")
            self._processing_message_shown = True
        
        # Match reranked contents back to paths
        reranked_paths = [
            path for content, path in zip(retrieved_contents, retrieved_paths)
            if content in reranked_contents
        ]
        
        return reranked_paths


class QueryExpander:
    def __init__(self):
        """
        Initialize QueryExpander and ensure required NLTK resources are available.
        """
        self._ensure_nltk_resource('tokenizers/punkt')
        self._ensure_nltk_resource('corpora/wordnet')

    def _ensure_nltk_resource(self, resource_path: str):
        """
        Ensure that a specific NLTK resource is available.
        """
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_path.split('/')[-1], quiet=True)

    def get_synonyms(self, word: str) -> list:
        """
        Retrieve synonyms for a given word using WordNet.

        Args:
            word (str): Input word.

        Returns:
            list: List of synonym strings.
        """
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
        except Exception as e:
            print(f"[Synonym Error] Failed to get synonyms for '{word}': {e}")
        return list(synonyms)

    def simple_tokenize(self, query: str) -> list:
        """
        Fallback tokenizer: removes punctuation and splits on whitespace.

        Args:
            query (str): Input query.

        Returns:
            list: Tokenized list of words.
        """
        return re.findall(r'\w+', query.lower())

    def context_expand_query(self, query: str, context_words: list = None) -> str:
        """
        Expand query with synonyms and optional context words.

        Args:
            query (str): Input query.
            context_words (list, optional): List of context-specific words.

        Returns:
            str: Expanded query string.
        """
        try:
            try:
                tokens = nltk.word_tokenize(query.lower())
            except Exception:
                tokens = self.simple_tokenize(query)

            expanded_tokens = tokens + [
                synonym
                for token in tokens
                for synonym in self.get_synonyms(token)[:2]
            ]

            if context_words:
                expanded_tokens.extend(context_words)

            return ' '.join(expanded_tokens)

        except Exception as e:
            print(f"[Expansion Error] Failed to expand query: {e}")
            return query.lower()

    def expand_query(self, query: str, num_variations: int = 3) -> list:
        """
        Generate simple variations of the query including synonyms and reversed order.

        Args:
            query (str): Input query.
            num_variations (int): Number of variations to return.

        Returns:
            list: List of query variations.
        """
        try:
            variations = [
                self.context_expand_query(query),
                query.lower(),
                ' '.join(reversed(query.split()))
            ]
            return variations[:num_variations]

        except Exception as e:
            print(f"[Variation Error] Failed to generate query variations: {e}")
            return [query.lower()]
        

class RAGEvaluator:
    """
    Evaluator for Retrieval-Augmented Generation (RAG) systems with semantic matching capabilities.
    
    This class provides methods to evaluate RAG systems by comparing retrieved results against
    ground truth using semantic similarity metrics.
    """
    
    def __init__(
        self, 
        ground_truth: Dict[str, List[str]], 
        similarity_thresholds: Optional[List[float]] = None
    ):
        """
        Initialize the RAG evaluator with ground truth data and similarity thresholds.
        
        Args:
            ground_truth: Dictionary mapping queries to lists of expected file paths
            similarity_thresholds: List of similarity thresholds for multi-stage matching
        """
        self.ground_truth = ground_truth
        
        # Multi-stage similarity thresholds from strict to loose matching
        self.similarity_thresholds = similarity_thresholds or [0.9, 0.7, 0.5]
        
        # Initialize embedding model for semantic matching
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_semantic_context(self, file_path: str) -> str:
        """
        Extract meaningful semantic context from a file path.
        
        Args:
            file_path: File path to extract context from
        
        Returns:
            Semantic context string derived from path components
        """
        path_parts = file_path.split(os.path.sep)
        
        # Use the last 3 path components for context
        context_parts = path_parts[-3:] if len(path_parts) >= 3 else path_parts
        
        # Process each part: remove file extensions and normalize
        normalized_parts = []
        for part in context_parts:
            part_without_extension = re.sub(r'\.[^.]+$', '', part)
            normalized_part = part_without_extension.replace('_', ' ').lower()
            normalized_parts.append(normalized_part)
        
        # Join the normalized parts with spaces
        context = ' '.join(normalized_parts)
        
        return context

    def calculate_semantic_similarity(self, file_path1: str, file_path2: str) -> float:
        """
        Compute semantic similarity between two file paths.
        
        Args:
            file_path1: First file path
            file_path2: Second file path
        
        Returns:
            Semantic similarity score between 0 and 1
        """
        # Extract semantic contexts
        context1 = self.extract_semantic_context(file_path1)
        context2 = self.extract_semantic_context(file_path2)
        
        # Compute embeddings
        embeddings = self.embedding_model.encode([context1, context2])
        
        # Compute cosine similarity
        embedding1_norm = np.linalg.norm(embeddings[0])
        embedding2_norm = np.linalg.norm(embeddings[1])
        
        # Avoid division by zero
        if embedding1_norm == 0 or embedding2_norm == 0:
            return 0.0
            
        similarity = np.dot(embeddings[0], embeddings[1]) / (embedding1_norm * embedding2_norm)
        
        return float(similarity)

    def find_best_match_score(self, expected_file: str, retrieved_files: List[str]) -> float:
        """
        Find the best semantic match score between an expected file and a list of retrieved files.
        
        Args:
            expected_file: The expected file path to match
            retrieved_files: List of retrieved file paths to match against
            
        Returns:
            Highest similarity score found
        """
        if not retrieved_files:
            return 0.0
            
        similarity_scores = [
            self.calculate_semantic_similarity(expected_file, retrieved_file)
            for retrieved_file in retrieved_files
        ]
        
        return max(similarity_scores)

    def calculate_recall(self, retrieved_results: Dict[str, List[str]], k: int = 10) -> float:
        total_relevant = 0
        total_expected = 0
        
        for query, expected_files in self.ground_truth.items():
            retrieved_files = list(dict.fromkeys(retrieved_results.get(query, [])))[:k]
            
            # Check if there are any expected files
            matched = sum(
                1 for expected in expected_files
                if any(
                    self.calculate_semantic_similarity(expected, retrieved) >= threshold
                    for retrieved in retrieved_files
                    for threshold in self.similarity_thresholds
                )
            )
            
            total_relevant += matched
            total_expected += len(expected_files)
        
        return total_relevant / total_expected if total_expected > 0 else 0.0
    
    def calculate_precision(self, retrieved_results: Dict[str, List[str]], k: int = 10) -> float:
        """
        Calculate precision using multi-stage semantic matching.
        
        Args:
            retrieved_results: Dictionary mapping queries to lists of retrieved file paths
            k: Number of top results to consider
        
        Returns:
            Precision score between 0 and 1
        """
        if not self.ground_truth:
            return 0.0
            
        query_precisions = []
        
        for query, expected_files in self.ground_truth.items():
            if not expected_files:
                continue
                
            retrieved_files = retrieved_results.get(query, [])[:k]
            if not retrieved_files:
                query_precisions.append(0.0)
                continue
                
            # Count relevant documents among retrieved ones
            relevant_count = 0
            for retrieved_file in retrieved_files:
                if self._is_relevant_file(retrieved_file, expected_files):
                    relevant_count += 1
            
            # Calculate precision for this query
            query_precision = relevant_count / len(retrieved_files) if retrieved_files else 0.0
            query_precisions.append(query_precision)
        
        # Compute average precision across all queries
        final_precision = np.mean(query_precisions) if query_precisions else 0.0
        
        return min(1.0, final_precision)
    
    def _is_relevant_file(self, retrieved_file: str, expected_files: List[str]) -> bool:
        """
        Check if a retrieved file is relevant by matching against expected files.
        
        Args:
            retrieved_file: Retrieved file path to check
            expected_files: List of expected (relevant) file paths
            
        Returns:
            True if relevant, False otherwise
        """
        if not expected_files:
            return False
            
        # Check if the retrieved file matches any expected file
        for expected_file in expected_files:
            # Try different similarity thresholds from strict to loose
            for threshold in self.similarity_thresholds:
                similarity = self.calculate_semantic_similarity(retrieved_file, expected_file)
                
                # If similarity is above threshold, consider it relevant
                if similarity >= threshold:
                    return True
                    
        return False