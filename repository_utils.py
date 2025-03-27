from git import Repo
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import nltk
from nltk.corpus import wordnet
import chardet
import re

# Cloning repository
def clone_repository(url, dest_folder):
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

def read_file_with_encoding(file_path):
    # First, try to detect the encoding
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
        
        # Try to read with the detected encoding
        with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
            return f.read()
    except Exception as e:
        # Fallback encodings if detection fails
        fallback_encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        for encoding in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    return f.read()
            except Exception as inner_e:
                print(f"Error reading {file_path} with encoding {encoding}: {inner_e}")
        
        print(f"Failed to read {file_path} with all encoding attempts.")
        return None

def prepare_repository_files(repo_path):
    allowed_extensions = ['.py', '.js', '.md', '.txt', '.json']
    
    def collect_files(directory):
        file_paths = []
        
        for root, dirs, files in os.walk(directory):
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


# Embeddings class
class RepositoryEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.file_paths = []

    def create_embeddings(self, file_contents):
        contents = list(file_contents.values())
        self.file_paths = list(file_contents.keys())
        embeddings = self.model.encode(contents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        return self.index

    def search(self, query, top_k=10):
        if self.index is None:
            raise ValueError("Index not created. Run create_embeddings first.")
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.file_paths[i] for i in indices[0]]

# Retriever class
class Retriever:
    def __init__(self, embedder):
        self.embedder = embedder

    def retrieve(self, query, top_k=10):
        return self.embedder.search(query, top_k)

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

    def generate_related_queries(self, query, num_variations=3):
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

# LLM Summarization class
class LLMSummarizer:
    def __init__(self, model="facebook/bart-large-cnn"):
        try:
            self.summarizer = pipeline("summarization", model=model)
        except Exception as e:
            print(f"Error initializing summarizer: {e}")
            self.summarizer = None

    def generate_summary(self, file_contents, max_length=150, min_length=30):
        # Handle various potential input issues
        if not file_contents or not isinstance(file_contents, str):
            return "Unable to generate summary. Invalid input."

        # Trim extremely long or very short inputs
        if len(file_contents) > 1000:
            file_contents = file_contents[:1000]
        
        if len(file_contents) < 50:
            return file_contents  # Return the content as-is if too short

        try:
            # Dynamically adjust max_length based on input length
            max_length = min(max_length, max(30, int(len(file_contents) * 0.3)))
            min_length = min(min_length, max_length - 10)

            # Attempt summarization
            summary = self.summarizer(
                file_contents, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False
            )

            # Extract summary text, handling potential empty results
            if summary and isinstance(summary, list) and summary[0]:
                return summary[0].get('summary_text', "Unable to generate summary.")
            
            return "Unable to generate summary."

        except Exception as e:
            print(f"Summary generation error: {e}")
            return "Unable to generate summary."

# Evaluation class
class RAGEvaluator:
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth

    def calculate_recall_at_k(self, retrieved_results, k=10):
        recalls = []
        for query, expected_files in self.ground_truth.items():
            expected_set = set(expected_files)
            retrieved_set = set(retrieved_results.get(query, [])[:k])
            recall = len(expected_set.intersection(retrieved_set)) / len(expected_set) if expected_set else (1.0 if not retrieved_set else 0.0)
            recalls.append(recall)
        return np.mean(recalls)

    def generate_report(self, retrieved_results, k=10):
        recall = self.calculate_recall_at_k(retrieved_results, k)
        print(f"Evaluation Report:\nRecall@{k}: {recall:.4f}")
        return {'recall': recall, 'k': k}
