import numpy as np
import torch
from typing import List, Dict, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re

class ListwiseReranker:
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = None,
        log_reranking: bool = True
    ):
        """
        Enhanced Listwise Reranker with evaluation features.
        
        Args:
            model_name (str): Hugging Face cross-encoder model name
            device (str): Compute device (cuda/cpu)
            log_reranking (bool): Enable detailed reranking logs
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Logging configuration
        self.log_reranking = log_reranking
        self.reranking_log = []

    def rerank(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 10,
        file_contents: Dict[str, str] = None
    ) -> List[str]:
        """
        Enhanced reranking with code-specific optimizations.
        
        Args:
            query: Search query
            candidates: List of candidate texts
            top_k: Number of top results to return
            file_contents: Original file contents for context
            
        Returns:
            Reranked candidates
        """
        if not candidates:
            return []

        # Extract code keywords from query
        code_keywords = self._extract_code_keywords(query)
        
        # Prepare enhanced inputs with emphasis on code structure
        inputs = []
        for candidate in candidates:
            # Extract file path for context
            file_path = self._extract_file_path(candidate)
            
            # Get file extension to determine language
            extension = file_path.split('.')[-1] if '.' in file_path else ''
            
            # Create enhanced query with type hints
            enhanced_query = query
            if extension:
                enhanced_query = f"[{extension}] {query}"
            
            # Add keywords if found
            if code_keywords:
                enhanced_query = f"{enhanced_query} {' '.join(code_keywords)}"
                
            inputs.append((enhanced_query, candidate))

        # Tokenize and get scores with longer max length for code
        features = self.tokenizer(
            inputs, 
            padding=True, 
            truncation=True, 
            max_length=512,  # Increased from default
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**features).logits.cpu().numpy().flatten()
        
        # Apply code-specific boosting factors
        boosted_scores = self._apply_code_boosting(scores, candidates, query)
        
        # Sort candidates by boosted scores
        ranked_indices = np.argsort(boosted_scores)[::-1]
        reranked_candidates = [candidates[i] for i in ranked_indices[:top_k]]
        
        # Log for evaluation
        if self.log_reranking:
            reranking_details = [
                {
                    'candidate': candidates[i],
                    'original_score': scores[i],
                    'boosted_score': boosted_scores[i],
                    'rank': rank + 1
                }
                for rank, i in enumerate(ranked_indices[:top_k])
            ]
            
            self.reranking_log.append({
                'query': query,
                'candidates': reranking_details
            })

        return reranked_candidates

    def _extract_code_keywords(self, query: str) -> List[str]:
        """Extract likely code-related keywords from query"""
        code_indicators = [
            'function', 'class', 'method', 'import', 'def', 
            'return', 'variable', 'parameter', 'async', 'await',
            'callback', 'promise', 'component', 'interface'
        ]
        
        keywords = []
        for word in query.lower().split():
            if word in code_indicators or (
                len(word) > 2 and (
                    '_' in word or  # snake_case
                    (word[0].islower() and any(c.isupper() for c in word[1:]))  # camelCase
                )
            ):
                keywords.append(word)
        
        return keywords

    def _extract_file_path(self, candidate: str) -> str:
        """Extract file path from candidate text"""
        match = re.search(r"File Path: ([^\n]+)", candidate)
        if match:
            return match.group(1)
        return ""

    def _apply_code_boosting(
        self, 
        scores: np.ndarray, 
        candidates: List[str], 
        query: str
    ) -> np.ndarray:
        """Apply code-specific boosting to scores"""
        boosted_scores = scores.copy()
        
        # Get query terms for exact matching
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        for i, candidate in enumerate(candidates):
            # Extract path and check for important file patterns
            path = self._extract_file_path(candidate)
            
            # Boost important file types
            if path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                boosted_scores[i] *= 1.15
                
            # Boost core/main files
            if any(pattern in path.lower() for pattern in ['main', 'core', 'index', 'app']):
                boosted_scores[i] *= 1.1
                
            # Boost for exact term matches in content
            content = candidate.lower()
            matched_terms = sum(1 for term in query_terms if term in content)
            term_match_ratio = matched_terms / len(query_terms) if query_terms else 0
            
            # Apply term match boosting
            boosted_scores[i] *= (1 + (term_match_ratio * 0.2))
        
        return boosted_scores

    def predict_relevance(self, query: str, candidate: str) -> float:
        """
        Predict relevance score between query and candidate with confidence.
        
        Args:
            query (str): Search query
            candidate (str): Candidate text
        
        Returns:
            float: Relevance score with confidence interval
        """
        inputs = self.tokenizer(
            [query, candidate], 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Softmax probabilities
            probabilities = outputs.logits.softmax(dim=1)
            
            # Relevance score
            relevance_score = probabilities[0][1].item()
            
            # Compute confidence interval
            logits = outputs.logits[0]
            confidence = torch.max(probabilities[0]).item()

        return {
            'score': relevance_score,
            'confidence': confidence,
            'logits': logits.tolist()
        }

    def get_reranking_log(self) -> List[Dict]:
        """
        Retrieve detailed reranking logs.
        
        Returns:
            List[Dict]: Detailed reranking logs
        """
        return self.reranking_log

    def reset_reranking_log(self):
        """
        Reset reranking logs.
        """
        self.reranking_log = []