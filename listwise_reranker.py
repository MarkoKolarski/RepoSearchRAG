import re
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ListwiseReranker:
    """A reranker that orders search results based on relevance to a query using a cross-encoder model."""

    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = None,
        log_reranking: bool = True,
        boost_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the reranker with a cross-encoder model and boosting configuration.
        
        Args:
            model_name: Hugging Face cross-encoder model name
            device: Compute device (cuda/cpu)
            log_reranking: Enable detailed reranking logs
            boost_config: Custom boosting configuration
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.log_reranking = log_reranking
        self.reranking_log = []

        # Default boosting configuration
        self.boost_config = boost_config or {
            "file_extension_boost": 0.05,
            "main_file_boost": 0.04,
            "token_overlap_weight": 0.15,
            "token_overlap_cap": 0.05,
            "filename_query_match_boost": 0.05,
            "early_query_mention_boost": 0.05,
            "comment_match_boost": 0.03,
            "test_file_boost": 0.05
        }

    def rerank(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 10,
        file_contents: Dict[str, str] = None
    ) -> List[str]:
        """
        Rerank candidates based on relevance to the query.
        
        Args:
            query: The search query
            candidates: List of candidate documents
            top_k: Number of top results to return
            file_contents: Optional dictionary mapping file paths to contents
            
        Returns:
            List of reranked candidates, limited to top_k results
        """
        if not candidates:
            return []

        code_keywords = self._extract_code_keywords(query)
        inputs = self._prepare_inputs(query, candidates, code_keywords)
        
        features = self.tokenizer(
            inputs, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**features).logits.cpu().numpy().flatten()

        boosted_scores = self._apply_code_boosting(scores, candidates, query)
        ranked_indices = np.argsort(boosted_scores)[::-1]
        reranked_candidates = [candidates[i] for i in ranked_indices[:top_k]]

        if self.log_reranking:
            self._log_reranking_results(query, candidates, scores, boosted_scores, ranked_indices, top_k)

        return reranked_candidates

    def _prepare_inputs(self, query: str, candidates: List[str], code_keywords: List[str]) -> List[tuple]:
        """Prepare inputs for the model by enhancing queries with file extensions and code keywords."""
        inputs = []
        for candidate in candidates:
            file_path = self._extract_file_path(candidate)
            extension = file_path.split('.')[-1] if '.' in file_path else ''
            enhanced_query = query
            
            if extension:
                enhanced_query = f"[{extension}] {query}"
                
            if code_keywords:
                enhanced_query = f"{enhanced_query} {' '.join(code_keywords)}"
                
            inputs.append((enhanced_query, candidate))
            
        return inputs

    def _log_reranking_results(
        self, 
        query: str, 
        candidates: List[str], 
        scores: np.ndarray, 
        boosted_scores: np.ndarray, 
        ranked_indices: np.ndarray, 
        top_k: int
    ) -> None:
        """Log detailed information about the reranking process."""
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

    def _extract_code_keywords(self, query: str) -> List[str]:
        """Extract programming-related keywords from the query."""
        code_indicators = [
            'function', 'class', 'method', 'import', 'def', 
            'return', 'variable', 'parameter', 'async', 'await',
            'callback', 'promise', 'component', 'interface'
        ]
        
        keywords = []
        for word in query.lower().split():
            is_code_indicator = word in code_indicators
            is_likely_code_term = (
                len(word) > 2 and (
                    '_' in word or 
                    (word[0].islower() and any(c.isupper() for c in word[1:]))
                )
            )
            
            if is_code_indicator or is_likely_code_term:
                keywords.append(word)
                
        return keywords

    def _extract_file_path(self, candidate: str) -> str:
        """Extract file path from candidate text."""
        match = re.search(r"File Path: ([^\n]+)", candidate)
        return match.group(1) if match else ""

    def _apply_code_boosting(self, scores: np.ndarray, candidates: List[str], query: str) -> np.ndarray:
        """Apply code-specific boosting factors to base relevance scores."""
        cfg = self.boost_config
        boosted_scores = scores.copy()
        query_terms = set(re.findall(r'\w+', query.lower()))

        for i, candidate in enumerate(candidates):
            path = self._extract_file_path(candidate)
            content = candidate.lower()
            boost = 1.0

            # Apply boosting factors
            boost = self._apply_extension_boost(boost, path, cfg)
            boost = self._apply_main_file_boost(boost, path, cfg)
            boost = self._apply_token_overlap_boost(boost, query, candidate, cfg)
            boost = self._apply_filename_match_boost(boost, path, query_terms, cfg)
            boost = self._apply_early_mention_boost(boost, content, query_terms, cfg)
            boost = self._apply_comment_match_boost(boost, content, query_terms, cfg)
            boost = self._apply_test_file_boost(boost, path, query, cfg)

            boosted_scores[i] = scores[i] * boost

        return boosted_scores
    
    def _apply_extension_boost(self, boost: float, path: str, cfg: Dict[str, float]) -> float:
        """Boost scores for common programming file extensions."""
        if path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
            boost += cfg["file_extension_boost"]
        return boost
    
    def _apply_main_file_boost(self, boost: float, path: str, cfg: Dict[str, float]) -> float:
        """Boost scores for main/core files."""
        if any(p in path.lower() for p in ['main', 'core', 'index', 'app']):
            boost += cfg["main_file_boost"]
        return boost
    
    def _apply_token_overlap_boost(self, boost: float, query: str, candidate: str, cfg: Dict[str, float]) -> float:
        """Boost scores based on token overlap between query and candidate."""
        overlap = self._token_overlap_ratio(query, candidate)
        boost += min(overlap * cfg["token_overlap_weight"], cfg["token_overlap_cap"])
        return boost
    
    def _apply_filename_match_boost(
        self, 
        boost: float, 
        path: str, 
        query_terms: set, 
        cfg: Dict[str, float]
    ) -> float:
        """Boost scores when query terms appear in the filename."""
        if any(term in path.lower() for term in query_terms):
            boost += cfg["filename_query_match_boost"]
        return boost
    
    def _apply_early_mention_boost(
        self, 
        boost: float, 
        content: str, 
        query_terms: set, 
        cfg: Dict[str, float]
    ) -> float:
        """Boost scores when query terms appear early in the content."""
        first_occurrence = min((content.find(term) for term in query_terms if term in content), default=-1)
        if 0 <= first_occurrence < 300:
            boost += cfg["early_query_mention_boost"]
        return boost
    
    def _apply_comment_match_boost(
        self, 
        boost: float, 
        content: str, 
        query_terms: set, 
        cfg: Dict[str, float]
    ) -> float:
        """Boost scores when query terms appear in comments."""
        comment_lines = [line for line in content.splitlines() if line.strip().startswith(('#', '//'))]
        comment_content = ' '.join(comment_lines).lower()
        comment_match = sum(1 for term in query_terms if term in comment_content)
        if comment_match > 0:
            boost += cfg["comment_match_boost"]
        return boost
    
    def _apply_test_file_boost(self, boost: float, path: str, query: str, cfg: Dict[str, float]) -> float:
        """Boost scores for test files when the query is test-related."""
        if any(w in query.lower() for w in ['test', 'assert', 'unittest', 'pytest']):
            if 'test' in path.lower():
                boost += cfg["test_file_boost"]
        return boost

    def _token_overlap_ratio(self, query: str, candidate: str) -> float:
        """
        Calculate the ratio of query tokens that appear in the candidate.
        
        Returns:
            Float between 0 and 1 representing overlap ratio
        """
        query_tokens = set(re.findall(r'\w+', query.lower()))
        candidate_tokens = set(re.findall(r'\w+', candidate.lower()))
        
        if not query_tokens:
            return 0.0
            
        return len(query_tokens & candidate_tokens) / len(query_tokens)