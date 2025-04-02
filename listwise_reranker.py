import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re

class ListwiseReranker:
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = None,
        log_reranking: bool = True,
        boost_config: Optional[Dict[str, float]] = None
    ):
        """
        Enhanced Listwise Reranker with evaluation features.
        
        Args:
            model_name (str): Hugging Face cross-encoder model name
            device (str): Compute device (cuda/cpu)
            log_reranking (bool): Enable detailed reranking logs
            boost_config (dict): Custom boosting configuration
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
        if not candidates:
            return []

        code_keywords = self._extract_code_keywords(query)

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
        code_indicators = [
            'function', 'class', 'method', 'import', 'def', 
            'return', 'variable', 'parameter', 'async', 'await',
            'callback', 'promise', 'component', 'interface'
        ]
        keywords = []
        for word in query.lower().split():
            if word in code_indicators or (
                len(word) > 2 and (
                    '_' in word or 
                    (word[0].islower() and any(c.isupper() for c in word[1:]))
                )
            ):
                keywords.append(word)
        return keywords

    def _extract_file_path(self, candidate: str) -> str:
        match = re.search(r"File Path: ([^\n]+)", candidate)
        if match:
            return match.group(1)
        return ""

    def _apply_code_boosting(self, scores: np.ndarray, candidates: List[str], query: str) -> np.ndarray:
        cfg = self.boost_config
        boosted_scores = scores.copy()
        query_terms = set(re.findall(r'\w+', query.lower()))

        for i, candidate in enumerate(candidates):
            path = self._extract_file_path(candidate)
            content = candidate.lower()
            boost = 1.0

            if path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                boost += cfg["file_extension_boost"]

            if any(p in path.lower() for p in ['main', 'core', 'index', 'app']):
                boost += cfg["main_file_boost"]

            overlap = self._token_overlap_ratio(query, candidate)
            boost += min(overlap * cfg["token_overlap_weight"], cfg["token_overlap_cap"])

            if any(term in path.lower() for term in query_terms):
                boost += cfg["filename_query_match_boost"]

            first_occurrence = min((content.find(term) for term in query_terms if term in content), default=-1)
            if 0 <= first_occurrence < 300:
                boost += cfg["early_query_mention_boost"]

            comment_lines = [line for line in content.splitlines() if line.strip().startswith(('#', '//'))]
            comment_content = ' '.join(comment_lines).lower()
            comment_match = sum(1 for term in query_terms if term in comment_content)
            if comment_match > 0:
                boost += cfg["comment_match_boost"]

            if any(w in query.lower() for w in ['test', 'assert', 'unittest', 'pytest']):
                if 'test' in path.lower():
                    boost += cfg["test_file_boost"]

            boosted_scores[i] = scores[i] * boost

        return boosted_scores

    def _token_overlap_ratio(self, query: str, candidate: str) -> float:
        query_tokens = set(re.findall(r'\w+', query.lower()))
        candidate_tokens = set(re.findall(r'\w+', candidate.lower()))
        if not query_tokens:
            return 0.0
        return len(query_tokens & candidate_tokens) / len(query_tokens)

    def predict_relevance(self, query: str, candidate: str) -> float:
        inputs = self.tokenizer(
            [query, candidate], 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = outputs.logits.softmax(dim=1)
            relevance_score = probabilities[0][1].item()
            logits = outputs.logits[0]
            confidence = torch.max(probabilities[0]).item()

        return {
            'score': relevance_score,
            'confidence': confidence,
            'logits': logits.tolist()
        }

    def get_reranking_log(self) -> List[Dict]:
        return self.reranking_log

    def reset_reranking_log(self):
        self.reranking_log = []


# 🔍 Evaluacija rerankera pomoću Recall@10
def recall_at_k(ranked_candidates: List[str], relevant_files: List[str], k: int = 10) -> float:
    top_k = ranked_candidates[:k]
    return float(any(file in top_k for file in relevant_files))


def evaluate_reranker(reranker: ListwiseReranker, dataset: List[Dict], generate_candidates_fn, top_k: int = 10):
    """
    Evaluates the reranker on a dataset using Recall@10.

    Args:
        reranker: The ListwiseReranker instance.
        dataset: List of dicts with 'question' and 'files' (ground truth).
        generate_candidates_fn: Function that takes query and returns candidate file paths.
        top_k: Number of top results to consider.

    Returns:
        float: Recall@k score.
    """
    total = len(dataset)
    recall_count = 0
    for sample in dataset:
        query = sample["question"]
        relevant_files = sample["files"]
        candidates = generate_candidates_fn(query)
        reranked = reranker.rerank(query, candidates, top_k=top_k)
        if any(f in reranked for f in relevant_files):
            recall_count += 1
    recall = recall_count / total
    print(f"Recall@{top_k}: {recall:.4f}")
    return recall