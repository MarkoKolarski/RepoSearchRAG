import numpy as np
import torch
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ListwiseReranker:
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = None
    ):
        """
        Initialize Listwise Reranker with cross-encoder model.
        
        Args:
            model_name (str): Hugging Face cross-encoder model name
            device (str): Compute device (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 10
    ) -> List[str]:
        """
        Rerank candidates using cross-encoder model.
        
        Args:
            query (str): Search query
            candidates (List[str]): List of candidate file contents
            top_k (int): Number of top results to return
        
        Returns:
            List[str]: Reranked candidates
        """
        if not candidates:
            return []

        # Prepare inputs
        inputs = [
            (query, candidate) for candidate in candidates
        ]

        # Tokenize and get scores
        features = self.tokenizer(
            inputs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**features).logits.cpu().numpy().flatten()

        # Sort candidates by scores
        ranked_indices = np.argsort(scores)[::-1]
        reranked_candidates = [candidates[i] for i in ranked_indices[:top_k]]

        return reranked_candidates

    def predict_relevance(self, query: str, candidate: str) -> float:
        """
        Predict relevance score between query and candidate.
        
        Args:
            query (str): Search query
            candidate (str): Candidate text
        
        Returns:
            float: Relevance score
        """
        inputs = self.tokenizer(
            [query, candidate], 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            relevance_score = outputs.logits.softmax(dim=1)[0][1].item()

        return relevance_score