import numpy as np
import torch
from typing import List, Dict, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        top_k: int = 10
    ) -> List[str]:
        """
        Rerank candidates with enhanced logging and detailed scoring.
        
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
        
        # Logging for evaluation
        if self.log_reranking:
            reranking_details = [
                {
                    'candidate': candidates[i],
                    'score': scores[i],
                    'rank': rank + 1
                }
                for rank, i in enumerate(ranked_indices[:top_k])
            ]
            
            self.reranking_log.append({
                'query': query,
                'candidates': reranking_details
            })

        return reranked_candidates

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