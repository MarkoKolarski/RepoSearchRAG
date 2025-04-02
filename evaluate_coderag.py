#!/usr/bin/env python
import os
import json
import argparse
import time
from typing import Dict, List
from repository_utils import (
    clone_repository,
    prepare_repository_files,
    AdvancedCodeRAGSystem,
    QueryExpander,
    RAGEvaluator,
)

class CodeRAGEvaluator:
    """
    Standalone evaluator for CodeRAG systems that measures retrieval performance
    using Recall@10 as the primary metric.
    """
    def __init__(
        self,
        repo_url: str,
        repo_path: str,
        dataset_path: str,
        retrieval_strategy: str = "diverse",
        top_k: int = 10,
    ):
        """
        Initialize the CodeRAG evaluator.

        Args:
            repo_url: URL of the repository to evaluate.
            repo_path: Local path where the repository will be cloned.
            dataset_path: Path to the evaluation dataset (JSON).
            retrieval_strategy: Retrieval strategy to use (default: "diverse").
            top_k: Number of top results to consider for metrics.
        """
        self.repo_url = repo_url
        self.repo_path = repo_path
        self.dataset_path = dataset_path
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k

        # Load dataset
        self.test_queries = self.load_dataset()

        # Performance metrics
        self.results = {}

    def load_dataset(self) -> Dict[str, List[str]]:
        """Load test queries from a dataset file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert dataset to expected format
        return {
            item["question"]: [
                os.path.join(self.repo_path, path.lstrip("/"))
                for path in item["files"]
            ]
            for item in data
        }

    def setup(self):
        """Set up the repository and prepare files for evaluation."""
        # Clone repository
        cloned_path = clone_repository(self.repo_url, self.repo_path)
        if not cloned_path:
            raise RuntimeError("Repository cloning failed.")

        # Prepare repository files
        self.repository_files = prepare_repository_files(cloned_path)
        if not self.repository_files:
            raise RuntimeError("No files found in repository.")

        # Initialize the RAG system
        self.query_expander = QueryExpander()
        self.rag_system = AdvancedCodeRAGSystem(
            retrieval_strategy=self.retrieval_strategy,
        )

        # Initialize the evaluator
        self.evaluator = RAGEvaluator(self.test_queries)

    def evaluate(self):
        """
        Evaluate the system using the specified retrieval strategy.

        Returns:
            Dict: Performance metrics for the evaluation.
        """
        start_time = time.time()

        # Dictionary to store retrieved results
        retrieved_results = {}

        # Process each query
        query_times = []
        for query, expected_files in self.test_queries.items():
            query_start = time.time()

            # Expand queries
            expanded_queries = self.query_expander.expand_query(query)
            #expanded_queries = [query]
            
            results = []
            for expanded_query in expanded_queries:
                query_results = self.rag_system.advanced_retrieve(
                    expanded_query,
                    self.repository_files,
                    self.top_k
                )
                results.extend(query_results)

            # Remove duplicates and truncate
            results = list(dict.fromkeys(results))[:self.top_k]
            retrieved_results[query] = results

            query_time = time.time() - query_start
            query_times.append(query_time)

        # Calculate Recall@10
        recall = self.evaluator.calculate_recall(retrieved_results, self.top_k)
        precison = self.evaluator.calculate_precision(retrieved_results, self.top_k)

        # Calculate total evaluation time
        total_time = time.time() - start_time

        return {
            "strategy": self.retrieval_strategy,
            "recall": recall,
            "precision": precison,
            "average_query_time": sum(query_times) / len(query_times),
            "total_time": total_time,
            "retrieved_results": retrieved_results
        }

    def save_results(self, output_path: str = "evaluation_results.json"):
        """Save evaluation results to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_path}")


def parse_args():
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate CodeRAG System Performance")
    parser.add_argument(
        "--repo_url",
        type=str,
        default="https://github.com/viarotel-org/escrcpy",  # Default repository URL
        help="GitHub repository URL",
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        default="repository",
        help="Local repository path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="escrcpy-commits-generated.json",  # Default dataset file
        help="Path to evaluation dataset (JSON format)",
    )
    parser.add_argument(
        "--retrieval_strategy",
        type=str,
        default="diverse",  # Default retrieval strategy
        choices=["default", "probabilistic", "diverse"],
        help="Retrieval strategy to use (default: diverse)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top search results (for Recall@k)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",  # Default output file
        help="Path to save evaluation results",
    )
    return parser.parse_args()


def main():
    """Main execution function for the evaluation script."""
    try:
        # Parse arguments
        args = parse_args()

        # Initialize evaluator
        evaluator = CodeRAGEvaluator(
            repo_url=args.repo_url,
            repo_path=args.repo_path,
            dataset_path=args.dataset,
            retrieval_strategy=args.retrieval_strategy,
            top_k=args.top_k,
        )

        # Setup repository and prepare files
        print("Setting up repository and preparing files...")
        evaluator.setup()

        # Run evaluation
        print("Running evaluation...")
        results = evaluator.evaluate()

        # Print results
        print(f"\nEvaluation Results:")
        print(f"  Strategy: {results['strategy']}")
        print(f"  Recall@{args.top_k}: {results['recall']:.4f}")
        print(f"  Average Query Time: {results['average_query_time']:.4f} seconds")
        print(f"  Total Evaluation Time: {results['total_time']:.4f} seconds")

        # Save results
        evaluator.results = results
        evaluator.save_results(args.output)

        print("Evaluation completed successfully!")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()