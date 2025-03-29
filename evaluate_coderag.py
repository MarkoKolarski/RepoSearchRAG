#!/usr/bin/env python
import os
import json
import argparse
import time
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from repository_utils import (
    clone_repository,
    prepare_repository_files,
    AdvancedCodeRAGSystem,
    QueryExpander,
    RAGEvaluator,
    create_summarizer,
)

class CodeRAGEvaluator:
    """
    Standalone evaluator for CodeRAG systems that measures retrieval performance
    using various metrics, with a focus on Recall@10.
    """
    
    def __init__(
        self, 
        repo_url: str = "https://github.com/viarotel-org/escrcpy",
        repo_path: str = "repository",
        retrieval_strategies: List[str] = ["default", "probabilistic", "diverse"],
        dataset_path: Optional[str] = None,
        use_large_summarizer: bool = False,
        top_k: int = 10,
    ):
        """
        Initialize the CodeRAG evaluator.
        
        Args:
            repo_url: URL of the repository to evaluate
            repo_path: Local path where repository will be cloned
            retrieval_strategies: List of retrieval strategies to evaluate
            dataset_path: Path to custom evaluation dataset (JSON)
            use_large_summarizer: Whether to use a larger summarization model
            top_k: Number of top results to consider for metrics
        """
        self.repo_url = repo_url
        self.repo_path = repo_path
        self.retrieval_strategies = retrieval_strategies
        self.dataset_path = dataset_path
        self.use_large_summarizer = use_large_summarizer
        self.top_k = top_k
        
        # Default test queries if no dataset is provided
        self.test_queries = {
            "android screen recording": [
                os.path.join(repo_path, "src", "composables", "useScreenshotAction", "index.js"),
                os.path.join(repo_path, "README.md")
            ],
            "device connection": [
                os.path.join(repo_path, "electron", "exposes", "adb", "helpers", "scanner", "index.js"),
                os.path.join(repo_path, "src", "dicts", "device", "index.js")
            ]
        }
        
        # Performance metrics
        self.results = {}
        
    def load_dataset(self):
        """Load test queries from a dataset file if provided."""
        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                # Convert dataset to expected format
                if isinstance(loaded_data, list):
                    # List of query/expected files pairs
                    self.test_queries = {
                        item["query"]: [
                            os.path.join(self.repo_path, path.lstrip("/"))
                            for path in item["expected_files"]
                        ]
                        for item in loaded_data
                    }
                elif isinstance(loaded_data, dict):
                    # Dictionary mapping queries to expected files
                    self.test_queries = {
                        query: [
                            os.path.join(self.repo_path, path.lstrip("/"))
                            for path in expected_files
                        ]
                        for query, expected_files in loaded_data.items()
                    }
                
                print(f"Loaded {len(self.test_queries)} test queries from dataset.")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Falling back to default test queries.")
        else:
            print("Using default test queries.")
    
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
        
        # Load dataset if provided
        self.load_dataset()
        
        # Create summarizer based on configuration
        self.summarizer = create_summarizer(use_large_model=self.use_large_summarizer)
        
        # Initialize query expander
        self.query_expander = QueryExpander()
        
        # Initialize the evaluator
        self.evaluator = RAGEvaluator(self.test_queries)
        
    def evaluate_strategy(self, strategy: str):
        """
        Evaluate a specific retrieval strategy.
        
        Args:
            strategy: The retrieval strategy to evaluate
            
        Returns:
            Dict: Performance metrics for the strategy
        """
        start_time = time.time()
        
        # Initialize RAG system with the strategy
        rag_system = AdvancedCodeRAGSystem(
            retrieval_strategy=strategy,
            summarizer=self.summarizer
        )
        
        # Dictionary to store retrieved results
        retrieved_results = {}
        
        # Process each query
        query_times = []
        for query, expected_files in self.test_queries.items():
            query_start = time.time()
            
            # Expand queries
            expanded_queries = self.query_expander.expand_query(query)
            
            results = []
            for expanded_query in expanded_queries:
                query_results = rag_system.advanced_retrieve(
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
        
        # Calculate metrics
        recall = self.evaluator.calculate_recall(retrieved_results, self.top_k)
        precision = self.evaluator.calculate_precision(retrieved_results, self.top_k)
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate latency metrics
        avg_query_time = np.mean(query_times)
        total_time = time.time() - start_time
        
        return {
            "strategy": strategy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "avg_query_time": avg_query_time,
            "total_time": total_time,
            "retrieved_results": retrieved_results
        }
    
    def run_evaluation(self):
        """Run evaluation on all configured retrieval strategies."""
        print(f"Starting evaluation with {len(self.retrieval_strategies)} retrieval strategies...")
        print(f"Test queries: {len(self.test_queries)}")
        
        for strategy in self.retrieval_strategies:
            print(f"\nEvaluating strategy: {strategy}")
            metrics = self.evaluate_strategy(strategy)
            self.results[strategy] = metrics
            
            # Print results
            print(f"Results for {strategy}:")
            print(f"  Recall@{self.top_k}: {metrics['recall']:.4f}")
            print(f"  Precision@{self.top_k}: {metrics['precision']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Avg Query Time: {metrics['avg_query_time']:.4f} seconds")
            print(f"  Total Time: {metrics['total_time']:.4f} seconds")
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        # Create a simplified version of results for saving
        simplified_results = {}
        for strategy, metrics in self.results.items():
            simplified_results[strategy] = {
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1_score": metrics["f1_score"],
                "avg_query_time": metrics["avg_query_time"],
                "total_time": metrics["total_time"],
                # Include retrieved results for each query
                "retrieved_files": {
                    query: [os.path.relpath(path, self.repo_path) for path in files]
                    for query, files in metrics["retrieved_results"].items()
                }
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def plot_results(self, output_path: str = "evaluation_plot.png"):
        """
        Generate and save a visualization of evaluation results.
        
        Args:
            output_path: Path to save the plot
        """
        strategies = list(self.results.keys())
        recalls = [self.results[s]["recall"] for s in strategies]
        precisions = [self.results[s]["precision"] for s in strategies]
        f1_scores = [self.results[s]["f1_score"] for s in strategies]
        query_times = [self.results[s]["avg_query_time"] for s in strategies]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot metrics
        x = np.arange(len(strategies))
        width = 0.25
        
        ax1.bar(x - width, recalls, width, label='Recall@10')
        ax1.bar(x, precisions, width, label='Precision@10')
        ax1.bar(x + width, f1_scores, width, label='F1 Score')
        
        ax1.set_xlabel('Retrieval Strategy')
        ax1.set_ylabel('Score')
        ax1.set_title('Retrieval Performance Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.set_ylim(0, 1.0)
        
        # Plot query times
        ax2.bar(strategies, query_times, color='orange')
        ax2.set_xlabel('Retrieval Strategy')
        ax2.set_ylabel('Average Query Time (seconds)')
        ax2.set_title('Latency Performance')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Performance plot saved to {output_path}")

def parse_args():
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate CodeRAG System Performance")
    parser.add_argument(
        "--repo_url", 
        type=str, 
        default="https://github.com/viarotel-org/escrcpy", 
        help="GitHub repository URL"
    )
    parser.add_argument(
        "--repo_path", 
        type=str, 
        default="repository", 
        help="Local repository path"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=None, 
        help="Path to evaluation dataset (JSON format)"
    )
    parser.add_argument(
        "--strategies", 
        type=str, 
        default="default,probabilistic,diverse", 
        help="Comma-separated list of retrieval strategies to evaluate"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=10, 
        help="Number of top search results (for Recall@k)"
    )
    parser.add_argument(
        "--use_large_summarizer", 
        action="store_true",
        help="Use larger, more capable summarization model"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="evaluation_results.json", 
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--plot", 
        type=str, 
        default="evaluation_plot.png", 
        help="Path to save evaluation plot"
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
            retrieval_strategies=args.strategies.split(','),
            dataset_path=args.dataset,
            use_large_summarizer=args.use_large_summarizer,
            top_k=args.top_k,
        )
        
        # Setup repository and prepare files
        print("Setting up repository and preparing files...")
        evaluator.setup()
        
        # Run evaluation
        evaluator.run_evaluation()
        
        # Save results
        evaluator.save_results(args.output)
        
        # Generate and save plot
        evaluator.plot_results(args.plot)
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()