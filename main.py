import argparse
import traceback
from repository_utils import (
    clone_repository, 
    prepare_repository_files, 
    AdvancedCodeRAGSystem,
    QueryExpander,
    RAGEvaluator,
)
import os

def parse_args():
    """
    Parse command-line arguments for the advanced CodeRAG system.
    """
    parser = argparse.ArgumentParser(description="Advanced CodeRAG: Repository Question Answering")
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
        "--top_k", 
        type=int, 
        default=10, 
        help="Number of top search results"
    )
    parser.add_argument(
        "--retrieval_strategy", 
        type=str, 
        default="diverse", 
        choices=["default", "probabilistic", "diverse"],
        help="Retrieval strategy"
    )
    parser.add_argument(
        "--generate_summaries", 
        action="store_true",
        help="Enable generation of file content summaries"
    )
    return parser.parse_args()

def read_file_content(file_path):
    """
    Safely read file content with error handling.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def main():
    """
    Main execution function for advanced CodeRAG system.
    """
    try:
        # Parse arguments
        args = parse_args()

        # Clone repository
        cloned_path = clone_repository(args.repo_url, args.repo_path)
        if not cloned_path:
            print("Repository cloning failed.")
            return

        # Prepare repository files
        repository_files = prepare_repository_files(cloned_path)
        if not repository_files:
            print("No files found in repository.")
            return

        # Initialize Advanced CodeRAG system
        rag_system = AdvancedCodeRAGSystem(
            retrieval_strategy=args.retrieval_strategy
        )

        # Initialize query expander
        query_expander = QueryExpander()

        # More precise test queries with expected context
        test_queries = {
            "android screen recording": [
                os.path.join(cloned_path, "src", "composables", "useScreenshotAction", "index.js"),
                os.path.join(cloned_path, "README.md")
            ],
            "device connection": [
                os.path.join(cloned_path, "electron", "exposes", "adb", "helpers", "scanner", "index.js"),
                os.path.join(cloned_path, "src", "dicts", "device", "index.js")
            ]
        }

        # Evaluator
        evaluator = RAGEvaluator(test_queries)
        retrieved_results = {}

        # Process queries
        for query, expected_files in test_queries.items():
            # Expand queries
            expanded_queries = query_expander.expand_query(query)
            
            results = []
            for expanded_query in expanded_queries:
                query_results = rag_system.advanced_retrieve(
                    expanded_query, 
                    repository_files, 
                    args.top_k
                )
                results.extend(query_results)
            
            # Remove duplicates and truncate
            results = list(dict.fromkeys(results))[:args.top_k]
            retrieved_results[query] = results

            print(f"\nQuery: {query}")
            for result in results:
                # Read file content
                content = read_file_content(result)
                
                print(f"File: {result}")
                
                # Generate summary only if flag is set
                if args.generate_summaries:
                    summary = rag_system.summarizer.generate_summary(content, result)
                    print(f"Summary: {summary}\n")
                else:
                    print()  # Just a newline for consistent formatting

        # Generate evaluation report
        evaluator.generate_report(retrieved_results)

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()