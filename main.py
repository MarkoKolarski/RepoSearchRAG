import argparse
import traceback
from repository_utils import (
    clone_repository, 
    prepare_repository_files, 
    CodeRAGSystem, 
    QueryExpander,
    RAGEvaluator
)

def parse_args():
    """
    Parse command-line arguments for the CodeRAG system.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="CodeRAG: Repository Question Answering")
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
    return parser.parse_args()

def main():
    """
    Main execution function for CodeRAG system.
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

        # Initialize CodeRAG system
        rag_system = CodeRAGSystem()
        rag_system.create_embeddings(repository_files)

        # Initialize query expander
        query_expander = QueryExpander()

        # Test queries (from task specification)
        test_queries = {
            "android screen recording": [
                "repository\\src\\composables\\useScreenshotAction\\index.js",
                "repository\\README.md"
            ],
            "device connection": [
                "repository\\electron\\exposes\\adb\\helpers\\scanner\\index.js",
                "repository\\src\\dicts\\device\\index.js"
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
                query_results = rag_system.retrieve_files(expanded_query, args.top_k)
                results.extend(query_results)
            
            # Remove duplicates and truncate
            results = list(dict.fromkeys(results))[:args.top_k]
            retrieved_results[query] = results

            print(f"\nQuery: {query}")
            for result in results:
                try:
                    with open(result, 'r', encoding='utf-8', errors='replace') as f:
                        file_content = f.read()[:1000]  # Limit content
                    
                    # Generate summary
                    summary = rag_system.generate_summary(file_content)
                    print(f"File: {result}")
                    print(f"Summary: {summary}\n")
                except Exception as file_error:
                    print(f"Error processing {result}: {file_error}")

        # Generate evaluation report
        evaluator.generate_report(retrieved_results)

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()