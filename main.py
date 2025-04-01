import argparse
import traceback
import os
from repository_utils import (
    clone_repository, 
    prepare_repository_files, 
    AdvancedCodeRAGSystem,
    QueryExpander,
    RAGEvaluator,
    create_summarizer,
)

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
    parser.add_argument(
        "--use_large_summarizer", 
        action="store_true",
        help="Use larger, more capable summarization model"
    )
    parser.add_argument(
        "--use_gemini",
        action="store_true",
        help="Use Google Gemini API for summarization"
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        help="Google Gemini API key (or set GOOGLE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-1.5-flash",
        help="Google Gemini model to use (default: gemini-1.5-flash)"
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

        # Get API key from command line or environment variable
        api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        
        # Create summarizer based on user input
        if args.use_gemini:
            if not api_key:
                print("Error: Google Gemini API key must be provided via --gemini_api_key or GOOGLE_API_KEY environment variable")
                return
            print(f"Using Google Gemini API with {args.gemini_model} model...")
            summarizer = create_summarizer(
                use_api=True, 
                api_key=api_key, 
                api_model_name=args.gemini_model
            )
        elif args.use_large_summarizer:
            print("Using large summarization model...")
            summarizer = create_summarizer(use_large_model=True)
        else:
            print("Using default small summarization model...")
            summarizer = create_summarizer()

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

        # Initialize Advanced CodeRAG system with custom summarizer
        rag_system = AdvancedCodeRAGSystem(
            retrieval_strategy=args.retrieval_strategy,
            summarizer=summarizer  # Pass the created summarizer
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