import argparse
import traceback
import os
from argparse import Namespace
from repository_utils import (
    clone_repository,
    prepare_repository_files,
    AdvancedCodeRAGSystem,
    QueryExpander,
    create_summarizer,
)


def parse_args() -> Namespace:
    """
    Parse command-line arguments for the Advanced CodeRAG system.
    """
    parser = argparse.ArgumentParser(
        description="Advanced CodeRAG: Repository Question Answering System"
    )

    # Repository options
    parser.add_argument(
        "--repo_url",
        type=str,
        default="https://github.com/viarotel-org/escrcpy",
        help="GitHub repository URL to clone"
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        default="repository",
        help="Local path where the repository will be stored"
    )

    # Retrieval configuration
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top search results to retrieve"
    )
    parser.add_argument(
        "--retrieval_strategy",
        type=str,
        default="diverse",
        choices=["default", "probabilistic", "diverse"],
        help="Retrieval strategy to use for document selection"
    )

    # Summarization options
    parser.add_argument(
        "--generate_summaries",
        action="store_true",
        help="Enable generation of file content summaries"
    )
    parser.add_argument(
        "--use_large_summarizer",
        action="store_true",
        help="Use a larger local summarization model (more accurate, less efficient)"
    )
    parser.add_argument(
        "--use_gemini",
        action="store_true",
        help="Use the Google Gemini API for summarization"
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        help="Google Gemini API key (or set the GOOGLE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-1.5-flash",
        help="Google Gemini model to use (default: gemini-1.5-flash)"
    )

    return parser.parse_args()


def read_file_content(file_path: str) -> str:
    """
    Safely read file content with error handling.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"[File Error] Failed to read {file_path}: {e}")
        return ""


def initialize_summarizer(args: Namespace):
    """
    Create a summarizer instance based on CLI arguments.
    """
    if not args.generate_summaries:
        return None

    api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")

    if args.use_gemini:
        if not api_key:
            print(
                "Error: Google Gemini API key must be provided via "
                "--gemini_api_key or GOOGLE_API_KEY environment variable."
            )
            return None

        print(f"Using Google Gemini API with model: {args.gemini_model}")
        return create_summarizer(
            use_api=True,
            api_key=api_key,
            api_model_name=args.gemini_model
        )

    model_type = "large" if args.use_large_summarizer else "default small"
    print(f"Using {model_type} summarization model...")

    return create_summarizer(use_large_model=args.use_large_summarizer)


def interactive_query_loop(rag_system, repo_files: list, args: Namespace):
    """
    Continuously prompt user for queries and return relevant files.
    """
    query_expander = QueryExpander()

    print("\nEnter your questions about the codebase (type 'exit' to quit):\n")
    while True:
        user_query = input("Your question: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        expanded_queries = query_expander.expand_query(user_query)
        all_results = []

        for expanded_query in expanded_queries:
            results = rag_system.advanced_retrieve(
                expanded_query,
                repo_files,
                args.top_k
            )
            all_results.extend(results)

        # Deduplicate results and keep top_k
        unique_results = list(dict.fromkeys(all_results))[:args.top_k]

        print(f"\nTop {args.top_k} relevant files for: \"{user_query}\"")
        for file_path in unique_results:
            print(f"• {file_path}")

            content = read_file_content(file_path)

            if args.generate_summaries and rag_system.summarizer:
                summary = rag_system.summarizer.generate_summary(content, file_path)
                print(f"Summary: {summary}\n")
            else:
                print()

        print("-" * 60)


def main():
    """
    Main execution function for the Advanced CodeRAG system.
    """
    try:
        args = parse_args()

        # Step 1: Initialize summarizer if needed
        summarizer = initialize_summarizer(args)

        # Step 2: Clone and index the repository
        repo_path = clone_repository(args.repo_url, args.repo_path)
        if not repo_path:
            print("Repository cloning failed.")
            return

        repo_files = prepare_repository_files(repo_path)
        if not repo_files:
            print("No files found in the repository.")
            return

        # Step 3: Initialize RAG system
        rag_system = AdvancedCodeRAGSystem(
            retrieval_strategy=args.retrieval_strategy,
            summarizer=summarizer
        )

        # Step 4: Start interactive loop
        interactive_query_loop(rag_system, repo_files, args)

    except Exception as e:
        print(f"[Fatal Error] Unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()