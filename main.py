import argparse
import traceback
from argparse import Namespace
from repository_utils import (
    AdvancedCodeRAGSystem,
    clone_repository,
    prepare_repository_files,
    interactive_query_loop,
    initialize_summarizer
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