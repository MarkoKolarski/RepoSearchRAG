import argparse
import traceback
from repository_utils import (
    clone_repository, prepare_repository_files, RepositoryEmbedder,
    Retriever, QueryExpander, LLMSummarizer, RAGEvaluator
)

def parse_args():
    parser = argparse.ArgumentParser(description="RAG pipeline for repository analysis")
    parser.add_argument("--repo_url", type=str, default="https://github.com/viarotel-org/escrcpy", help="GitHub repository URL")
    parser.add_argument("--repo_path", type=str, default="repository", help="Local repository path")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top search results")
    return parser.parse_args()

def main():
    try:
        # Step 1: Clone Repository
        args = parse_args()
        cloned_path = clone_repository(args.repo_url, args.repo_path)
        
        if not cloned_path:
            print("Failed to clone repository. Exiting.")
            return

        repository_files = prepare_repository_files(cloned_path)
        
        if not repository_files:
            print("No files found in the repository. Exiting.")
            return

        # Step 2: Create Embeddings
        embedder = RepositoryEmbedder()
        embedder.create_embeddings(repository_files)
        
        retriever = Retriever(embedder)
        query_expander = QueryExpander()
        llm_summarizer = LLMSummarizer()

        # Example queries for testing
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

        evaluator = RAGEvaluator(test_queries)
        retrieved_results = {}

        for query in test_queries.keys():
            try:
                # Expand queries
                expanded_queries = query_expander.generate_related_queries(query)
                results = []

                # Retrieve results for each expanded query
                for expanded_query in expanded_queries:
                    query_results = retriever.retrieve(expanded_query, args.top_k)
                    results.extend(query_results)

                # Remove duplicates and truncate
                results = list(dict.fromkeys(results))[:args.top_k]
                retrieved_results[query] = results

                print(f"\nQuery: {query}")
                
                # Process and summarize results
                for result in results:
                    try:
                        with open(result, 'r', encoding='utf-8', errors='replace') as f:
                            file_content = f.read()
                            
                            # Limit file content if extremely large
                            if len(file_content) > 1000:
                                file_content = file_content[:1000]
                            
                            summary = llm_summarizer.generate_summary(file_content)
                            print(f"File: {result}")
                            print(f"Summary: {summary}\n")
                    except Exception as file_error:
                        print(f"Error processing {result}: {file_error}")

            except Exception as query_error:
                print(f"Error processing query {query}: {query_error}")

        # Generate evaluation report
        evaluator.generate_report(retrieved_results)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()