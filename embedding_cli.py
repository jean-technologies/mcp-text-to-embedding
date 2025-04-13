#!/usr/bin/env python3
"""
Command-line interface for working with text embeddings.
This tool allows you to:
- Generate embeddings from text
- Save embeddings to the repository
- Compare texts for semantic similarity
- Search for similar embeddings
"""

import os
import sys
import argparse
import numpy as np
from dotenv import load_dotenv
from text_to_embedding import generate_embedding
from similarity_search import (
    find_similar_embeddings,
    save_embedding,
    load_embedding,
    list_saved_embeddings,
    cosine_similarity
)

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Text embedding tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate embedding command
    gen_parser = subparsers.add_parser("generate", help="Generate an embedding from text")
    gen_parser.add_argument("text", help="Text to embed")
    gen_parser.add_argument("--id", help="ID to save the embedding with")
    gen_parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model to use")
    
    # List embeddings command
    list_parser = subparsers.add_parser("list", help="List all saved embeddings")
    
    # Compare texts command
    compare_parser = subparsers.add_parser("compare", help="Compare two texts for similarity")
    compare_parser.add_argument("text1", help="First text to compare")
    compare_parser.add_argument("text2", help="Second text to compare")
    compare_parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model to use")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar embeddings")
    search_parser.add_argument("query", help="Query text to search for")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--threshold", type=float, default=0.7, help="Minimum similarity score (0-1)")
    search_parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model to use")
    
    # Delete embedding command
    delete_parser = subparsers.add_parser("delete", help="Delete an embedding")
    delete_parser.add_argument("id", help="ID of the embedding to delete")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    if args.command == "generate":
        # Generate and optionally save an embedding
        embedding = generate_embedding(args.text, args.model)
        
        if embedding is None:
            print("Error: Failed to generate embedding")
            return 1
        
        print(f"Generated embedding with {len(embedding)} dimensions")
        
        if args.id:
            file_path = save_embedding(embedding, args.id)
            print(f"Saved embedding as {args.id}")
        
    elif args.command == "list":
        # List all saved embeddings
        embeddings = list_saved_embeddings()
        
        if not embeddings:
            print("No embeddings found")
            return 0
        
        print(f"Found {len(embeddings)} embeddings:")
        for embedding_id in embeddings:
            print(f"- {embedding_id}")
    
    elif args.command == "compare":
        # Compare two texts for similarity
        embedding1 = generate_embedding(args.text1, args.model)
        embedding2 = generate_embedding(args.text2, args.model)
        
        if embedding1 is None or embedding2 is None:
            print("Error: Failed to generate embeddings")
            return 1
        
        similarity = cosine_similarity(embedding1, embedding2)
        
        print(f"Similarity: {similarity:.4f}")
        
        # Interpret the similarity score
        if similarity > 0.9:
            print("The texts are very similar in meaning")
        elif similarity > 0.7:
            print("The texts are moderately similar")
        elif similarity > 0.5:
            print("The texts are somewhat similar")
        else:
            print("The texts are not very similar")
    
    elif args.command == "search":
        # Search for similar embeddings
        query_embedding = generate_embedding(args.query, args.model)
        
        if query_embedding is None:
            print("Error: Failed to generate embedding for query text")
            return 1
        
        similar_embeddings = find_similar_embeddings(
            query_embedding,
            top_k=args.top_k,
            threshold=args.threshold
        )
        
        if not similar_embeddings:
            print("No similar embeddings found. Try lowering the threshold or adding more embeddings.")
            return 0
        
        print(f"Found {len(similar_embeddings)} similar embeddings:")
        for embedding_id, score in similar_embeddings:
            print(f"- {embedding_id}: {score:.4f} similarity")
    
    elif args.command == "delete":
        # Delete an embedding
        file_path = os.path.join("embeddings", f"{args.id}.json")
        
        if not os.path.exists(file_path):
            print(f"Embedding with ID '{args.id}' not found")
            return 1
        
        try:
            os.remove(file_path)
            print(f"Embedding with ID '{args.id}' successfully deleted")
        except Exception as e:
            print(f"Error deleting embedding: {e}")
            return 1
    
    else:
        print("Please specify a command. Use --help for more information.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 