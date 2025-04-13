"""
MCP Server for Text-to-Embedding Service

This file implements a Model Context Protocol server that exposes tools
to generate embeddings from text and conduct similarity searches using OpenAI's embedding models.

Requirements:
- Python 3.10+
- mcp package: pip install "mcp[cli]"
- openai package: pip install openai
- python-dotenv: pip install python-dotenv
- numpy: pip install numpy

To run this server:
python mcp_server.py
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Import similarity search functionality
from similarity_search import (
    find_similar_embeddings,
    save_embedding,
    load_embedding,
    list_saved_embeddings,
)

# Import MCP
from mcp.server.fastmcp import FastMCP, Context

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Create the MCP server
mcp = FastMCP("Text-to-Embedding")

def generate_embedding(text, model="text-embedding-3-small"):
    """Generate an embedding for the provided text using OpenAI's embedding model."""
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        
        # Extract the embedding from the response
        embedding = response.data[0].embedding
        
        return np.array(embedding)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

@mcp.tool()
def text_to_embedding(text: str, embedding_id: str, model: str = "text-embedding-3-small") -> str:
    """Convert text to embeddings and save it with the given ID.
    
    Args:
        text: The input text to generate embeddings for
        embedding_id: Identifier to save the embedding with
        model: The embedding model to use (default: text-embedding-3-small)
    
    Returns:
        A message confirming the embedding was saved
    """
    # Generate the embedding
    embedding = generate_embedding(text, model)
    
    if embedding is None:
        return "Error: Failed to generate embedding"
    
    # Save the embedding
    file_path = save_embedding(embedding, embedding_id)
    
    return f"Embedding saved with ID: {embedding_id}"


@mcp.tool()
def similarity_search(
    query_text: str, 
    top_k: int = 5, 
    threshold: float = 0.7,
    model: str = "text-embedding-3-small"
) -> str:
    """Find similar embeddings to the query text.
    
    Args:
        query_text: Text to find similar embeddings for
        top_k: Number of similar embeddings to return
        threshold: Minimum similarity score (0.0-1.0)
        model: The embedding model to use
    
    Returns:
        A formatted string describing the similar embeddings
    """
    # Generate the query embedding
    query_embedding = generate_embedding(query_text, model)
    
    if query_embedding is None:
        return "Error: Failed to generate embedding for query text"
    
    # Find similar embeddings
    similar_embeddings = find_similar_embeddings(
        query_embedding, 
        top_k=top_k,
        threshold=threshold
    )
    
    if not similar_embeddings:
        return "No similar embeddings found. Try lowering the threshold or adding more embeddings."
    
    # Format the results
    results = ["Similar embeddings found:"]
    for embedding_id, score in similar_embeddings:
        results.append(f"- {embedding_id}: {score:.4f} similarity")
    
    return "\n".join(results)


@mcp.tool()
def compare_texts(
    text1: str, 
    text2: str, 
    model: str = "text-embedding-3-small"
) -> str:
    """Compare the semantic similarity between two text strings.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        model: The embedding model to use
    
    Returns:
        Similarity score and interpretation
    """
    # Generate embeddings
    embedding1 = generate_embedding(text1, model)
    embedding2 = generate_embedding(text2, model)
    
    if embedding1 is None or embedding2 is None:
        return "Error: Failed to generate embeddings"
    
    # Calculate cosine similarity
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    similarity = np.dot(embedding1_norm, embedding2_norm)
    
    # Interpret the similarity score
    interpretation = ""
    if similarity > 0.9:
        interpretation = "The texts are very similar in meaning"
    elif similarity > 0.7:
        interpretation = "The texts are moderately similar"
    elif similarity > 0.5:
        interpretation = "The texts are somewhat similar"
    else:
        interpretation = "The texts are not very similar"
    
    return f"Similarity score: {similarity:.4f}\n{interpretation}"


@mcp.tool()
def list_embeddings() -> str:
    """List all saved embeddings in the repository.
    
    Returns:
        A list of all embedding IDs
    """
    embeddings = list_saved_embeddings()
    
    if not embeddings:
        return "No embeddings found in the repository"
    
    result = ["Saved embeddings:"]
    for embedding_id in embeddings:
        result.append(f"- {embedding_id}")
    
    return "\n".join(result)


@mcp.tool()
def delete_embedding(embedding_id: str) -> str:
    """Delete an embedding from the repository.
    
    Args:
        embedding_id: ID of the embedding to delete
    
    Returns:
        Confirmation message
    """
    file_path = Path(f"embeddings/{embedding_id}.json")
    
    if not file_path.exists():
        return f"Embedding with ID '{embedding_id}' not found"
    
    try:
        os.remove(file_path)
        return f"Embedding with ID '{embedding_id}' successfully deleted"
    except Exception as e:
        return f"Error deleting embedding: {e}"

if __name__ == "__main__":
    try:
        print("Starting Text-to-Embedding MCP server...")
        mcp.run()
    except ImportError:
        print("Error: Python 3.10+ and the MCP package are required.")
        print("Please upgrade your Python version and install the MCP package:")
        print("  pip install 'mcp[cli]'")
        print("\nFor now, you can use the embedding_cli.py script directly.") 