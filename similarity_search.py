import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Define the embeddings directory
EMBEDDINGS_DIR = Path("embeddings")

def load_all_embeddings() -> Dict[str, np.ndarray]:
    """
    Load all saved embeddings from the embeddings directory.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping filename (without extension) to embedding vector
    """
    embeddings = {}
    
    # Create directory if it doesn't exist
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Loop through all JSON files in the embeddings directory
    for file_path in EMBEDDINGS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                embedding_list = json.load(f)
            
            # Store the embedding with its filename (without extension) as the key
            embeddings[file_path.stem] = np.array(embedding_list)
        except Exception as e:
            print(f"Error loading embedding from {file_path}: {e}")
    
    return embeddings

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (between -1 and 1)
    """
    # Normalize the embeddings
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1_norm, embedding2_norm)
    
    return float(similarity)

def find_similar_embeddings(
    query_embedding: np.ndarray, 
    top_k: int = 5,
    threshold: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Find the most similar embeddings to the query embedding.
    
    Args:
        query_embedding: The embedding to compare against
        top_k: Number of similar embeddings to return
        threshold: Minimum similarity score to include in results
    
    Returns:
        List of tuples containing (embedding_id, similarity_score)
    """
    # Load all saved embeddings
    all_embeddings = load_all_embeddings()
    
    if not all_embeddings:
        return []
    
    # Calculate similarity scores
    similarities = [
        (embedding_id, cosine_similarity(query_embedding, embedding))
        for embedding_id, embedding in all_embeddings.items()
    ]
    
    # Sort by similarity score (highest first) and filter by threshold
    similarities = [
        (embedding_id, score) 
        for embedding_id, score in similarities 
        if score >= threshold
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return similarities[:top_k]

def save_embedding(embedding: np.ndarray, embedding_id: str) -> str:
    """
    Save an embedding to the embeddings directory.
    
    Args:
        embedding: The embedding vector to save
        embedding_id: Identifier for the embedding
    
    Returns:
        Path where the embedding was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Create the file path
    file_path = EMBEDDINGS_DIR / f"{embedding_id}.json"
    
    # Convert numpy array to list for JSON serialization
    embedding_list = embedding.tolist()
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(embedding_list, f)
    
    return str(file_path)

def load_embedding(embedding_id: str) -> Optional[np.ndarray]:
    """
    Load a specific embedding by ID.
    
    Args:
        embedding_id: The ID of the embedding to load
    
    Returns:
        The embedding vector or None if not found
    """
    file_path = EMBEDDINGS_DIR / f"{embedding_id}.json"
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            embedding_list = json.load(f)
        
        return np.array(embedding_list)
    except Exception as e:
        print(f"Error loading embedding {embedding_id}: {e}")
        return None

def list_saved_embeddings() -> List[str]:
    """
    List all saved embedding IDs.
    
    Returns:
        List of embedding IDs (filenames without extension)
    """
    # Create directory if it doesn't exist
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Return list of filenames without extension
    return [file_path.stem for file_path in EMBEDDINGS_DIR.glob("*.json")]

if __name__ == "__main__":
    # Example usage
    print("Saved embeddings:", list_saved_embeddings()) 