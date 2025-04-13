import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def generate_embedding(text, model="text-embedding-3-small"):
    """
    Generate an embedding for the provided text using OpenAI's embedding model.
    
    Args:
        text (str): The input text to generate embeddings for
        model (str): The OpenAI embedding model to use
        
    Returns:
        np.ndarray: The embedding vector as a numpy array
    """
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

def save_embedding(embedding, filename="embedding.json"):
    """
    Save an embedding to a JSON file.
    
    Args:
        embedding (np.ndarray): The embedding vector
        filename (str): The path to save the embedding to
    """
    # Convert numpy array to list for JSON serialization
    embedding_list = embedding.tolist()
    
    with open(filename, 'w') as f:
        json.dump(embedding_list, f)
    
    print(f"Embedding saved to {filename}")

def load_embedding(filename="embedding.json"):
    """
    Load an embedding from a JSON file.
    
    Args:
        filename (str): The path to load the embedding from
        
    Returns:
        np.ndarray: The embedding vector as a numpy array
    """
    try:
        with open(filename, 'r') as f:
            embedding_list = json.load(f)
        
        return np.array(embedding_list)
    except Exception as e:
        print(f"Error loading embedding: {e}")
        return None

def embedding_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.
    
    Args:
        embedding1 (np.ndarray): The first embedding
        embedding2 (np.ndarray): The second embedding
        
    Returns:
        float: The cosine similarity (between -1 and 1)
    """
    # Normalize the embeddings
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1_norm, embedding2_norm)
    
    return similarity

if __name__ == "__main__":
    # Example usage
    while True:
        user_input = input("\nEnter text to generate embedding (or 'q' to quit): ")
        
        if user_input.lower() == 'q':
            break
        
        embedding = generate_embedding(user_input)
        
        if embedding is not None:
            print(f"Generated embedding with shape: {embedding.shape}")
            
            # Ask if user wants to save embedding
            save_choice = input("Save embedding? (y/n): ")
            if save_choice.lower() == 'y':
                filename = input("Enter filename (default: embedding.json): ") or "embedding.json"
                save_embedding(embedding, filename) 