import numpy as np
from similarity_search import (
    list_saved_embeddings,
    load_embedding,
    find_similar_embeddings,
    cosine_similarity
)

# List all saved embeddings
print("Saved embeddings:", list_saved_embeddings())

# Load test embeddings
embedding1 = load_embedding("test-embedding-1")
embedding2 = load_embedding("test-embedding-2")
embedding3 = load_embedding("test-embedding-3")

if embedding1 is None or embedding2 is None or embedding3 is None:
    print("Error: Failed to load one or more test embeddings")
    exit(1)

# Calculate similarity between embeddings
similarity_1_2 = cosine_similarity(embedding1, embedding2)
similarity_1_3 = cosine_similarity(embedding1, embedding3)
similarity_2_3 = cosine_similarity(embedding2, embedding3)

print(f"Similarity between embedding 1 and 2: {similarity_1_2:.4f}")
print(f"Similarity between embedding 1 and 3: {similarity_1_3:.4f}")
print(f"Similarity between embedding 2 and 3: {similarity_2_3:.4f}")

# Test find_similar_embeddings function
print("\nSimilar embeddings to embedding 1:")
similar_to_1 = find_similar_embeddings(embedding1, threshold=0.0)  # Set threshold to 0 to see all results
for embedding_id, score in similar_to_1:
    print(f"- {embedding_id}: {score:.4f} similarity") 