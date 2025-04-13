import numpy as np
import json
from pathlib import Path

# Ensure embeddings directory exists
embeddings_dir = Path("/Users/jonathanpolitzki/Desktop/Coding/mcp-text-to-embedding/embeddings")
embeddings_dir.mkdir(exist_ok=True)

# Create first test embedding (1536 dimensions like OpenAI's models)
embedding1 = np.random.rand(1536)
file_path1 = embeddings_dir / "test-embedding-1.json"
with open(file_path1, 'w') as f:
    json.dump(embedding1.tolist(), f)
print(f"Test embedding 1 created at {file_path1}")

# Create second test embedding 
embedding2 = np.random.rand(1536)
# Make it somewhat similar to the first one
embedding2 = 0.7 * embedding1 + 0.3 * embedding2
file_path2 = embeddings_dir / "test-embedding-2.json"
with open(file_path2, 'w') as f:
    json.dump(embedding2.tolist(), f)
print(f"Test embedding 2 created at {file_path2}")

# Create a third test embedding (more different)
embedding3 = np.random.rand(1536)
file_path3 = embeddings_dir / "test-embedding-3.json"
with open(file_path3, 'w') as f:
    json.dump(embedding3.tolist(), f)
print(f"Test embedding 3 created at {file_path3}") 