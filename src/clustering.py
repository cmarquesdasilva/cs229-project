import faiss
import numpy as np
import pandas as pd

def create_clusters(embeddings, texts, prompt_toxicity, num_clusters=3, device='cpu'):
    """Create clusters for the embeddings using FAISS Kmeans"""
    d = embeddings.shape[1] 
    prompt_toxicity = np.array(prompt_toxicity, dtype=float)
    
    #Initialize KMeans clustering in FAISS
    kmeans = faiss.Kmeans(d, num_clusters, gpu=(device == 'cuda'))

    # Normalize embeddings if using cosine similarity
    embeddings_np = embeddings.cpu().numpy()

    faiss.normalize_L2(embeddings_np)
    
    # Train the KMeans clustering
    kmeans.train(embeddings_np)

    # Assign each embedding to the nearest cluster center
    _, cluster_assignments = kmeans.index.search(embeddings_np, 1)  

    # Flatten the cluster assignments to get a 1D array of cluster labels
    cluster_assignments = cluster_assignments.flatten()
    # Check that all arrays are of the same length before creating the DataFrame
    if len(texts) != len(prompt_toxicity) or len(texts) != len(cluster_assignments):
        raise ValueError(f"Length mismatch: texts ({len(texts)}), prompt_toxicity ({len(prompt_toxicity)}), cluster_assignments ({len(cluster_assignments)})")
    
    # Create a DataFrame with the cluster assignments
    df = pd.DataFrame({
        'text': texts,
        'prompt_toxicity': prompt_toxicity,
        'cluster': cluster_assignments
    })

    return df