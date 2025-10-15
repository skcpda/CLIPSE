"""
Adaptive topical batching for CLIP training.
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import Sampler


def make_topical_indices(text_features_np, k_target=80, seed=13):
    """
    Create topical clusters from text features.
    
    Args:
        text_features_np: [N, D] l2-normalized numpy array
        k_target: Target number of clusters
        seed: Random seed for reproducibility
    
    Returns:
        clusters: List of per-cluster index lists
        stats: Dict with clustering statistics
    """
    # Adapt K to the number of distinct vectors
    uniq = np.unique(text_features_np.round(6), axis=0)
    n_dist = len(uniq)
    k = int(min(k_target, max(8, n_dist)))
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(text_features_np)
    
    # Create per-cluster index lists
    clusters = [np.where(labels == c)[0].tolist() for c in range(k)]
    
    stats = dict(k_used=k, n_distinct=n_dist)
    return clusters, stats


class TopicalBatchSampler(Sampler):
    """
    Round-robin sampler that draws from topical clusters to build batches.
    """
    
    def __init__(self, clusters, batch_size, drop_last=True):
        """
        Args:
            clusters: List of per-cluster index lists
            batch_size: Batch size
            drop_last: Whether to drop incomplete batches
        """
        self.clusters = clusters
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Flatten all indices
        self.all_indices = []
        for cluster in clusters:
            self.all_indices.extend(cluster)
        
        # Calculate number of batches
        self.num_batches = len(self.all_indices) // batch_size
        if not drop_last and len(self.all_indices) % batch_size != 0:
            self.num_batches += 1
    
    def __iter__(self):
        """Generate batch indices."""
        # Shuffle clusters for randomness
        cluster_order = list(range(len(self.clusters)))
        np.random.shuffle(cluster_order)
        
        # Round-robin sampling
        cluster_iters = [iter(cluster) for cluster in self.clusters]
        batch = []
        
        for _ in range(self.num_batches):
            batch = []
            while len(batch) < self.batch_size:
                # Try to get next item from each cluster in round-robin
                for cluster_idx in cluster_order:
                    try:
                        idx = next(cluster_iters[cluster_idx])
                        batch.append(idx)
                        if len(batch) >= self.batch_size:
                            break
                    except StopIteration:
                        # This cluster is exhausted, skip
                        continue
            
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
    
    def __len__(self):
        return self.num_batches
