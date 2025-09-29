import random
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from torch.utils.data import Sampler
import torch

class TopicalBatchSampler(Sampler):
    """
    K-means clustering topical batch sampler for Flickr8k.
    Clusters captions by semantic similarity and samples batches from clusters.
    """
    def __init__(self, dataset, batch_size=256, k_clusters=80, p_topical=0.5, 
                 refresh_every_epochs=2, sampler_temperature=0.7, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k_clusters = k_clusters
        self.p_topical = p_topical
        self.refresh_every_epochs = refresh_every_epochs
        self.sampler_temperature = sampler_temperature
        self.seed = seed
        
        # Cache for caption embeddings
        self.caption_embeddings = None
        self.cluster_assignments = None
        self.cluster_sizes = None
        self.epoch_count = 0
        
        # Initialize clustering
        self._compute_clusters()
        
        # Generate initial batches
        self.batches = self._generate_batches()

    def _compute_clusters(self):
        """Compute k-means clustering of captions."""
        print(f"Computing k-means clustering with k={self.k_clusters}...")
        
        # Extract all captions
        captions = []
        for i in range(len(self.dataset)):
            _, caption, _ = self.dataset[i]
            captions.append(caption)
        
        # Simple text embedding (you can replace with sentence-transformers)
        # For now, use a simple bag-of-words approach
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        caption_embeddings = vectorizer.fit_transform(captions).toarray()
        
        # K-means clustering - adjust k if we have fewer unique samples
        actual_k = min(self.k_clusters, len(caption_embeddings))
        kmeans = KMeans(n_clusters=actual_k, random_state=self.seed, n_init=10)
        cluster_assignments = kmeans.fit_predict(caption_embeddings)
        
        # Store results
        self.caption_embeddings = caption_embeddings
        self.cluster_assignments = cluster_assignments
        self.actual_k = actual_k
        
        # Compute cluster sizes
        self.cluster_sizes = np.bincount(cluster_assignments, minlength=actual_k)
        print(f"Using {actual_k} clusters (requested {self.k_clusters})")
        print(f"Cluster sizes: min={self.cluster_sizes.min()}, max={self.cluster_sizes.max()}, mean={self.cluster_sizes.mean():.1f}")

    def _generate_batches(self):
        """Generate batches using topical sampling."""
        batches = []
        indices = list(range(len(self.dataset)))
        
        # Shuffle indices
        random.shuffle(indices)
        
        # Generate batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Decide whether to make this a topical batch
            if random.random() < self.p_topical:
                batch_indices = self._make_topical_batch(batch_indices)
            
            if len(batch_indices) >= self.batch_size // 2:  # Keep batch if at least half full
                batches.append(batch_indices)
        
        return batches

    def _make_topical_batch(self, batch_indices):
        """Convert a batch to be topical by sampling from a single cluster."""
        # Choose a cluster based on size (temperature-scaled)
        cluster_probs = self.cluster_sizes ** (1.0 / self.sampler_temperature)
        cluster_probs = cluster_probs / cluster_probs.sum()
        
        chosen_cluster = np.random.choice(self.actual_k, p=cluster_probs)
        
        # Get all indices in this cluster
        cluster_indices = [i for i, cluster in enumerate(self.cluster_assignments) 
                          if cluster == chosen_cluster]
        
        # If cluster is too small, allow some cross-cluster spill
        if len(cluster_indices) < self.batch_size:
            # Add some random indices from other clusters
            other_indices = [i for i, cluster in enumerate(self.cluster_assignments) 
                           if cluster != chosen_cluster]
            needed = self.batch_size - len(cluster_indices)
            spill_indices = random.sample(other_indices, min(needed, len(other_indices)))
            cluster_indices.extend(spill_indices)
        
        # Sample batch_size indices from cluster
        if len(cluster_indices) >= self.batch_size:
            return random.sample(cluster_indices, self.batch_size)
        else:
            return cluster_indices

    def __iter__(self):
        # Refresh clusters every refresh_every_epochs
        if self.epoch_count % self.refresh_every_epochs == 0:
            self._compute_clusters()
            self.batches = self._generate_batches()
        
        self.epoch_count += 1
        
        # Shuffle batches
        random.shuffle(self.batches)
        
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
