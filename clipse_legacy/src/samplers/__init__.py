# Sampling utilities for CLIP training
from .topical_batch import make_topical_indices, TopicalBatchSampler

__all__ = ['make_topical_indices', 'TopicalBatchSampler']
