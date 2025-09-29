import random
from collections import defaultdict
from torch.utils.data import Sampler

def extract_topic_from_caption(text: str, mode="noun"):
    # very simple heuristic: lowercase tokens; pick first nouny-looking word
    # you can replace with spaCy/NP chunking if you like
    import re
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    for t in toks:
        # toy filter of common nouns; loosen as needed
        if len(t) > 3:
            return t
    return toks[0] if toks else "misc"

class TopicalBatchSampler(Sampler):
    """
    Groups indices into batches where captions share a coarse topic token.
    Good for stressing false-negative scenario.
    """
    def __init__(self, dataset, batch_size=64, key="noun", shuffle=True, seed=42):
        self.ds = dataset
        self.bs = batch_size
        self.key = key
        self.shuffle = shuffle
        random.seed(seed)

        # bucket indices by topic
        buckets = defaultdict(list)
        for i, (_, cap, _) in enumerate(self.ds.split_pairs):
            topic = extract_topic_from_caption(cap, mode=key)
            buckets[topic].append(i)

        self.batches = []
        for topic, idxs in buckets.items():
            if self.shuffle:
                random.shuffle(idxs)
            for k in range(0, len(idxs), self.bs):
                chunk = idxs[k:k+self.bs]
                if len(chunk) >= max(4, self.bs//2):
                    self.batches.append(chunk)

        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)
