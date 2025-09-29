import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms

class DummyFlickrDataset(Dataset):
    """
    Dummy dataset for testing - generates random images and captions
    """
    def __init__(self, num_samples=100, split="train", val_ratio=0.1, transform=None, seed=42):
        self.split = split
        self.transform = transform
        random.seed(seed)
        
        # Generate dummy data
        self.samples = []
        topics = ["dog", "cat", "flower", "car", "person", "building", "tree", "bird"]
        
        for i in range(num_samples):
            topic = random.choice(topics)
            caption = f"A photo of a {topic}"
            # Create dummy image path
            img_path = f"dummy_{i:03d}.jpg"
            self.samples.append((img_path, caption, topic))
        
        # Split train/val
        random.shuffle(self.samples)
        n_val = int(len(self.samples) * val_ratio)
        self.val = self.samples[:n_val]
        self.train = self.samples[n_val:]
        
        self.split_samples = self.train if split == "train" else self.val
        
        # Default transform for dummy data
        self.preprocess = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.split_samples)

    def __getitem__(self, idx):
        img_path, caption, topic = self.split_samples[idx]
        
        # Generate random image (RGB, 224x224) - already normalized
        image = torch.randn(3, 224, 224)
        # Apply normalization directly
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image, caption, img_path
