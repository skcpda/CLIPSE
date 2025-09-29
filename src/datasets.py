import os, random, csv
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

class FlickrTSV(Dataset):
    """
    Expects a TSV file with lines: relative_image_path \t caption
    root/
      Images/...
      captions.tsv
    """
    def __init__(self, root: str, tsv: str, split: str = "train", val_ratio=0.1, transform=None, seed=42):
        self.root = root
        self.tsv_path = os.path.join(root, tsv)
        self.transform = transform
        rng = random.Random(seed)

        pairs = []
        with open(self.tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2: continue
                img_rel, cap = row[0].strip(), row[1].strip()
                pairs.append((img_rel, cap))

        rng.shuffle(pairs)
        n_val = int(len(pairs) * val_ratio)
        self.val = pairs[:n_val]
        self.train = pairs[n_val:]

        self.split_pairs = self.train if split == "train" else self.val

        self.preprocess = transform or transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self): return len(self.split_pairs)

    def __getitem__(self, idx):
        img_rel, caption = self.split_pairs[idx]
        img_path = os.path.join(self.root, img_rel)
        
        try:
            # Try to open the image
            with open(img_path, 'rb') as f:
                # Check if it's a real image file
                if f.read(2) == b'\xff\xd8':  # JPEG magic number
                    f.seek(0)
                    image = Image.open(f).convert("RGB")
                else:
                    # Not a real image, create dummy
                    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        except (IOError, OSError):
            # If image doesn't exist or can't be opened, create a dummy image
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        image = self.preprocess(image)
        return image, caption, img_rel

def collate_text_tokenize(batch, tokenizer):
    images, texts, ids = zip(*batch)
    images = torch.stack(images, dim=0)
    tokenized = tokenizer(list(texts))
    # Ensure tokenized is a tensor (not dict) for OpenCLIP
    if isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    return images, tokenized, ids
