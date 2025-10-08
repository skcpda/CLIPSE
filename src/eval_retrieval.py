"""
Retrieval evaluation for CLIP models.
"""
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm


def build_model(local_model_dir: str):
    """Build CLIP model and processor from local directory."""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = os.path.dirname(local_model_dir)
    
    model = CLIPModel.from_pretrained(local_model_dir, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(local_model_dir, local_files_only=True)
    
    return model, processor


def evaluate_retrieval(model, processor, test_loader, device, k_values=[1, 5, 10]):
    """
    Evaluate image-text retrieval performance.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        test_loader: DataLoader with test data
        device: Device to run on
        k_values: List of k values for R@k evaluation
    
    Returns:
        Dict with R@k scores
    """
    model.eval()
    
    all_image_features = []
    all_text_features = []
    all_labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get features
            image_features = model.get_image_features(batch["pixel_values"])
            text_features = model.get_text_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            
            # Create labels (assuming batch contains ground truth)
            batch_size = image_features.size(0)
            labels = torch.arange(batch_size)
            all_labels.append(labels)
    
    # Concatenate all features
    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print("Computing similarities...")
    # Compute similarity matrix
    similarity = torch.matmul(image_features, text_features.T)
    
    # Image-to-text retrieval
    print("Evaluating image-to-text retrieval...")
    i2t_scores = {}
    for k in k_values:
        _, indices = torch.topk(similarity, k, dim=1)
        correct = (indices == labels.unsqueeze(1)).any(dim=1).float().mean()
        i2t_scores[f"i2t_r@{k}"] = correct.item()
    
    # Text-to-image retrieval
    print("Evaluating text-to-image retrieval...")
    t2i_scores = {}
    for k in k_values:
        _, indices = torch.topk(similarity.T, k, dim=1)
        correct = (indices == labels.unsqueeze(1)).any(dim=1).float().mean()
        t2i_scores[f"t2i_r@{k}"] = correct.item()
    
    # Combine scores
    scores = {**i2t_scores, **t2i_scores}
    
    return scores


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CLIP model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--model_dir", type=str, 
                       default="/nfs_home/users/poonam/sanw_experiments/data/clip_models/clip-vit-base-patch32",
                       help="Path to model directory")
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, processor = build_model(args.model_dir)
    model = model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # TODO: Replace with actual test dataset
    # For now, create dummy test data
    from torch.utils.data import Dataset, DataLoader
    class DummyTestDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            from PIL import Image
            import numpy as np
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            text = f"test image {idx}"
            return {"image": img, "text": text}
    
    test_dataset = DummyTestDataset(size=100)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    print("Starting evaluation...")
    scores = evaluate_retrieval(model, processor, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
