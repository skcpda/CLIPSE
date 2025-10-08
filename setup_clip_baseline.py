#!/usr/bin/env python3
"""
Setup script to download and test the baseline CLIP model on the cluster.
This script downloads openai/clip-vit-base-patch32 and verifies it works correctly.
"""

import torch
import transformers
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def test_clip_model():
    """Download and test the CLIP model."""
    print("🔧 Setting up CLIP baseline model...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("Using CPU")
        device = "cpu"
    
    # Download the model
    print("\n📥 Downloading CLIP model...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ Successfully downloaded CLIP model")
    except Exception as e:
        print(f"❌ Failed to download CLIP model: {e}")
        return False
    
    # Move to device
    model = model.to(device)
    
    # Test basic functionality
    print("\n🧪 Testing CLIP functionality...")
    
    # Create a simple test image (random)
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Test texts
    test_texts = ["a red image", "a blue image", "a dog", "a cat"]
    
    try:
        # Process inputs
        inputs = processor(
            text=test_texts,
            images=test_image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
        print(f"✅ Image embedding shape: {image_embeds.shape}")
        print(f"✅ Text embedding shape: {text_embeds.shape}")
        
        # Test similarity computation
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(image_embeds, text_embeds.T)
        print(f"✅ Similarity matrix shape: {similarity.shape}")
        print(f"✅ Similarity values: {similarity.cpu().numpy()}")
        
        # Test temperature scaling
        temperature = 0.07
        logits = similarity / temperature
        probs = torch.softmax(logits, dim=-1)
        print(f"✅ Softmax probabilities: {probs.cpu().numpy()}")
        
        print("\n🎉 CLIP model setup successful!")
        print("✅ Model loaded with pretrained weights")
        print("✅ Image and text encoding working")
        print("✅ Similarity computation working")
        print("✅ Temperature scaling working")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIP functionality test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_clip_model()
    if success:
        print("\n🚀 Baseline CLIP model is ready for SANW experiments!")
    else:
        print("\n💥 Setup failed - check the errors above")
