#!/usr/bin/env python3
"""
Simple CLIP test script that tries to use existing packages or install minimal requirements.
"""

import sys
import subprocess
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
        return True
    except subprocess.CalledProcessError:
        return False

def test_clip():
    """Test CLIP functionality."""
    print("🔧 Testing CLIP setup...")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Try to import required packages
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("✅ Transformers available")
    except ImportError:
        print("❌ Transformers not available")
        return False
    
    # Test basic CLIP functionality
    try:
        print("\n🧪 Testing CLIP model loading...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP model loaded successfully")
        
        # Test basic functionality
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='red')
        test_texts = ["a red image", "a blue image"]
        
        # Process inputs
        inputs = processor(
            text=test_texts,
            images=test_image,
            return_tensors="pt",
            padding=True
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
        print(f"✅ Image embedding shape: {image_embeds.shape}")
        print(f"✅ Text embedding shape: {text_embeds.shape}")
        
        # Test similarity computation
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        similarity = torch.matmul(image_embeds, text_embeds.T)
        print(f"✅ Similarity matrix shape: {similarity.shape}")
        print(f"✅ Similarity values: {similarity.cpu().numpy()}")
        
        print("\n🎉 CLIP baseline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_clip()
    if success:
        print("\n🚀 CLIP baseline is ready!")
    else:
        print("\n💥 CLIP test failed")
        sys.exit(1)
