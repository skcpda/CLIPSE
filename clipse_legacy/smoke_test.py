#!/usr/bin/env python3
"""
Quick smoke test for the SANW implementation.
"""
import os
import sys
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from losses import clip_ce_loss, sanw_debias_loss, sanw_bandpass_loss
from data import CLIPCollator
from utils import CSVLogger


def test_losses():
    """Test all loss functions."""
    print("🧪 Testing loss functions...")
    
    # Create dummy data
    B = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dummy logits
    logits_per_image = torch.randn(B, B, device=device)
    logits_per_text = torch.randn(B, B, device=device)
    
    # Dummy features
    img_feats = F.normalize(torch.randn(B, 512, device=device), dim=-1)
    txt_feats = F.normalize(torch.randn(B, 512, device=device), dim=-1)
    
    # Test baseline loss
    loss_baseline = clip_ce_loss(logits_per_image, logits_per_text)
    print(f"✅ Baseline loss: {loss_baseline.item():.4f}")
    
    # Test SANW-Debias loss
    loss_debias, stats_debias = sanw_debias_loss(
        logits_per_image, logits_per_text, img_feats, txt_feats
    )
    print(f"✅ SANW-Debias loss: {loss_debias.item():.4f}")
    print(f"   Stats: {stats_debias}")
    
    # Test SANW-Bandpass loss
    loss_bandpass, stats_bandpass = sanw_bandpass_loss(
        logits_per_image, logits_per_text, img_feats, txt_feats
    )
    print(f"✅ SANW-Bandpass loss: {loss_bandpass.item():.4f}")
    print(f"   Stats: {stats_bandpass}")
    
    return True


def test_collate():
    """Test CLIP collate function."""
    print("\n🧪 Testing CLIP collate...")
    
    # Create dummy processor
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        collate = CLIPCollator(processor)
        
        # Create dummy batch
        batch = []
        for i in range(4):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            text = f"test image {i}"
            batch.append({"image": img, "text": text})
        
        # Test collate
        result = collate(batch)
        print(f"✅ Collate result keys: {list(result.keys())}")
        print(f"   Pixel values shape: {result['pixel_values'].shape}")
        print(f"   Input IDs shape: {result['input_ids'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Collate test failed: {e}")
        return False


def test_logging():
    """Test CSV logging."""
    print("\n🧪 Testing CSV logging...")
    
    try:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        logger = CSVLogger("logs/test_log.csv")
        logger.log_step(epoch=1, step=1, loss=0.5, temp=0.07, lr=1e-4, w_mean=0.3)
        logger.log_epoch(epoch=1, val_r1=0.8, val_r5=0.9, val_r10=0.95)
        print("✅ CSV logging works")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("🚀 Starting SANW smoke tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    tests = [
        test_losses,
        test_collate,
        test_logging
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n🎉 Smoke tests completed: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✅ All tests passed! SANW implementation is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
