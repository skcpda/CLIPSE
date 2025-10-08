#!/usr/bin/env python3
"""
Quick test to verify SANW implementation works on cluster.
"""
import os
import sys
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test all imports work."""
    print("🧪 Testing imports...")
    try:
        from losses import clip_ce_loss, sanw_debias_loss, sanw_bandpass_loss
        from data import CLIPCollator
        from utils import CSVLogger
        from datasets import Flickr8kDataset
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_losses():
    """Test loss functions with dummy data."""
    print("🧪 Testing loss functions...")
    try:
        from losses import clip_ce_loss, sanw_debias_loss, sanw_bandpass_loss
        import torch.nn.functional as F
        
        # Create dummy data
        B = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        logits_per_image = torch.randn(B, B, device=device)
        logits_per_text = torch.randn(B, B, device=device)
        img_feats = F.normalize(torch.randn(B, 512, device=device), dim=-1)
        txt_feats = F.normalize(torch.randn(B, 512, device=device), dim=-1)
        
        # Test losses
        loss_baseline = clip_ce_loss(logits_per_image, logits_per_text)
        loss_debias, stats_debias = sanw_debias_loss(logits_per_image, logits_per_text, img_feats, txt_feats)
        loss_bandpass, stats_bandpass = sanw_bandpass_loss(logits_per_image, logits_per_text, img_feats, txt_feats)
        
        print(f"✅ Baseline loss: {loss_baseline.item():.4f}")
        print(f"✅ SANW-Debias loss: {loss_debias.item():.4f}")
        print(f"✅ SANW-Bandpass loss: {loss_bandpass.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        return False

def test_config_loading():
    """Test config loading."""
    print("🧪 Testing config loading...")
    try:
        config_path = "configs/cluster_baseline.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            print(f"✅ Config loaded: {cfg['experiment_name']}")
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Quick SANW Test on Cluster")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    tests = [
        test_imports,
        test_losses,
        test_config_loading
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n🎉 Quick test completed: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✅ All tests passed! Ready for full experiments.")
        return True
    else:
        print("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
