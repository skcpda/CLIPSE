#!/usr/bin/env python3
"""
Systematic cluster memory testing to identify exact OOM scenarios.
"""
import os
import sys
import torch
import psutil
import gc
from transformers import CLIPModel, CLIPProcessor

def get_memory_info():
    """Get current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def test_scenario(name, test_func):
    """Run a test scenario and report memory usage."""
    print(f"\n=== TESTING: {name} ===")
    
    # Clear memory before test
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    initial_memory = get_memory_info()
    print(f"Initial memory: {initial_memory['rss']:.1f}MB RSS, {initial_memory['vms']:.1f}MB VMS")
    
    try:
        result = test_func()
        final_memory = get_memory_info()
        print(f"Final memory: {final_memory['rss']:.1f}MB RSS, {final_memory['vms']:.1f}MB VMS")
        print(f"Memory increase: {final_memory['rss'] - initial_memory['rss']:.1f}MB")
        print(f"✅ SUCCESS: {name}")
        return result
    except Exception as e:
        final_memory = get_memory_info()
        print(f"Final memory: {final_memory['rss']:.1f}MB RSS, {final_memory['vms']:.1f}MB VMS")
        print(f"Memory increase: {final_memory['rss'] - initial_memory['rss']:.1f}MB")
        print(f"❌ FAILED: {name} - {type(e).__name__}: {e}")
        return None

def test_1_basic_torch():
    """Test 1: Basic PyTorch operations."""
    print("Testing basic PyTorch operations...")
    
    # Test tensor creation
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    z = torch.mm(x, y)
    
    return {"tensor_shape": z.shape, "memory_used": "~8MB"}

def test_2_small_model():
    """Test 2: Small neural network."""
    print("Testing small neural network...")
    
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    # Forward pass
    x = torch.randn(32, 1000)
    y = model(x)
    
    return {"model_params": sum(p.numel() for p in model.parameters()), "output_shape": y.shape}

def test_3_clip_processor_only():
    """Test 3: CLIP processor only (no model)."""
    print("Testing CLIP processor only...")
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Process some data
    images = [torch.randn(3, 224, 224) for _ in range(4)]
    texts = ["test image"] * 4
    
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    return {"input_keys": list(inputs.keys()), "input_shapes": {k: v.shape for k, v in inputs.items()}}

def test_4_clip_model_safetensors():
    """Test 4: CLIP model with safetensors."""
    print("Testing CLIP model with safetensors...")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    return {"model_loaded": True, "device": str(next(model.parameters()).device)}

def test_5_clip_model_no_safetensors():
    """Test 5: CLIP model without safetensors."""
    print("Testing CLIP model without safetensors...")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=False)
    return {"model_loaded": True, "device": str(next(model.parameters()).device)}

def test_6_clip_forward_pass():
    """Test 6: CLIP forward pass."""
    print("Testing CLIP forward pass...")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Small batch
    images = [torch.randn(3, 224, 224) for _ in range(2)]
    texts = ["test image"] * 2
    
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    
    return {
        "logits_per_image_shape": outputs.logits_per_image.shape,
        "logits_per_text_shape": outputs.logits_per_text.shape
    }

def test_7_clip_training_step():
    """Test 7: CLIP training step."""
    print("Testing CLIP training step...")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Training setup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Small batch
    images = [torch.randn(3, 224, 224) for _ in range(2)]
    texts = ["test image"] * 2
    
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    # Forward pass
    outputs = model(**inputs)
    
    # Compute loss
    targets = torch.arange(2)
    loss_i = torch.nn.functional.cross_entropy(outputs.logits_per_image, targets)
    loss_t = torch.nn.functional.cross_entropy(outputs.logits_per_text, targets)
    loss = 0.5 * (loss_i + loss_t)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return {"loss": loss.item(), "gradients_computed": True}

def test_8_large_batch():
    """Test 8: Large batch size."""
    print("Testing large batch size...")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Large batch
    batch_size = 32
    images = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    texts = ["test image"] * batch_size
    
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    
    return {"batch_size": batch_size, "output_shape": outputs.logits_per_image.shape}

def main():
    """Run all memory tests."""
    print("=== CLUSTER MEMORY TESTING ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Set minimal threading
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Test scenarios
    scenarios = [
        ("Basic PyTorch", test_1_basic_torch),
        ("Small Model", test_2_small_model),
        ("CLIP Processor Only", test_3_clip_processor_only),
        ("CLIP Model (safetensors)", test_4_clip_model_safetensors),
        ("CLIP Model (no safetensors)", test_5_clip_model_no_safetensors),
        ("CLIP Forward Pass", test_6_clip_forward_pass),
        ("CLIP Training Step", test_7_clip_training_step),
        ("Large Batch", test_8_large_batch),
    ]
    
    results = {}
    
    for name, test_func in scenarios:
        result = test_scenario(name, test_func)
        results[name] = result
        
        # Clear memory between tests
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print("\n=== SUMMARY ===")
    for name, result in results.items():
        status = "✅ PASS" if result is not None else "❌ FAIL"
        print(f"{name}: {status}")
    
    # Identify failure points
    failures = [name for name, result in results.items() if result is None]
    if failures:
        print(f"\n❌ FAILED SCENARIOS: {', '.join(failures)}")
        print("These are the exact scenarios that cause OOM on the cluster.")
    else:
        print("\n✅ ALL TESTS PASSED - No OOM issues detected.")

if __name__ == "__main__":
    main()
