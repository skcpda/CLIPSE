#!/usr/bin/env python3
"""
Simple gradient diagnostic script for CLIP training.
"""
import os
import sys
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

def test_gradient_flow():
    """Test if gradients flow to CLIP encoders."""
    print("=== CLIP Gradient Flow Test ===")
    
    # Set environment
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Load model
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Create dummy batch
    batch_size = 4
    images = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    texts = ["A test image"] * batch_size
    
    # Process batch
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Batch shape: {inputs['pixel_values'].shape}")
    
    # Forward pass
    print("Running forward pass...")
    model.train()
    outputs = model(**inputs)
    
    # Check outputs
    print(f"Logits per image shape: {outputs.logits_per_image.shape}")
    print(f"Logits per text shape: {outputs.logits_per_text.shape}")
    
    # Compute loss
    targets = torch.arange(batch_size, device=device)
    loss_i = F.cross_entropy(outputs.logits_per_image, targets)
    loss_t = F.cross_entropy(outputs.logits_per_text, targets)
    loss = 0.5 * (loss_i + loss_t)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check specific components
    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    text_params = sum(p.numel() for p in model.text_model.parameters())
    logit_scale_params = model.logit_scale.numel()
    
    print(f"Vision model parameters: {vision_params:,}")
    print(f"Text model parameters: {text_params:,}")
    print(f"Logit scale parameters: {logit_scale_params}")
    
    # Backward pass
    print("Running backward pass...")
    loss.backward()
    
    # Check gradients
    def get_grad_norm(module):
        total_norm = 0.0
        param_count = 0
        for p in module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    vision_grad_norm = get_grad_norm(model.vision_model)
    text_grad_norm = get_grad_norm(model.text_model)
    logit_scale_grad = model.logit_scale.grad.item() if model.logit_scale.grad is not None else 0.0
    
    print(f"Vision gradient norm: {vision_grad_norm:.6f}")
    print(f"Text gradient norm: {text_grad_norm:.6f}")
    print(f"Logit scale gradient: {logit_scale_grad:.6f}")
    
    # Check if any parameters have gradients
    vision_has_grad = any(p.grad is not None for p in model.vision_model.parameters())
    text_has_grad = any(p.grad is not None for p in model.text_model.parameters())
    
    print(f"Vision has gradients: {vision_has_grad}")
    print(f"Text has gradients: {text_has_grad}")
    
    # Test optimizer
    print("Testing optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"Optimizer param groups: {len(optimizer.param_groups)}")
    for i, group in enumerate(optimizer.param_groups):
        param_count = sum(p.numel() for p in group['params'])
        print(f"Group {i}: {param_count:,} parameters, lr={group['lr']}")
    
    # Test step
    optimizer.step()
    optimizer.zero_grad()
    
    print("=== Test completed ===")
    
    return {
        'vision_grad_norm': vision_grad_norm,
        'text_grad_norm': text_grad_norm,
        'logit_scale_grad': logit_scale_grad,
        'vision_has_grad': vision_has_grad,
        'text_has_grad': text_has_grad
    }

if __name__ == "__main__":
    try:
        results = test_gradient_flow()
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
