#!/usr/bin/env python3
"""
Minimal gradient test for CLIP - works within memory constraints.
"""
import os
import sys
import torch
import torch.nn.functional as F

# Set minimal threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def test_minimal_gradient_flow():
    """Test gradient flow with minimal memory usage."""
    print("=== MINIMAL GRADIENT FLOW TEST ===")
    
    # Force CPU to avoid CUDA memory issues
    device = "cpu"
    print(f"Using device: {device}")
    
    # Create minimal CLIP-like model
    print("Creating minimal model...")
    
    # Simple embedding layers
    vision_embed = torch.nn.Linear(3*224*224, 512)  # Simplified vision encoder
    text_embed = torch.nn.Linear(77*512, 512)       # Simplified text encoder  
    logit_scale = torch.nn.Parameter(torch.ones([]) * 2.6592)  # CLIP's initial logit scale
    
    model = torch.nn.Module()
    model.vision_embed = vision_embed
    model.text_embed = text_embed
    model.logit_scale = logit_scale
    
    model = model.to(device)
    
    # Create minimal batch
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    texts = torch.randn(batch_size, 77, 512, device=device)  # Simplified text tokens
    
    print(f"Batch shape - Images: {images.shape}, Texts: {texts.shape}")
    
    # Forward pass
    print("Running forward pass...")
    model.train()
    
    # Simplified forward
    img_features = vision_embed(images.view(batch_size, -1))
    txt_features = text_embed(texts.view(batch_size, -1))
    
    # Normalize features
    img_features = F.normalize(img_features, dim=-1)
    txt_features = F.normalize(txt_features, dim=-1)
    
    # Compute similarities
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * img_features @ txt_features.t()
    logits_per_text = logits_per_image.t()
    
    print(f"Logits per image shape: {logits_per_image.shape}")
    print(f"Logit scale: {logit_scale.item():.4f}")
    
    # Compute loss
    targets = torch.arange(batch_size, device=device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    loss = 0.5 * (loss_i + loss_t)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    
    vision_grad_norm = get_grad_norm(model.vision_embed)
    text_grad_norm = get_grad_norm(model.text_embed)
    logit_scale_grad = model.logit_scale.grad.item() if model.logit_scale.grad is not None else 0.0
    
    print(f"Vision gradient norm: {vision_grad_norm:.6f}")
    print(f"Text gradient norm: {text_grad_norm:.6f}")
    print(f"Logit scale gradient: {logit_scale_grad:.6f}")
    
    # Check if any parameters have gradients
    vision_has_grad = any(p.grad is not None for p in model.vision_embed.parameters())
    text_has_grad = any(p.grad is not None for p in model.text_embed.parameters())
    
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
    
    print("=== Test completed successfully ===")
    
    return {
        'vision_grad_norm': vision_grad_norm,
        'text_grad_norm': text_grad_norm,
        'logit_scale_grad': logit_scale_grad,
        'vision_has_grad': vision_has_grad,
        'text_has_grad': text_has_grad,
        'loss': loss.item()
    }

if __name__ == "__main__":
    try:
        results = test_minimal_gradient_flow()
        print(f"\n=== FINAL RESULTS ===")
        print(f"Vision gradient norm: {results['vision_grad_norm']:.6f}")
        print(f"Text gradient norm: {results['text_grad_norm']:.6f}")
        print(f"Logit scale gradient: {results['logit_scale_grad']:.6f}")
        print(f"Vision has gradients: {results['vision_has_grad']}")
        print(f"Text has gradients: {results['text_has_grad']}")
        print(f"Loss: {results['loss']:.4f}")
        
        # Analysis
        if results['vision_grad_norm'] > 0 and results['text_grad_norm'] > 0:
            print("\n✅ GRADIENT FLOW WORKING - Both encoders receiving gradients")
        elif results['logit_scale_grad'] > 0:
            print("\n⚠️  PARTIAL GRADIENT FLOW - Only logit_scale receiving gradients")
        else:
            print("\n❌ NO GRADIENT FLOW - No parameters receiving gradients")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
