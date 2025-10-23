#!/usr/bin/env python3
"""
Comprehensive evaluation script for CLIPSE experiments
Runs retrieval evaluation on all checkpoints and organizes results
"""

import os
import json
import subprocess
import pandas as pd
from pathlib import Path
import argparse

def run_evaluation(checkpoint_path, output_path, dataset_index, model_name="ViT-B-32"):
    """Run evaluation for a single checkpoint"""
    
    cmd = [
        "python", "tools/eval_flickr8k_retrieval.py",
        "--model_name", model_name,
        "--checkpoint", checkpoint_path,
        "--dataset_index", dataset_index,
        "--split", "test",
        "--out", output_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return False
    
    print(f"✅ Evaluation completed: {output_path}")
    return True

def organize_results(results_dir):
    """Organize and summarize evaluation results"""
    
    results = {}
    
    # Find all JSON result files
    for json_file in Path(results_dir).glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        method_name = json_file.stem.replace('_test', '')
        results[method_name] = data
    
    # Create summary table
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE RETRIEVAL EVALUATION RESULTS")
    print("="*80)
    
    # Image-to-Text results
    print("\n📸 IMAGE-TO-TEXT RETRIEVAL:")
    print(f"{'Method':<15} {'R@1':<8} {'R@5':<8} {'R@10':<9} {'MedR':<8} {'MeanR':<8}")
    print("-" * 70)
    
    for method, data in results.items():
        i2t = data['image_to_text']
        print(f"{method:<15} {i2t['R@1']:<8.2f} {i2t['R@5']:<8.2f} {i2t['R@10']:<9.2f} {i2t['MedR']:<8.1f} {i2t['MeanR']:<8.1f}")
    
    # Text-to-Image results
    print("\n📝 TEXT-TO-IMAGE RETRIEVAL:")
    print(f"{'Method':<15} {'R@1':<8} {'R@5':<8} {'R@10':<9} {'MedR':<8} {'MeanR':<8}")
    print("-" * 70)
    
    for method, data in results.items():
        t2i = data['text_to_image']
        print(f"{method:<15} {t2i['R@1']:<8.2f} {t2i['R@5']:<8.2f} {t2i['R@10']:<9.2f} {t2i['MedR']:<8.1f} {t2i['MeanR']:<8.1f}")
    
    # Save detailed results
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {summary_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive CLIPSE evaluation")
    parser.add_argument("--data_dir", required=True, help="Path to Flickr8k data directory")
    parser.add_argument("--checkpoints_dir", required=True, help="Path to checkpoints directory")
    parser.add_argument("--output_dir", default="results/eval", help="Output directory for results")
    parser.add_argument("--model_name", default="ViT-B-32", help="Model name")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build dataset index
    dataset_index = os.path.join(args.output_dir, "flickr8k_index_test.jsonl")
    print(f"Building dataset index...")
    
    index_cmd = [
        "python", "tools/build_flickr8k_index.py",
        "--data_dir", args.data_dir,
        "--output", dataset_index,
        "--split", "test"
    ]
    
    result = subprocess.run(index_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error building dataset index: {result.stderr}")
        return
    
    print(f"✅ Dataset index built: {dataset_index}")
    
    # Find all checkpoint directories
    checkpoint_dirs = []
    for item in os.listdir(args.checkpoints_dir):
        item_path = os.path.join(args.checkpoints_dir, item)
        if os.path.isdir(item_path):
            checkpoint_dirs.append((item, item_path))
    
    print(f"Found checkpoint directories: {[d[0] for d in checkpoint_dirs]}")
    
    # Run evaluation for each checkpoint
    for method_name, checkpoint_dir in checkpoint_dirs:
        # Look for the best checkpoint (epoch 15)
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_epoch_15.pt")
        
        if os.path.exists(checkpoint_file):
            output_path = os.path.join(args.output_dir, f"{method_name}_test.json")
            
            print(f"\n🔍 Evaluating {method_name}...")
            success = run_evaluation(checkpoint_file, output_path, dataset_index, args.model_name)
            
            if not success:
                print(f"❌ Failed to evaluate {method_name}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_file}")
    
    # Organize and summarize results
    print(f"\n📊 Organizing results...")
    organize_results(args.output_dir)
    
    print(f"\n🎉 Comprehensive evaluation completed!")

if __name__ == "__main__":
    main()
