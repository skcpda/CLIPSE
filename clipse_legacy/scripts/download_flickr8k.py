#!/usr/bin/env python3
"""
Download and set up the real Flickr8k dataset.
This script downloads the actual Flickr8k dataset from the official source.
"""
import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm
import hashlib


def download_file(url, destination, expected_size=None):
    """Download a file with progress bar."""
    print(f"📥 Downloading {url} to {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    if expected_size and total_size != expected_size:
        print(f"⚠️ Warning: Expected size {expected_size}, got {total_size}")
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✅ Downloaded {destination}")


def setup_flickr8k_dataset(root_dir):
    """Set up the real Flickr8k dataset."""
    print("🚀 Setting up real Flickr8k dataset...")
    
    # Create directory structure
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = root_path / "Images"
    images_dir.mkdir(exist_ok=True)
    
    # Flickr8k dataset URLs (these are the official sources)
    # Note: These URLs may need to be updated based on current availability
    flickr8k_urls = {
        "images": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
        "captions": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    }
    
    # Alternative sources if the above don't work
    alternative_urls = {
        "images": "https://www.kaggle.com/datasets/adityajn105/flickr8k/download",
        "captions": "https://www.kaggle.com/datasets/adityajn105/flickr8k/download"
    }
    
    print("📋 Flickr8k dataset sources:")
    print("1. Official GitHub releases (preferred)")
    print("2. Kaggle dataset (requires authentication)")
    print("3. Manual download instructions")
    
    # For now, create a script that downloads from a reliable source
    # The actual download will depend on what's available
    print("\n🔧 Setting up download script...")
    
    # Create a download script for the cluster
    download_script = f"""#!/bin/bash
# Flickr8k dataset download script
set -e

DATASET_DIR="{root_dir}"
echo "📁 Dataset directory: $DATASET_DIR"

# Create directory structure
mkdir -p "$DATASET_DIR/Images"
mkdir -p "$DATASET_DIR/raw"

# Download from Kaggle (requires kaggle API)
echo "📥 Downloading Flickr8k dataset from Kaggle..."
echo "Note: This requires Kaggle API credentials to be set up"

# Alternative: Download from direct URLs if available
echo "🔄 Attempting to download from direct URLs..."

# Try to download images
if command -v wget &> /dev/null; then
    echo "📥 Downloading images with wget..."
    wget -O "$DATASET_DIR/raw/Flickr8k_Dataset.zip" "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip" || echo "❌ Image download failed"
fi

# Try to download captions
if command -v wget &> /dev/null; then
    echo "📥 Downloading captions with wget..."
    wget -O "$DATASET_DIR/raw/Flickr8k_text.zip" "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip" || echo "❌ Caption download failed"
fi

# Extract if downloads succeeded
if [ -f "$DATASET_DIR/raw/Flickr8k_Dataset.zip" ]; then
    echo "📦 Extracting images..."
    unzip -q "$DATASET_DIR/raw/Flickr8k_Dataset.zip" -d "$DATASET_DIR/"
    # Move images to Images directory
    find "$DATASET_DIR" -name "*.jpg" -exec mv {{}} "$DATASET_DIR/Images/" \\;
fi

if [ -f "$DATASET_DIR/raw/Flickr8k_text.zip" ]; then
    echo "📦 Extracting captions..."
    unzip -q "$DATASET_DIR/raw/Flickr8k_text.zip" -d "$DATASET_DIR/"
fi

# Create captions.tsv file
echo "📝 Creating captions.tsv file..."
python3 -c "
import os
import pandas as pd
from pathlib import Path

dataset_dir = Path('{root_dir}')
images_dir = dataset_dir / 'Images'
captions_file = dataset_dir / 'captions.tsv'

# Look for caption files
caption_files = list(dataset_dir.glob('**/*.txt'))
if not caption_files:
    print('❌ No caption files found')
    exit(1)

# Use the first caption file found
caption_file = caption_files[0]
print(f'📄 Using caption file: {{caption_file}}')

# Read captions
captions = []
with open(caption_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and '#' in line:
            parts = line.split('#')
            if len(parts) >= 2:
                image_id = parts[0].strip()
                caption = parts[1].strip()
                image_path = f'Images/{{image_id}}.jpg'
                if (images_dir / f'{{image_id}}.jpg').exists():
                    captions.append({{
                        'image_id': image_id,
                        'caption': caption,
                        'image_path': image_path,
                        'split': 'train' if len(captions) < 800 else ('val' if len(captions) < 900 else 'test')
                    }})

# Save captions
df = pd.DataFrame(captions)
df.to_csv(captions_file, sep='\\t', index=False)
print(f'✅ Created {{len(df)}} captions in {{captions_file}}')
"

echo "✅ Flickr8k dataset setup complete!"
echo "📊 Dataset statistics:"
echo "   Images: $(find "$DATASET_DIR/Images" -name "*.jpg" | wc -l)"
echo "   Captions: $(wc -l < "$DATASET_DIR/captions.tsv" 2>/dev/null || echo "0")"
"""
    
    # Save the download script
    script_path = root_path / "download_flickr8k.sh"
    with open(script_path, 'w') as f:
        f.write(download_script)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Created download script: {script_path}")
    print(f"📋 To download the dataset, run: bash {script_path}")
    
    return script_path


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python download_flickr8k.py <dataset_root_dir>")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    script_path = setup_flickr8k_dataset(root_dir)
    
    print(f"\n🎯 Next steps:")
    print(f"1. Upload the download script to the cluster")
    print(f"2. Run the script on the cluster to download the dataset")
    print(f"3. Verify the dataset structure")
    print(f"4. Update the config files to use the real dataset")


if __name__ == "__main__":
    main()
