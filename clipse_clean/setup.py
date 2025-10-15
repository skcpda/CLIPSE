from setuptools import setup, find_packages

setup(
    name="clipse",
    version="1.0.0",
    description="CLIP with SANW (Similarity-Aware Negative Weighting)",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.0",
        "torchvision>=0.20.0", 
        "open-clip-torch>=2.20.0",
        "timm>=0.9.0",
        "ftfy>=6.0.0",
        "regex>=2023.0.0",
        "tqdm>=4.64.0",
        "huggingface-hub>=0.20.0",
        "pandas>=1.5.0",
        "pillow>=9.0.0",
        "scikit-learn>=1.3.0",
        "safetensors>=0.4.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0"
    ],
    python_requires=">=3.10",
)
