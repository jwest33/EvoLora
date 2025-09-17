"""
Setup script for LoRALab
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="loralab",
    version="0.1.0",
    author="LoRALab Team",
    description="R-Zero inspired LoRA adapter generator for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "tqdm>=4.65.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
        ]
    },
    entry_points={
        "console_scripts": [
            "loralab=loralab.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "loralab": ["configs/*.yaml", "configs/*.json"],
    },
)
