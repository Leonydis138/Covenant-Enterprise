#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="Covenant-Enterprise",
    version="1.0.0",
    description="Covenant Enterprise AI Framework",
    packages=find_packages(where="src") if (Path("src").exists()) else find_packages(),
    package_dir={"": "src"} if (Path("src").exists()) else None,
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pandas",
        "scipy",
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "web3",
        "ipfshttpclient",
        "openai",
        "anthropic",
        "google-generativeai",
        "prometheus-client",
        "jaeger-client"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "mypy",
            "pylint",
            "isort"
        ]
    },
    entry_points={
        "console_scripts": [
            "covenant=covenant_cli:main",
        ],
    },
)
