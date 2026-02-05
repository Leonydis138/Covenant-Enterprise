from setuptools import setup, find_packages

# Optional: Only if you need Path usage
from pathlib import Path

here = Path(__file__).parent.resolve()

setup(
    name="Covenant-Enterprise",
    version="1.0.0",
    description="Covenant Enterprise AI Framework",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/Leonydis138/Covenant-Enterprise",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
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
        "jaeger-client",
        "pytest",
        "pytest-cov",
        "black",
        "mypy",
        "pylint",
        "isort"
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "mypy", "pylint", "isort"]
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "covenant=covenant_cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
