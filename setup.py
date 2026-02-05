from setuptools import setup, find_packages
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

README = (BASE_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="covenant-ai-enterprise",
    version="1.0.0",
    description="Covenant.AI Enterprise — Constitutional, Ethical, Multi-Agent AI System",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Covenant AI",
    license="MIT",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "scipy",
        "scikit-learn",
        "requests",
        "httpx",
        "prometheus-client",
        "loguru",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
            "pylint",
        ],
        "llm": [
            "openai",
            "anthropic",
            "google-generativeai",
        ],
        "blockchain": [
            "web3",
            "ipfshttpclient",
        ],
    },
    entry_points={
        "console_scripts": [
            "covenant=covenant_cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
