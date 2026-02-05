from setuptools import setup, find_packages

setup(
    name="Covenant-Enterprise",
    version="0.1.0",
    description="Covenant AI Enhanced Enterprise Suite",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/Leonydis138/Covenant-Enterprise",
    packages=find_packages(where="Covenant-Enterprise"),
    package_dir={"": "Covenant-Enterprise"},
    include_package_data=True,
    install_requires=[
        # Core Dependencies
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "typing-extensions>=4.9.0",
        "python-dotenv>=1.0.0",
        "click>=8.3.1",
        "rich>=13.7.0",
        "tenacity>=8.2.0",

        # Web Framework
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "httpx>=0.28.1",
        "aiohttp>=3.13.3",
        "asyncio>=3.4.3",

        # Database & Cache
        "sqlalchemy>=2.0.25",
        "redis[hiredis]>=5.0.1",

        # AI / ML Core
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.8.0",
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "torchvision>=2.1.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.13.2",
        "safetensors>=0.3.0",

        # Blockchain & Networking
        "web3>=6.9.0",
        "ipfshttpclient>=0.7.0",
        "eth-abi>=5.2.0",
        "eth-account>=0.13.7",
        "eth-hash>=0.7.1",
        "eth-keys>=0.7.0",
        "eth-typing>=5.2.1",
        "eth-utils>=5.3.1",
        "hexbytes>=1.3.1",

        # Google & OpenAI APIs
        "openai>=1.8.0",
        "anthropic>=0.77.1",
        "google-generativeai>=0.8.6",
        "google-api-python-client>=2.189.0",
        "google-api-core>=2.29.0",
        "google-auth>=2.49.0",
        "google-auth-httplib2>=0.3.0",

        # Observability & Monitoring
        "prometheus-client>=0.17.0",
        "jaeger-client>=4.8.0",
        "opentracing>=2.4.0",

        # Dev & Testing
        "pytest>=8.2.0",
        "pytest-cov>=4.1.0",
        "black>=26.1.0",
        "mypy>=1.5.1",
        "pylint>=2.20.0",
        "isort>=7.0.0",

        # Cryptography
        "cryptography>=46.0.4",
        "pycryptodome>=3.20.0",
        "pycparser>=2.21",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "mypy",
            "pylint",
            "isort",
        ]
    },
    python_requires=">=3.11",
)
