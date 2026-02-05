from setuptools import setup, find_packages

setup(
    name="covenant",
    version="1.0.0",
    description="Covenant-AI Enterprise Package",
    author="Leonydis138",
    author_email="your-email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        # Core
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "typing-extensions>=4.9.0",

        # Web
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "httpx>=0.26.0",

        # DB & cache
        "sqlalchemy>=2.0.25",
        "redis[hiredis]>=5.0.1",

        # AI/ML
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.4.0",
        "torch>=2.1.0",

        # Config
        "python-dotenv>=1.0.0",

        # Utilities
        "click>=8.1.0",
        "rich>=13.7.0",
        "tenacity>=8.2.0",

        # Async
        "asyncio>=3.4.3",
        "aiohttp>=3.9.0",

        # Cryptography
        "cryptography>=42.0.0",
    ],
    extras_require={
        "dev": [
            "black",
            "mypy",
            "pylint",
            "pytest",
            "pytest-cov",
        ]
    },
    entry_points={
        "console_scripts": [
            "covenant-cli=covenant_cli:main",
        ]
    },
)
