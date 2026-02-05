"""
Setup configuration for COVENANT.AI
"""

from setuptools import setup, find_packages
from pathlib import Path
from itertools import chain

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core dependencies
INSTALL_REQUIRES = [
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "typing-extensions>=4.9.0",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "httpx>=0.26.0",
    "sqlalchemy>=2.0.25",
    "redis[hiredis]>=5.0.1",
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.4.0",
    "torch>=2.1.0",
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "rich>=13.7.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.23.0",
        "pytest-cov>=4.1.0",
        "black>=24.1.0",
        "ruff>=0.1.0",
        "mypy>=1.8.0",
    ],
    "quantum": [
        "qiskit>=0.45.0",
        "qiskit-aer>=0.13.0",
    ],
    "llm": [
        "openai>=1.10.0",
        "anthropic>=0.18.0",
        "transformers>=4.36.0",
    ],
    "blockchain": [
        "web3>=6.15.0",
        "cryptography>=42.0.0",
    ],
}

# Combine all extras into "all"
EXTRAS_REQUIRE["all"] = list(chain.from_iterable(EXTRAS_REQUIRE.values()))

setup(
    name="covenant-ai",
    version="2.0.0",
    description="Constitutional Alignment Framework for Autonomous Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Covenant.AI Team",
    author_email="team@covenant-ai.org",
    url="https://github.com/covenant-ai/covenant-ai",
    project_urls={
        "Documentation": "https://docs.covenant-ai.org",
        "Bug Tracker": "https://github.com/covenant-ai/covenant-ai/issues",
        "Source Code": "https://github.com/covenant-ai/covenant-ai",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "covenant=covenant.cli:main",
            "covenant-server=covenant.api.main:run_server",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    keywords=[
        "ai-safety",
        "alignment",
        "autonomous-agents",
        "constitutional-ai",
        "ethics",
        "multi-agent",
    ],
    include_package_data=True,
    zip_safe=False,
)
