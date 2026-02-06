"""
Setup configuration for Covenant Enterprise.
Enhanced version with comprehensive metadata and dependencies.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Get the project root directory
ROOT_DIR = Path(__file__).parent

# Read the long description from README
readme_path = ROOT_DIR / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Constitutional Alignment Framework for Autonomous Intelligence"

# Read requirements
def read_requirements(filename: str) -> list:
    """Read requirements from a file."""
    req_path = ROOT_DIR / filename
    if not req_path.exists():
        return []
    
    with open(req_path, encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                # Remove inline comments
                line = line.split("#")[0].strip()
                if line:
                    requirements.append(line)
        return requirements

# Read version from __init__.py
def get_version():
    """Extract version from package __init__.py."""
    init_py = ROOT_DIR / "src" / "covenant" / "__init__.py"
    if init_py.exists():
        content = init_py.read_text(encoding="utf-8")
        version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
    return "1.0.0"

# Core requirements
install_requires = read_requirements("requirements.txt")

# Development requirements
extras_require = {
    "dev": [
        "pytest>=8.3.0",
        "pytest-asyncio>=0.24.0",
        "pytest-cov>=6.0.0",
        "pytest-mock>=3.14.0",
        "pytest-timeout>=2.3.0",
        "hypothesis>=6.122.0",
        "faker>=33.1.0",
        "black>=24.10.0",
        "isort>=5.13.0",
        "pylint>=3.3.0",
        "mypy>=1.13.0",
        "flake8>=7.1.0",
        "bandit[toml]>=1.8.0",
        "pre-commit>=4.0.0",
    ],
    "docs": [
        "mkdocs>=1.6.0",
        "mkdocs-material>=9.5.0",
        "mkdocstrings[python]>=0.26.0",
    ],
    "monitoring": [
        "prometheus-client>=0.21.0",
        "opentelemetry-api>=1.28.0",
        "opentelemetry-sdk>=1.28.0",
        "jaeger-client>=4.8.0",
    ],
    "blockchain": [
        "web3>=7.6.0",
        "ipfshttpclient>=0.8.0a2",
    ],
    "ml": [
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "transformers>=4.46.0",
        "sentence-transformers>=3.3.0",
        "faiss-cpu>=1.9.0",
    ],
}

# All extras
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    # Basic Information
    name="covenant-enterprise",
    version=get_version(),
    author="Covenant.AI Team",
    author_email="team@covenant-ai.org",
    maintainer="Covenant.AI Team",
    maintainer_email="team@covenant-ai.org",
    
    # Description
    description="Constitutional Alignment Framework for Autonomous Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/Leonydis138/Covenant-Enterprise",
    project_urls={
        "Bug Reports": "https://github.com/Leonydis138/Covenant-Enterprise/issues",
        "Source": "https://github.com/Leonydis138/Covenant-Enterprise",
        "Documentation": "https://covenant-ai.readthedocs.io/",
        "Changelog": "https://github.com/Leonydis138/Covenant-Enterprise/blob/main/CHANGELOG.md",
        "Discussions": "https://github.com/Leonydis138/Covenant-Enterprise/discussions",
    },
    
    # Package Configuration
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests*", "examples*", "docs*"]),
    include_package_data=True,
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Python Version Requirement
    python_requires=">=3.11,<4.0",
    
    # Entry Points
    entry_points={
        "console_scripts": [
            "covenant=covenant.cli:main",
            "covenant-server=covenant.api.main:run_server",
        ],
    },
    
    # Classifiers
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Topic :: Security",
        
        # Framework
        "Framework :: FastAPI",
        "Framework :: AsyncIO",
        
        # Typing
        "Typing :: Typed",
    ],
    
    # Keywords
    keywords=[
        "ai",
        "artificial-intelligence",
        "autonomous-systems",
        "constitutional-ai",
        "ethics",
        "alignment",
        "safety",
        "governance",
        "compliance",
        "enterprise",
        "blockchain",
        "audit",
        "verification",
        "constraints",
        "policy-enforcement",
    ],
    
    # Package Data
    package_data={
        "covenant": [
            "py.typed",
            "*.yaml",
            "*.json",
            "*.toml",
        ],
    },
    
    # Zip Safe
    zip_safe=False,
    
    # Platforms
    platforms=["any"],
    
    # License
    license="MIT",
    
    # Additional Metadata
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
)
