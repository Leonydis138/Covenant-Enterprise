from setuptools import setup, find_packages
from pathlib import Path

# Get the long description from README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Constitutional Alignment Framework for Autonomous Intelligence"

# Read requirements
req_path = Path(__file__).parent / "requirements.txt"
if req_path.exists():
    with open(req_path, encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="covenant-ai",
    version="0.1.0",
    author="Covenant.AI Team",
    author_email="team@covenant-ai.org",
    description="Constitutional Alignment Framework for Autonomous Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Leonydis138/Covenant-Enterprise",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests*", "examples*"]),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    keywords="ai, autonomous, constitutional, ethics, alignment, safety",
    project_urls={
        "Bug Reports": "https://github.com/Leonydis138/Covenant-Enterprise/issues",
        "Source": "https://github.com/Leonydis138/Covenant-Enterprise",
    },
)
