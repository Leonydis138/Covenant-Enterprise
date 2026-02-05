from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="covenant-ai",
    version="0.1.0",
    author="Covenant.AI Team",
    description="Constitutional Alignment Framework for Autonomous Intelligence",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Leonydis138/Covenant-Enterprise",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
)
