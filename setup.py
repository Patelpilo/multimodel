from setuptools import setup, find_packages

setup(
    name="multimodal-rag",
    version="1.0.0",
    description="Multi-Modal RAG Document Intelligence System",
    author="AI Engineer",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith('#')
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "multimodal-rag=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
)