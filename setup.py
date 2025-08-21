#!/usr/bin/env python3
"""
Setup script for ASDP (AI Survey Data Processor) Application
Ministry of Statistics and Programme Implementation (MoSPI)
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ai-survey-data-processing",
    version="1.0.0",
    author="MoSPI Development Team",
    author_email="contact@mospi.gov.in",
    description="AI-Enhanced Application for Automated Data Preparation, Estimation and Report Writing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://datainnovation.mospi.gov.in/",
    project_urls={
        "Bug Reports": "https://github.com/mospi/ai-survey-data-processing/issues",
        "Source": "https://github.com/mospi/ai-survey-data-processing",
        "Documentation": "https://github.com/mospi/ai-survey-data-processing/blob/main/README.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Government",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Accounting",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-survey-processor=run:main",
            "test-survey-processor=test_app:run_tests",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.html", "*.css", "*.js", "*.csv", "*.txt", "*.md"],
    },
    keywords=[
        "survey",
        "data-processing",
        "statistics",
        "government",
        "mospi",
        "ai",
        "machine-learning",
        "data-cleaning",
        "report-generation",
        "statistical-analysis",
    ],
    platforms=["any"],
    license="MIT",
    zip_safe=False,
)
