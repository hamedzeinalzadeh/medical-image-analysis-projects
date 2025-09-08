"""
Setup script for the evaluation package.

This script configures the package for installation and distribution.
"""

from setuptools import setup, find_packages
import os

# Read the README file for the long description
def read_readme():
    """Read README.md file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read version from __init__.py
def get_version():
    """Extract version from __init__.py."""
    init_path = os.path.join(os.path.dirname(__file__), 'evaluation', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name='evaluation-keypoints',
    version=get_version(),
    author='Evaluation Package Team',
    author_email='contact@example.com',
    description='A Python package for evaluating keypoint detection models',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/evaluation-keypoints',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
        ],
        'test': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=['keypoint detection', 'evaluation', 'computer vision', 'metrics', 'PCK'],
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/evaluation-keypoints/issues',
        'Source': 'https://github.com/yourusername/evaluation-keypoints',
        'Documentation': 'https://github.com/yourusername/evaluation-keypoints#readme',
    },
)