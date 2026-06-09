from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"


setup(
    name="M_Ana_package",
    version="0.1.0",
    description="A Python package for data manipulation, analysis, modeling, statistics, and time-series workflows.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/mohamed1249/M-Ana",
    author="Muhammad Abdulsalam",
    author_email="mohameedmtmn@gmail.com",
    keywords="data science, analysis, machine learning, visualization, statistics, time series",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "plotly",
        "plotly-express",
        "matplotlib",
        "scipy",
        "pandas",
        "numpy",
        "scikit-learn",
        "pyperclip",
        "pyarrow",
        "sas7bdat",
        "pyreadstat",
        "statsmodels",
        "pyod",
        "tensorflow",
        "pandas-sas",
        "h5py",
        "pingouin",
        "pymc",
        "arviz",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
