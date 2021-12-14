from setuptools import setup, find_packages

__version__ = "1.0"
setup(
    name="kaggle-jigsaw-toxic-severity-rating",
    version=__version__,
    python_requires="~=3.7",
    install_requires=[
        "lightgbm==3.3.1",
        "optuna==2.9.1",
        "pandas==1.3.2",
        "pyarrow==5.0.0",
        "scikit-learn==0.23.2",
        "pytorch-lightning==1.4.5",
        "sentence-transformers==2.1.0",
        "transformers==4.9.2",
        "tqdm==4.62.1",
    ],
    extras_require={
        "tests": [
            "black==21.9b0",
            "mypy==0.910",
            "pytest==6.2.3",
            "pytest-cov==2.11.1",
        ],
        "notebook": ["jupyterlab==1.2.21", "ipywidgets==7.6.3", "seaborn==0.11.2"],
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="Jigsaw Toxic Severity Rating - Kaggle competition",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/kaggle-jigsaw-toxic-severity-rating",
)
