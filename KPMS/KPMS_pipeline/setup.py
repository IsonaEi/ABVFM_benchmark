from setuptools import setup, find_packages

setup(
    name="kpms_custom",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "keypoint-moseq",
        "jax",
        "jaxlib",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "plotly",
        "pyyaml",
        "opencv-python",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "kpms=kpms_custom.core.cli:main",
        ],
    },
    python_requires=">=3.9",
)
