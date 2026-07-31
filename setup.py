from setuptools import setup, find_packages

setup(
    name="xMHashSeg",
    version="0.1.0",
    author="Jialong Zhang",
    description="xMHashSeg",
    packages=find_packages(),
    install_requires=[
        "tabulate",
        "scikit-learn",
        "open3d",
        "yacs",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)