from setuptools import setup, find_packages

setup(
    name="psoriasis_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'opencv-python',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy'
    ],
) 