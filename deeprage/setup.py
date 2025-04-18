from setuptools import setup, find_packages

setup(
    name='deeprage',
    version='0.1.0',
    description='DeepRage: your all‑in‑one data profiling, visualization, and modeling toolkit',
    author='Your Name',
    packages=find_packages(),  # finds the deeprage/ folder
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'prettytable',
        'ydata_profiling',
        'fastapi',
        'uvicorn',
        'click'
    ],
    entry_points={
        'console_scripts': [
            'deeprage=deeprage.cli:main',
        ],
    },
)
