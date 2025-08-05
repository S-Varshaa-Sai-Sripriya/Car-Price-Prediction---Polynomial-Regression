from setuptools import setup, find_packages

setup(
    name='car_price_prediction',
    version='0.1.0',
    author='Varshaa Sai Sripriya Saisheshadhri',
    description='Polynomial Regression project to predict car prices',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
        'kagglehub',
        'ipykernel',
        'pyyaml',
        'joblib',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
