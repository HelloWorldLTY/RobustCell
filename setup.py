from setuptools import setup, find_packages

setup(
    name='RobustCell',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'aiohttp>=3.9.3',
        'aiosignal>=1.3.1',
        'async-timeout>=4.0.3',
        'attrs>=23.2.0',
        'Brotli>=1.1.0',  # PyPI package name for brotli
        'cffi>=1.16.0',
        'charset-normalizer>=3.3.2',
        'colorama>=0.4.6',
        'idna>=3.6',
        'Jinja2>=3.1.3',
        'joblib>=1.3.2',
        'MarkupSafe>=2.1.5',  # For markupsafe
        'multidict>=6.0.5',
        'packaging>=24.0',
        'psutil>=5.9.8',
        'pycparser>=2.21',
        'pyparsing>=3.1.2',
        'requests>=2.31.0',
        'scikit-learn>=1.3.2',
        'setuptools>=69.2.0',
        'tqdm>=4.66.2',
        'typing-extensions>=4.10.0',
        'yarl>=1.9.4'
    ],
    python_requires='>=3.6',
    author='Tianyu Liu, Yijia Xiao, Xiao Luo',
    author_email='your.email@example.com',
    description='A Python package for robust cell modeling.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/HelloWorldLTY/RobustCell',
)