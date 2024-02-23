from setuptools import setup, find_packages

setup(
    name='quick_frame',
    version='0.1.0',
    author='Tomas Zisko',
    author_email='tom.zisko@proton.me',
    description='Extends pd.DataFrame with additional functionality for data analysis and preprocessing',
    long_description=open('README.md').read(),  # Provide the path to your README file
    long_description_content_type='text/markdown',
    url='https://github.com/tomms17/QuickFrame.git',  # Provide the URL to your project repository
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)