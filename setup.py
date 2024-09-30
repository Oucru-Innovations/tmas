from setuptools import setup, find_packages
from pathlib import Path

import os
from setuptools import setup, find_packages

# Use a fallback if _file_ is not defined
try:
    this_directory = os.path.abspath(os.path.dirname(_file_))
except NameError:
    this_directory = os.getcwd()  # Fallback to current working directory

# Read the content of your README file
long_description = ""
readme_path = os.path.join(this_directory, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, encoding='utf-8') as f:
        long_description = f.read()

# Debugging: print detected packages
packages = find_packages()
print("Detected packages:", packages)  # This should print the packages detected by find_packages()

setup(
    name="tmas",
    version="1.0.2",
    author="Hoang-Anh T Vo, Sang Nguyen, Ai-Quynh T Tran, Han Nguyen",
    author_email='tmas.2002team@gmail.com', 
    description='The package uses deep learning to detect M. tuberculosis growth in 96-well microtiter plates and determines **Minimum Inhibitory Concentrations (MICs)**.',
    long_description_content_type='text/markdown',  # Use Markdown for README
    long_description=long_description,
    url='https://github.com/Oucru-Innovations/tmas',
    packages=packages,  # This will include the packages detected by find_packages()
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "scikit-learn",
        "requests",
        "ultralytics",
    ],
    package_data = {
        'tmas.tmas.src.models': ['*.pt'],
        'tmas.config': ["*.json"],
        'tmas.data': ['*']
    },
    entry_points={
        "console_scripts": [
            "run_tmas=scripts.run_tmas:main",  # This references the main function in scripts/run_tmas.py
        ],
    },
    include_package_data=True,
    python_requires='>=3.9',
    # project_urls={  # Optional: for adding additional URLs about your project
    #     'Bug Reports': 'https://github.com/yourusername/your-repo-name/issues',
    #     'Source': 'https://github.com/yourusername/your-repo-name',
    # },
)