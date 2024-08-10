from setuptools import setup, find_packages

setup(
    name="tmas",
    version="0.1.0",
    package_dir={"": "src"},  # Pointing to the src directory
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "scikit-learn",
        "requests",
        "ultralytics",
    ],
    entry_points={
        "console_scripts": [
            "run_tmas=src.tmas.scripts.run_tmas:main",
        ],
    },
    include_package_data=True,
)