from setuptools import setup, find_packages

setup(
    name="tmas",
    version="0.1.0",
    packages=find_packages(),
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
            "run_tmas=tmas.scripts.run_tmas:main",
        ],
    },
    include_package_data=True,
)
