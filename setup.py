from setuptools import setup, find_packages

# Debugging: print detected packages
packages = find_packages()
print("Detected packages:", packages)  # This should print the packages detected by find_packages()

setup(
    name="tmas",
    version="0.1.0",
    packages=packages,  # This will include the packages detected by find_packages()
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
            "run_tmas=scripts.run_tmas:main",  # This references the main function in scripts/run_tmas.py
        ],
    },
    include_package_data=True,
)
