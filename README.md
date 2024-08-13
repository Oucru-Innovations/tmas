# tmas

Detect bacterial growths and determine MICs on 96-microtiter plate images

## Installation

### Installation - GitHub

1. Clone the repository and navigate to the project directory.
   ```bash
   git clone https://github.com/Oucru-Innovations/tmas.git
   cd tmas
   ```
2. Install the package using:

   ```bash
   pip install -r requirements.txt
   ```
   This may take a while if those packages are not available in the current environment. The internet connection and the hardware capacity also affect the installation speed. 

3. Run TMAS:

   ```bash
   python -m scripts.run_tmas
   ```

### Installation - Python Package (Upcoming when all features finalised)

```bash
$ pip install tmas
```

## Overview

- The package uses deep learning to detect _M. tuberculosis_ growth in 96-well microtiter plates and determines Minimum Inhibitory Concentrations (MICs).
- The model weights (`best_model_yolo.pt`) is downloaded automatically and separately to avoid including large files directly in the package. It is downloaded only once. 

## Usage

- TODO

## Contributing

## License

`tmas` was created by RMIT 2OO2 TEAM. It is licensed under the terms of the MIT license.

## Credits

`tmas` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
