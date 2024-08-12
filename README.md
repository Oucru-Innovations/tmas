# tmas

Detect bacterial growths and determine MICs on 96-microtiter plate images

## Installation

### Installation - Python Package

```bash
$ pip install tmas
```

### Installation - GitHub

1. Clone the repository and navigate to the project directory.
2. Install the package using:

   ```bash
   pip install .
   ```

3. Run the TMAS system:

   ```bash
   python -m scripts.run_tmas
   ```

## Overview

- The package uses deep learning to detect _M. tuberculosis_ growth in 96-well microtiter plates and determines Minimum Inhibitory Concentrations (MICs).
- The model weights (`best_model_yolo.pt`) is downloaded automatically and separately to avoid including large files directly in the package.

## Usage

- TODO

## Contributing

## License

`tmas` was created by RMIT 2OO2 TEAM. It is licensed under the terms of the MIT license.

## Credits

`tmas` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
