![SarcAsM logo](./docs/images/logo.png)

### Integrated multiscale analysis of sarcomere structure and function

[![License](https://img.shields.io/pypi/l/SarcAsM.svg)](https://github.com/danihae/SarcAsM/raw/main/LICENSE)
[![Build Status](https://api.cirrus-ci.com/github/yourusername/SarcAsM.svg)](https://cirrus-ci.com/yourusername/SarcAsM)
[![Code coverage](https://codecov.io/gh/yourusername/SarcAsM/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/SarcAsM)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/SarcAsM.svg)](https://python.org)
[![Python package index](https://img.shields.io/pypi/v/SarcAsM.svg)](https://pypi.org/project/SarcAsM)
[![Python package index download statistics](https://img.shields.io/pypi/dm/SarcAsM.svg)](https://pypistats.org/packages/SarcAsM)
[![Development Status](https://img.shields.io/pypi/status/SarcAsM.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![DOI](https://zenodo.org/badge/xxxxxxx.svg)](https://zenodo.org/badge/latestdoi/xxxxxx)

**SarcAsM** is a novel computational tool designed for integrated multiscale analyses of sarcomere structure and function. Sarcomeres are the fundamental units of muscle contraction, and understanding their structure and dynamics is key to understanding cardiomyocyte function. SarcAsM employs machine learning techniques for automated, unbiased, and integrated multiscale assessment of sarcomeric z-bands, sarcomeres, and myofibrils. This allows for a comprehensive understanding of how sarcomere scale dynamics translate into cardiomyocyte function.

## Installation

There are two ways to install use SarcAsM:

### 1. Installation of standalone application (beta version)

For users who want to use the SarcAsM GUI, we provide an easy installation method using a batch file. This method does not require any prerequisites.

1. Download the full distribution zip file.
2. Extract the zip file on your system.
3. Run the `install.bat` script.

### 2. Installation of Python package

For more experienced users or developers, SarcAsM can be installed via pip or directly from GitHub. This method requires a Python environment and Git.

#### Create environment
```sh
conda create -y -n sarcasm-env python=3.9
conda activate sarcasm-env
```

#### Installation via PyPI (available upon publication):
```sh
pip install sarcasm
```

#### Installation from GitHub (available upon publication):
```sh
pip install git+https://github.com/danihae/sarcasm.git
```

#### For reviewers (local directory):
Make sure Python (>=3.9) is installed and/or create environment (see above). 
1. Download SarcAsM package from Zenodo repository.
2. Install bio-image-unet ``pip install git+https://github.com/danihae/bio-image-unet``.

3. Install requirements by running ``pip install -r requirements.txt`` in console from sarcasm directory.
4. Add SarcAsM package path to system path and test import. 
```python
import sys

# Determine the path to your local SarcAsM directory
package_path = "/path/to/sarcasm/"

# Add the path to the system path
sys.path.append(package_path)

# Test: import the SarcAsM module or package from the local directory
from sarcasm import *
```
## Usage

A detailed online documentation with tutorials and examples can be found [here](https://filedn.eu/lKfS794F9UgX7PDuBQcfChB/SarcAsM/).
A data set for testing is available in the folder 'test_data'.

### Standalone application with GUI (beta version)
After setting up, the SarcAsM app is started by running the provided batch file `run.bat`. For more information on how to use SarcAsM, please refer to the [User Guide](link-to-user-guide).
Alternatively, the app can be started by executing `__main__.py`. 

### Python package
After installation, SarcAsM is imported by ``from sarcasm import *``. If SarcAsM is in a local directory, add the path to the system path (see above). 
Examples and tutorials can be found under 'docs/notebooks' and the [online documentation](https://filedn.eu/lKfS794F9UgX7PDuBQcfChB/SarcAsM/). 

## Support

If you encounter any issues or have any questions about using SarcAsM, please [open an issue](link-to-issue-tracker) on our GitHub repository.

## License

SarcAsM is open-source software released under [insert license here].

## Citation

If you use SarcAsM in your research, please cite our paper:

[insert citation here]
