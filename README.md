![SarcAsM logo](./docs/images/logo.png)

### A Python package for comprehensive analysis of sarcomere structure and function

[![Supported Python versions](https://img.shields.io/pypi/pyversions/SarcAsM.svg)](https://python.org)
[![Python package index](https://img.shields.io/pypi/v/SarcAsM.svg)](https://pypi.org/project/SarcAsM)
[![Python package index download statistics](https://img.shields.io/pypi/dm/SarcAsM.svg)](https://pypistats.org/packages/SarcAsM)
[![Development Status](https://img.shields.io/pypi/status/SarcAsM.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![DOI](https://zenodo.org/badge/xxxxxxx.svg)](https://zenodo.org/badge/latestdoi/xxxxxx)

SarcAsM is an advanced Python package designed to analyze the structure and dynamics of sarcomeres in cardiomyocytes. By leveraging machine learning techniques, SarcAsM provides automated, fast, and unbiased assessments of sarcomeric Z-bands, sarcomeres, myofibrils, and sarcomere domains. Furthermore, SarcAsM tracks and analyzes the motion of individual sarcomere ~20 nm superresolution.
## Installation

There are two ways to install and use SarcAsM:

### 1. Installation of Python package

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
1. [Download SarcAsM](https://e.pcloud.link/publink/show?code=kZUVoTZeXydDUIgTvJkRhDxLlheNp2G87w7).
2. Install requirements by running ``pip install -r requirements.txt`` in console from sarcasm directory.
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

### 2. Installation of standalone application (beta version)

For users who want to use the SarcAsM GUI, we provide an easy installation method using a batch file. This method does not require any prerequisites.

1. Download the full distribution zip file.
2. Extract the zip file on your system.
3. Run the `install.bat` script.

## Usage

A detailed online documentation with tutorials and examples can be found [here](https://filedn.eu/lKfS794F9UgX7PDuBQcfChB/SarcAsM_docs/).
A data set for testing is available in the folder 'test_data'.

### Standalone application with GUI (beta version)
After setting up, the SarcAsM app is started by running the provided batch file `run.bat`.
Alternatively, the app can be started by executing `sarcasm/app/__main__.py`. 

### Python package
After installation, SarcAsM is imported by ``from sarcasm import *``. If SarcAsM is in a local directory, add the path to the system path (see above). 
Examples and tutorials can be found under 'docs/notebooks' and in the [online documentation](https://filedn.eu/lKfS794F9UgX7PDuBQcfChB/SarcAsM_docs/). 

## Support

If you encounter any issues or have any questions about using SarcAsM, please [open an issue](link-to-issue-tracker) on our GitHub repository.

## License

This software is patent pending (Patent Application No. DE 10 2024 112 939.5, Priority Date: 8.5.2024).

### Academic and Non-Commercial Use

This software is free for academic and non-commercial use. Users are granted a non-exclusive, non-transferable license to use and modify the software for research, educational, and other non-commercial purposes.

### Commercial Use Restrictions

Commercial use of this software is strictly prohibited without obtaining a separate license agreement. This includes but is not limited to:

- Using the software in a commercial product or service
- Using the software to provide services to third parties
- Reselling or redistributing the software

For commercial licensing inquiries, please contact:

**MBM ScienceBridge GmbH**,
Hans-Adolf-Krebs-Weg 1,
37077 Göttingen,
https://sciencebridge.de/en/

All rights not expressly granted are reserved. Unauthorized use may result in legal action.


## Citation

If you use SarcAsM in your research, please cite our paper:

[insert citation here upon publication]
