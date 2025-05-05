<p align="center">
  <img alt="SarcAsM logo" src="https://raw.githubusercontent.com/danihae/sarcasm/main/docs/images/logo_gray.png" width="500">
</p>

**A Python package for comprehensive analysis of sarcomere structure and function in cardiomyocytes**

[![Supported Python versions](https://img.shields.io/pypi/pyversions/sarc-asm.svg)](https://python.org)
[![Python package index](https://img.shields.io/pypi/v/sarc-asm.svg)](https://pypi.org/project/sarc-asm)
[![bioRxiv Preprint](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.04.29.650605v1-orange)](https://www.biorxiv.org/content/10.1101/2025.04.29.650605v1)

## Overview

SarcAsM (Sarcomere Analysis Multitool) is an AI-powered Python package for cardiomyocyte sarcomere analysis. It enables precise multilevel structural and functional assessment of sarcomeres in microscopy images and movies, making it suitable for drug screening, disease phenotyping, and biomechanical studies.

For details, check out our preprint: 

Daniel Haertter, Lara Hauke, Til Driehorst, Kengo Nishi, Jaden Long, Malte Tiburcy, Branimir Berecic, et al. 2025. “SarcAsM: AI-Based Multiscale Analysis of Sarcomere Organization and Contractility in Cardiomyocytes.” bioRxiv. https://doi.org/10.1101/2025.04.29.650605.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Support](#support)
- [Citation](#citation)
- [License](#license)

## Features

- Robust AI-based sarcomere feature detection in microscopy images
- Comprehensive multilevel analysis of sarcomere structure
    - Z-bands morphometrics and lateral alignment
    - Sarcomere lengths and orientations (sarcomere 'vectors')
    - Myofibril lengths and shape
    - Cell-level myofibril domains 
- High-precision tracking of individual and average sarcomere motion with ~20 nm accuracy
- Functional analysis of sarcomere contraction dynamics
- Interactive Graphical User Interface (GUI)
- Comprehensive Python API for integration into custom workflows
- Batch processing capabilities for high-throughput analysis

![Graphical abstract](https://raw.githubusercontent.com/danihae/sarcasm/main/docs/images/graphical_abstract.png)

**Summary of SarcAsM workflow and analyzed features**

## Installation

### Option 1: Installation via PyPI (Recommended)

The easiest way to install SarcAsM is via pip:

```
pip install sarc-asm
```

### Option 2: Installation from GitHub

For the latest development version:

```
pip install git+https://github.com/danihae/sarcasm.git
```

### Setting up a dedicated environment (Recommended)

We strongly recommend creating a dedicated conda environment to avoid dependency conflicts:

```
conda create -y -n sarcasm-env python=3.10
conda activate sarcasm-env
pip install sarc-asm
```

**Note:** The full installation usually takes less than 5 minutes, depending on your internet connection. For computers equipped with an NVIDIA GPU, ensure the installed PyTorch and CUDA toolkit versions are compatible. See PyTorch installation instructions for details.

## Usage

### Python Package

After installation, SarcAsM can be imported and used in your Python scripts or Jupyter notebooks:

```
# Example workflow for structural analysis
from sarcasm import Structure, Export

# Load an image or movie
sarc = Structure("path/to/your/image_or_movie.tif")

# Detect sarcomeres
sarc.detect_sarcomeres()

# Analyze sarcomere length and orientations (sarcomere 'vectors')
sarc.analyze_sarcomere_vectors()

# Analyze Z-bands
sarc.analyze_z_bands()

# Analyze myofibrils
sarc.analyze_myofibrils()

# Analyze domains
sarc.analyze_sarcomere_domains()

# Export data to xlsx file (summary statistics of each frame, full data stored as json in file base directory)
Export.export_structure_data('/path/to/xlsx/file.xlsx', sarc_obj)
```

Check out `quickstart_demo.ipynb` in the repository root or our [documentation](https://sarcasm.readthedocs.io/)
 for a practical introduction to SarcAsM's functionalities.

### Graphical User Interface

![SarcAsM GUI Screenshot](https://raw.githubusercontent.com/danihae/sarcasm/main/docs/images/app_screenshot.png)

The SarcAsM application provides an intuitive GUI built with Napari as image viewer for easy analysis:

```
# After activating your environment (e.g., conda activate sarcasm-env)
python -m sarcasm_app
```

Alternatively, you can run `./sarcasm_app/__main__.py` directly from the SarcAsM root directory if the necessary dependencies are in your PYTHONPATH.

## Documentation

Detailed documentation, including tutorials, API reference, and usage examples, can be found at:

[https://sarcasm.readthedocs.io/](https://sarcasm.readthedocs.io/)

Additional resources:
- Example notebooks located in the `docs/notebooks` directory within the repository.
- Sample data for testing purposes is available at [https://doi.org/10.5281/zenodo.8232838](https://doi.org/10.5281/zenodo.8232838).

## Support

If you encounter any issues, have questions, or want to suggest improvements:
- Please check the [online documentation](https://sarcasm.readthedocs.io/) first.
- If the issue persists, [open an issue](https://github.com/danihae/SarcAsM/issues) on our GitHub repository. Provide as much detail as possible, including steps to reproduce the problem, error messages, and your operating system/environment details.

## Citation

If you use SarcAsM in your research, please cite our preprint for now (peer-reviewed publication will follow):

Daniel Haertter, Lara Hauke, Til Driehorst, Kengo Nishi, Jaden Long, Malte Tiburcy, Branimir Berecic, et al. 2025. “SarcAsM: AI-Based Multiscale Analysis of Sarcomere Organization and Contractility in Cardiomyocytes.” bioRxiv. https://doi.org/10.1101/2025.04.29.650605.

## License

This software is patent pending (Patent Application No. DE 10 2024 112 939.5, Priority Date: 8.5.2024).

### Academic and Non-Commercial Use
This software is free for academic and non-commercial use. Users are granted a non-exclusive, non-transferable license to use the software for research, educational, and other non-commercial purposes.

### Commercial Use Restrictions
Commercial use of this software is strictly prohibited without obtaining a separate license agreement. This includes but is not limited to:
- Using the software in a commercial product or service
- Using the software to provide services to third parties
- Reselling or redistributing the software

For commercial licensing inquiries, please contact:

**MBM ScienceBridge GmbH**,
Hans-Adolf-Krebs-Weg 1,
37077 Göttingen,
Germany,
https://sciencebridge.de/en/

All rights not expressly granted are reserved. Unauthorized use may result in legal action.
