============
Installation
============

Installation via pip
====================

#. Create and activate a new environment (tested with Python >=3.10)::

        conda create -n sarcasm python==3.10
        conda activate sarcasm

#. Install `PyTorch <https://pytorch.org/get-started/locally/>`_. For CPU and Apple Silicon (M1, M2, ...), install the plain version. For CUDA-capable systems with NVIDIA GPU, install `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_. Ensure that the CUDA toolkit version matches the PyTorch version requirements.

#. Install SarcAsM and its dependencies via pip from PyPI::

        pip install sarc-asm

Installation via github
=======================

#. Clone repo from git::

        git clone https://github.com/danihae/SarcAsM
        cd sarcasm

#. Create a new environment (tested with Python >=3.10)::

        conda create -n sarcasm python==3.10
        conda activate sarcasm

#. Install `PyTorch <https://pytorch.org/get-started/locally/>`_. For CUDA-capable systems, install `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_. Ensure that the CUDA toolkit version matches the PyTorch version requirements.

#. Finally, install all the required packages from pip via::

        pip install .

The full installation usually takes less than 2 min, depending on internet connection.
