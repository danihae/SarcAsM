============
Installation
============

Installation via pip
====================

#. Create a new environment::

        conda create -n sarcasm

#. Install `PyTorch <https://pytorch.org/get-started/locally/>`_. For CUDA-capable systems, install `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_. Ensure that the CUDA toolkit version matches the PyTorch version requirements.

#. Install SarcAsM and its dependencies via pip from PyPI::

        pip install sarcasm

Installation via github
=======================

#. Clone repo from git::

        git clone https://github.com/danihae/SarcAsM
        cd sarcasm

#. Create a new environment::

        conda create -n sarcasm

#. Install `PyTorch <https://pytorch.org/get-started/locally/>`_. For CUDA-capable systems, install `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_. Ensure that the CUDA toolkit version matches the PyTorch version requirements.

#. Finally, install all the required packages from pip via::

        pip install -r requirements.txt