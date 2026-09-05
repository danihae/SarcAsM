============
Installation
============

Installation via pip
====================

Using uv (Recommended)
----------------------

`uv <https://docs.astral.sh/uv/>`_ is a modern, fast Python package and project manager. It's significantly faster than traditional tools.

#. Install uv (if not already installed)::

        # On macOS and Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # On Windows
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

#. Create and activate a new environment (Python 3.12 or 3.13)::

        uv venv --python 3.12 sarcasm
        source sarcasm/bin/activate  # On macOS/Linux
        # Or: sarcasm\Scripts\activate  # On Windows

#. Install SarcAsM and its dependencies via uv::

        uv pip install sarc-asm

   .. note::
      PyTorch will be installed automatically with CUDA support if available. For specific CUDA versions or CPU-only installation, see the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

Using conda
-----------

Alternatively, you can use conda for environment management:

#. Create and activate a new environment (Python 3.12 or 3.13)::

        conda create -n sarcasm python=3.12
        conda activate sarcasm

#. Install SarcAsM and its dependencies via pip from PyPI::

        pip install sarc-asm

   .. note::
      PyTorch will be installed automatically with CUDA support if available. For specific CUDA versions or CPU-only installation, see the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

Installation via GitHub
=======================

Using uv (Recommended)
----------------------

#. Clone repo from git::

        git clone https://github.com/danihae/SarcAsM
        cd SarcAsM

#. Create a new environment (Python 3.12 or 3.13)::

        uv venv --python 3.12
        source .venv/bin/activate  # On macOS/Linux
        # Or: .venv\Scripts\activate  # On Windows

#. Install all the required packages via::

        uv pip install .

   .. note::
      PyTorch will be installed automatically with CUDA support if available. For specific CUDA versions or CPU-only installation, see the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

Using conda
-----------

#. Clone repo from git::

        git clone https://github.com/danihae/SarcAsM
        cd SarcAsM

#. Create a new environment (Python 3.12 or 3.13)::

        conda create -n sarcasm python=3.12
        conda activate sarcasm

#. Install all the required packages from pip via::

        pip install .

   .. note::
      PyTorch will be installed automatically with CUDA support if available. For specific CUDA versions or CPU-only installation, see the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

The full installation usually takes less than 2 min, depending on internet connection.
