@echo off
rem "need a python distribution in path to actually build the venv"

conda create -n sarcasm_venv python=3.10
conda activate sarcasm_venv
conda install pip
conda install --file requirements.txt
conda remove torch
rem "install pytorch with pip since the site states that conda packages are no longer available"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
rem "torch+cuda after requirements, otherwise it will get overridden"
conda deactivate sarcasm_venv
pause