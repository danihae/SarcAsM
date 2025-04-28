@echo off
rem "need a python distribution in path to actually build the venv"

conda create -n sarcasm_venv python=3.10
conda activate sarcasm_venv
conda install pip
conda install --file requirements.txt
conda deactivate sarcasm_venv
pause