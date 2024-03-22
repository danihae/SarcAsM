@echo off
rem "need a python distribution in path to actually build the venv"
python -m venv --copies venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install git+https://github.com/danihae/bio-image-unet
venv\Scripts\python.exe -m pip install -r requirements.txt
pause