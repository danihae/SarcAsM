@echo off
rem "need a python distribution in path to actually build the venv"
python -m venv --copies venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install git+https://github.com/danihae/bio-image-unet
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip uninstall -y torch
rem "todo: it would be better (maybe) to uninstall torch before the next command"
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
rem "torch+cuda after requirements, otherwise it will get overridden"
pause