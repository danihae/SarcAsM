@echo off
rem "the minimal dist should only contain sarcasm, test networks, test image"
rem "create_clean_venv.bat"
rem "a run script"

rem "0) CLEAN AND CREATE"
rmdir /S /Q distribution_min
del sarcasm_min.7z
mkdir distribution_min
mkdir distribution_min\sarcasm
mkdir distribution_min\sarcasm_app
mkdir distribution_min\contraction_net

rem "1) COPY SARCASM"
xcopy ..\sarcasm distribution_min\sarcasm /E/H
xcopy ..\sarcasm_app distribution_min\sarcasm_app /E/H
xcopy ..\contraction_net distribution_min\contraction_net /E/H
copy __main__.py distribution_min\__main__.py
copy ..\requirements.txt distribution_min\requirements.txt


rem "2) COPY RUN BAT, CREATE_CLEAN_VENV.BAT"
copy run.cmd distribution_min\run.cmd
copy create_clean_venv.cmd distribution_min\create_clean_venv.cmd
copy create_clean_venv_cuda.cmd distribution_min\create_clean_venv_cuda.cmd

rem "3) COPY TEST NETWORKS"
mkdir distribution_min\test
mkdir distribution_min\models
mkdir distribution_min\test\images
xcopy ..\models distribution_min\models

rem "4) ADD README"
(
echo This minimal distribution needs a working python greater equal 3.10 version installed.
echo It also requires installed GIT.
echo Otherwise it will not work, the create_clean_venv.bat requires python in the path-variable.
echo execute create_clean_venv.bat or the one with cuda depending on your system.
echo this will take a while, if its done you can use the run script.
)>distribution_min\readme.txt



rem "5) PACKAGE DISTRIBUTION (ZIP)"
echo "START PACKAGING"
7z.exe a -t7z sarcasm_min.7z distribution_min\*
rem "tar -a -cf "distribution_min.zip" distribution_min"

rem "6) CLEANUP"
rmdir /S /Q distribution_min
pause
