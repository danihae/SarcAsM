@echo off
rem Requirement for using this script is 7z added to path variable!
rem "THIS FILE SHOULD BE PLACED IN A SUBDIRECTORY called DIST in THE ROOT DIRECTORY OF THE SARCASM PROJECT"
more ..\venv\pyvenv.cfg|findstr home >tmp.txt
set /P HOME_STRING=<tmp.txt
del tmp.txt
echo "%HOME_STRING%"
rem "VENV PATH IS CREATED WITH SUBSTRING"
set VENV_PATH=%HOME_STRING:~7%
echo "%VENV_PATH%"
rem "WE GOT THE VENV_PATH, TASKS TO DO"
rem "0) CLEAN DISTRIBUTION DIRECTORY"
rem "1) COPY VENV-INTERPRETER FROM VENV_PATH TO DISTRIBUTION DIRECTORY"
rem "2) COPY VENV DIRECTORY TO DISTRIBUTION DIRECTORY"
rem "3) COPY SARCASM PACKAGE(specifically copy directories (package and guiqt)) TO DISTRIBUTION DIRECTORY"
rem "ADD 3) also copy a __main__.py file that is correct and small for starting sarcasm"
rem "4) COPY NECCESSARY LIBS/DLLs TO DISTRIBUTION DIRECTORY\Lib"
rem "5) COPY INSTALL.BAT and RUN.BAT to DISTRIBUTION DIRECTORY"
rem "6) ZIP THE DISTRIBUTION DIRECTORY"
rem "7) CLEANUP"

rem "0) CLEAN AND CREATE"
rmdir /S /Q distribution
del sarcasm.7z
mkdir distribution
mkdir distribution\sarcasm
mkdir distribution\lib

rem "0.1) CHECK IF DOCS ARE BUILD - IF YES COPY THEM"


rem "1) COPY VENV-INTERPRETER"
mkdir distribution\Python
xcopy %VENV_PATH% distribution\Python /E/H

rem "2) COPY VENV DIRECTORY TO DISTRIBUTION DIRECTORY"
mkdir distribution\venv
xcopy ..\venv distribution\venv /E/H


rem "3) COPY SARCASM"
mkdir distribution\sarcasm
xcopy ..\sarcasm distribution\sarcasm /E/H
copy __main__.py distribution\__main__.py
copy ..\requirements.txt distribution\requirements.txt

rem "4) COPY DLLs"
mkdir distribution\lib
xcopy lib distribution\lib /E/H

rem "5) COPY INSTALL/RUN BAT"
copy install.cmd distribution\install.cmd
copy run.cmd distribution\run.cmd

rem "5.5) COPY TEST NETWORKS AND ONE TESTIMAGE"
mkdir distribution\test
mkdir distribution\models
mkdir distribution\test\images
xcopy ..\models distribution\models
copy ..\test_data\2019_10kPa.tif distribution\test\images\2019_10kPa.tif

rem "5.6) add a small readme"
(
echo This distribution should work out of the box.
echo First run install.bat
echo This will set the path's of the venv and install VC Redistribution if necessary
echo The next step would be run.bat for executing the program
)>distribution\readme.txt


rem "6) PACKAGE DISTRIBUTION (ZIP)"
echo "START PACKAGING"
rem tar -a -cf "distribution.zip" distribution
7z.exe a -t7z sarcasm.7z distribution\*

rem "7) CLEANUP"
rmdir /S /Q distribution



pause