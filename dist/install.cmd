@echo off
echo "THIS FILE CONTAINS SETTING/MANIPULATION LOGIC FOR THE CFG FILE OF THE VENV"

echo 1) init stuff
rem "set current directory in a variable"
set CURRENT_SARCASM_DIR=%~dp0

echo replace pyvenv.cfg file with current path's
(
echo home = %CURRENT_SARCASM_DIR%\Python
echo implementation = CPython
echo version_info = 3.9.5.final.0
echo virtualenv = 20.4.7
echo include-system-site-packages = false
echo base-prefix = %CURRENT_SARCASM_DIR%\Python
echo base-exec-prefix = %CURRENT_SARCASM_DIR%\Python
echo base-executable = %CURRENT_SARCASM_DIR%\Python\python.exe
)>venv\pyvenv.cfg

echo 0.5) install vcredist, get architecture (x86,x64), install the right vcredist ^if necessary
rem "check if vc redist is already installed"
rem "unfortunately wmic seems to require admin privileges"
wmic product get name|findstr /b /n /r /c:"^.*C++.*Redistributable" >tmp.txt
set /P REDIST_INSTALLED=<tmp.txt
del tmp.txt
IF "%REDIST_INSTALLED%" == "" (GOTO install_redist) ELSE (GOTO continuation_point)

:install_redist
echo %PROCESSOR_ARCHITECTURE%|findstr 64>tmp.txt
set /P IS_64_BIT=<tmp.txt
del tmp.txt
rem "check if IS_64_BIT is empty -> if empty use the x86 vcredist otherwise the x64 one"
echo %IS_64_BIT%
IF "%IS_64_BIT%" == "" (GOTO is_empty) ELSE (GOTO is_not_empty)
:is_empty
call "lib\win\VC_redist.x86.exe"
goto continuation_point

:is_not_empty
call "lib\win\VC_redist.x64.exe"
goto continuation_point

:continuation_point
echo "Install finished. Run the application via run.cmd"
pause