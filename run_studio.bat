@echo off
REM ====================================================================
REM  ESP-XFORM Studio launcher (Windows)
REM  Launches the GUI with the fixed conda env (tcn_cuda) interpreter,
REM  so it works from ANY terminal / editor / working directory.
REM  Usage: double-click this file, or run  run_studio.bat  from anywhere.
REM  (Falls back to system python if the conda env is not found.)
REM ====================================================================
setlocal
set "PYEXE=C:\Users\BanGO\Anaconda3\envs\tcn_cuda\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
set "PYTHONIOENCODING=utf-8"
REM  %~dp0 = this script's folder (project root, with trailing backslash)
"%PYEXE%" "%~dp0tools\xform_studio.py" %*
if errorlevel 1 (
  echo.
  echo [run_studio] launch failed, exit code %errorlevel%. See error above.
  pause
)
endlocal
