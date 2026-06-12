@echo off
REM ====================================================================
REM  ESP-XFORM Studio launcher (Windows)
REM  Launches the GUI with the fixed conda env (tcn_cuda) interpreter,
REM  so it works from ANY terminal / editor / working directory.
REM  
REM  Usage:
REM    run_studio.bat              - 启动上位机 GUI
REM    run_studio.bat --train      - 直接启动训练（使用默认配置）
REM    run_studio.bat --train --help - 显示训练参数帮助
REM  
REM  (Falls back to system python if the conda env is not found.)
REM ====================================================================
setlocal
set "PYEXE=C:\Users\BanGO\Anaconda3\envs\tcn_cuda\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
set "PYTHONIOENCODING=utf-8"
set "PROJ_ROOT=%~dp0"

if "%~1"=="" (
    REM 无参数：启动上位机 GUI
    "%PYEXE%" "%PROJ_ROOT%tools\xform_studio.py" %*
) else if /i "%~1"=="--train" (
    REM --train 参数：启动训练
    shift
    "%PYEXE%" "%PROJ_ROOT%scripts\train_mask.py" --gpu-accelerate %*
) else (
    REM 其他参数：直接传给 studio
    "%PYEXE%" "%PROJ_ROOT%tools\xform_studio.py" %*
)

if errorlevel 1 (
  echo.
  echo [run_studio] launch failed, exit code %errorlevel%. See error above.
  pause
)
endlocal