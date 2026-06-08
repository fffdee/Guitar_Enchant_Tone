@echo off
setlocal EnableDelayedExpansion

rem ============================================================
rem  export_model.bat
rem  MaskNet checkpoint -> xform_model.bin
rem
rem  Usage:
rem    export_model.bat --model outputs\mask\checkpoints\best.pt ^
rem                     --out   xform_model.bin
rem
rem    export_model.bat --model outputs\mask\checkpoints\best.pt ^
rem                     --out   E:\sd_card\xform_model.bin       ^
rem                     --proc-dir dataset\proc
rem
rem    export_model.bat --help
rem ============================================================

rem -- 0. locate project root (same directory as this .bat) ---
set "_RC=0"
set "PROJ_ROOT=%~dp0"
if "!PROJ_ROOT:~-1!"=="\" set "PROJ_ROOT=!PROJ_ROOT:~0,-1!"

rem -- 1. find Python: prefer tcn_cuda conda env --------------
set "PYTHON_EXE="
for %%C in (
    "%USERPROFILE%\Anaconda3\envs\tcn_cuda\python.exe"
    "%USERPROFILE%\anaconda3\envs\tcn_cuda\python.exe"
    "%LOCALAPPDATA%\Anaconda3\envs\tcn_cuda\python.exe"
    "%USERPROFILE%\Anaconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
) do (
    if "!PYTHON_EXE!"=="" (
        if exist %%~C set "PYTHON_EXE=%%~C"
    )
)

if "!PYTHON_EXE!"=="" (
    echo [WARN] tcn_cuda env not found, falling back to PATH python
    echo        numpy/torch may be missing - install them first
    set "PYTHON_EXE=python"
) else (
    echo [env]  Python: !PYTHON_EXE!
)

rem -- 2. verify numpy + torch are available ------------------
"!PYTHON_EXE!" -c "import numpy, torch" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] numpy or torch not found in selected Python env.
    echo.
    echo   Install them with:
    echo     pip install numpy torch
    echo   or activate the correct conda env and retry.
    echo.
    goto :end_pause
)

rem -- 3. cd to project root so _paths.py can find src/ ------
cd /d "!PROJ_ROOT!"

rem -- 4. show usage when called with no arguments ------------
if "%~1"=="" (
    echo.
    echo  Usage:
    echo    export_model.bat --model ^<checkpoint.pt^> --out ^<output.bin^>
    echo.
    echo  Examples:
    echo    export_model.bat --model outputs\mask\checkpoints\best.pt --out xform_model.bin
    echo    export_model.bat --model outputs\mask\checkpoints\best.pt ^
    echo                     --out   outputs\mask\exports\xform_model.bin ^
    echo                     --proc-dir dataset\proc
    echo.
    echo  Full help:
    echo    export_model.bat --help
    echo.
    pause
    goto :end_pause
)

rem -- 5. run export script -----------------------------------
echo.
echo [export] Running export_bin.py ...
echo.
"!PYTHON_EXE!" scripts\export_bin.py %*
set "_RC=!errorlevel!"

echo.
if "!_RC!"=="0" (
    echo [OK] Export finished successfully.
) else (
    echo [FAIL] Export failed with exit code !_RC!
    echo        Check the error messages above.
)

:end_pause
echo.
endlocal
exit /b !_RC!
