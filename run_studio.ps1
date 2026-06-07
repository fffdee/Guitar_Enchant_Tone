# ====================================================================
#  ESP-XFORM Studio launcher (PowerShell)
#  Launches the GUI with the fixed conda env (tcn_cuda) interpreter,
#  so it works from ANY terminal / editor / working directory.
#  Usage: run  ./run_studio.ps1  from anywhere.
#  (Falls back to system python if the conda env is not found.)
# ====================================================================
$pyexe = "C:\Users\BanGO\Anaconda3\envs\tcn_cuda\python.exe"
if (-not (Test-Path $pyexe)) { $pyexe = "python" }
$env:PYTHONIOENCODING = "utf-8"
$script = Join-Path $PSScriptRoot "tools\xform_studio.py"
& $pyexe $script @args
