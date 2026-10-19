@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUTF8=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not errorlevel 0 exit /b 1
set "ENV_PY=%CD%\.venv\Scripts\python.exe"
set "BASE_PY="
set "BASE_ARGS="

rem Keep this file ASCII with CRLF. Do not put UTF-8 messages in cmd.exe.
rem Python Install Manager can return NEGATIVE error codes. Only zero succeeds.
echo ==================================================
echo Image editor - Install project environment
echo Models are downloaded separately. See README.md for Chinese instructions.
echo Supports 64-bit Python 3.10 through 3.13.
echo ==================================================
if not "%~1"=="" if /i not "%~1"=="cpu" goto :usage
if not exist "requirements.txt" goto :missing_project
if not exist "bootstrap.py" goto :missing_project

echo [1/5] Check or create the project virtual environment...
if not exist "%ENV_PY%" goto :find_python
call :check_environment
if "%ERRORLEVEL%"=="0" goto :environment_ready
echo Existing .venv is unusable. Looking for an installed Python before repair.

:find_python
where py >nul 2>nul
if not "%ERRORLEVEL%"=="0" goto :try_path_python
for %%V in (3.12 3.11 3.10 3.13) do (
    call :try_launcher %%V
    if defined BASE_PY goto :create_environment
)
:try_path_python
where python >nul 2>nul
if not "%ERRORLEVEL%"=="0" goto :missing_python
python -c "import struct,sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
if not "%ERRORLEVEL%"=="0" goto :missing_python
set "BASE_PY=python"
set "BASE_ARGS="

:create_environment
if not exist ".venv" goto :new_environment
rem Resolve and check both absolute paths before renaming an invalid environment.
"%BASE_PY%" %BASE_ARGS% -c "from pathlib import Path; from datetime import datetime; root=Path.cwd().resolve(); src=root/'.venv'; dst=root/('.venv.backup_'+datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S_%%f')); assert src.resolve().parent == root and dst.resolve().parent == root, 'Refusing to move an environment outside this project'; src.rename(dst); print('Previous environment preserved at:',dst)"
if not "%ERRORLEVEL%"=="0" goto :backup_failed
:new_environment
"%BASE_PY%" %BASE_ARGS% -m venv ".venv"
if not "%ERRORLEVEL%"=="0" goto :fail
call :check_environment
if not "%ERRORLEVEL%"=="0" goto :bad_environment

:environment_ready
echo [2/5] Update pip, setuptools and wheel...
"%ENV_PY%" -m pip install --upgrade pip setuptools wheel --retries 3 --timeout 60
if not "%ERRORLEVEL%"=="0" goto :fail

set "USE_CUDA=0"
set "TASK_TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if /i "%~1"=="cpu" goto :torch_selected
where nvidia-smi >nul 2>nul
if not "%ERRORLEVEL%"=="0" goto :torch_selected
nvidia-smi -L >nul 2>nul
if not "%ERRORLEVEL%"=="0" goto :torch_selected
set "USE_CUDA=1"
set "TASK_TORCH_INDEX=https://download.pytorch.org/whl/cu130"
:torch_selected
if defined TORCH_INDEX_URL set "TASK_TORCH_INDEX=%TORCH_INDEX_URL%"
echo [3/5] Check PyTorch. Package index: %TASK_TORCH_INDEX%
if "%USE_CUDA%"=="0" echo CPU mode selected. Image generation may be slow.
"%ENV_PY%" -c "import bootstrap; import torch,torchvision; assert %USE_CUDA% or torch.version.cuda is None, 'CPU build required'; device='cuda' if %USE_CUDA% else 'cpu'; x=torch.ones((16,16),device=device); assert float((x@x).sum().cpu()) == 4096.0; print('Existing PyTorch works',torch.__version__,device)" >nul 2>nul
if "%ERRORLEVEL%"=="0" goto :dependencies
"%ENV_PY%" -m pip install --upgrade --force-reinstall torch torchvision --index-url "%TASK_TORCH_INDEX%" --retries 3 --timeout 60
if not "%ERRORLEVEL%"=="0" goto :fail

:dependencies
echo [4/5] Install project requirements...
"%ENV_PY%" -m pip install -r "requirements.txt" --retries 3 --timeout 60
if not "%ERRORLEVEL%"=="0" goto :fail
"%ENV_PY%" -m pip check
if not "%ERRORLEVEL%"=="0" goto :fail

echo [5/5] Verify imports and actual computation without loading models...
"%ENV_PY%" -c "import bootstrap; import torch,torchvision,gradio,diffusers,transformers,accelerate,safetensors,huggingface_hub,cv2,lpips,numpy,PIL,pycocotools; import app; device='cuda' if %USE_CUDA% else 'cpu'; x=torch.ones((16,16),device=device); assert float((x@x).sum().cpu()) == 4096.0; print('PyTorch',torch.__version__,'mode',device); print('Web app and dependency imports passed')"
if not "%ERRORLEVEL%"=="0" goto :fail
echo.
echo SUCCESS. Next run the model-download BAT, then start.bat.
if not defined IMAGEEDIT_NO_PAUSE pause
popd
exit /b 0

:try_launcher
py -%1 -c "import struct,sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
if not "%ERRORLEVEL%"=="0" exit /b 1
set "BASE_PY=py"
set "BASE_ARGS=-%1"
echo Using installed Python %1.
exit /b 0

:check_environment
"%ENV_PY%" -c "import struct,sys; assert (3,10) <= sys.version_info[:2] <= (3,13), 'Python 3.10 through 3.13 required'; assert struct.calcsize('P') == 8, '64-bit Python required'; assert sys.prefix != sys.base_prefix, 'Invalid project venv'; print('Python',sys.version.split()[0]); print('Environment',sys.executable)"
exit /b %ERRORLEVEL%

:missing_project
echo ERROR: Missing project files. Fully extract the repository first.
goto :fail
:missing_python
echo ERROR: No working 64-bit Python 3.10 through 3.13 was found.
echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
echo Enable Python Launcher or Add Python to PATH, then rerun this BAT.
echo Your existing .venv has not been moved or deleted.
goto :fail
:backup_failed
echo ERROR: Could not back up .venv. Close project Python processes and retry.
goto :fail
:bad_environment
echo ERROR: The new .venv failed validation. See the error above.
goto :fail
:usage
echo Usage: double-click this file, or run it with cpu as the only argument.
goto :fail
:fail
echo.
echo FAILED. See the error above. No model download was started.
echo You can rerun after fixing the error. Backups are never deleted automatically.
echo For PyTorch builds see https://pytorch.org/get-started/locally/
echo Set TORCH_INDEX_URL to override the package index, or use the cpu argument.
if not defined IMAGEEDIT_NO_PAUSE pause
popd
exit /b 1
