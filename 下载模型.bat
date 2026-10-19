@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUTF8=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not errorlevel 0 exit /b 1
set "ENV_PY=%CD%\.venv\Scripts\python.exe"
set "MODEL_NAMES=clipseg inpaint ip2p clip"
if /i "%~1"=="minimal" set "MODEL_NAMES=inpaint"
if not "%~1"=="" if /i not "%~1"=="minimal" goto :usage

rem Keep this file ASCII with CRLF for compatibility with Windows cmd.exe.
echo ==================================================
echo Image editor - Download models
echo Several GB of free space are required. Existing downloads will be reused.
echo See README.md for Chinese instructions.
echo ==================================================
if not exist "%ENV_PY%" goto :missing_environment
if not exist "download_models.py" goto :missing_project
if not exist "tools\download_advanced_models.py" goto :missing_project
if not exist "tools\download_lama.py" goto :missing_project
"%ENV_PY%" -c "import bootstrap; import huggingface_hub,torch,numpy,PIL; import config" >nul 2>nul
if not "%ERRORLEVEL%"=="0" goto :missing_environment
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"
echo Hugging Face endpoint: %HF_ENDPOINT%
echo Set HF_ENDPOINT=https://huggingface.co to use the official endpoint.
echo.
echo [1/4] Download base models: %MODEL_NAMES%
"%ENV_PY%" "download_models.py" %MODEL_NAMES%
if not "%ERRORLEVEL%"=="0" goto :fail
echo [2/4] Download Grounding DINO and SAM...
"%ENV_PY%" "tools\download_advanced_models.py"
if not "%ERRORLEVEL%"=="0" goto :fail
echo [3/4] Download and verify LaMa...
"%ENV_PY%" "tools\download_lama.py"
if not "%ERRORLEVEL%"=="0" goto :fail
echo [4/4] Check model directories...
"%ENV_PY%" -c "import config; names='%MODEL_NAMES%'.split()+list(config.EXTRA_MODEL_REPOS); [print(n,config.model_path(n)) for n in names]; assert config.LAMA_MODEL_PATH.is_file(), 'LaMa file missing'; print('Model directory checks passed')"
if not "%ERRORLEVEL%"=="0" goto :fail
echo.
echo SUCCESS. Run start.bat and open the address shown in the terminal.
if not defined IMAGEEDIT_NO_PAUSE pause
popd
exit /b 0

:missing_environment
echo ERROR: Project environment is missing or unusable. Run the install BAT first.
goto :fail
:missing_project
echo ERROR: Missing model download scripts. Fully extract the repository first.
goto :fail
:usage
echo Usage: double-click for all models, or add minimal for default quality mode.
goto :fail
:fail
echo.
echo FAILED. See the error above. You can rerun to resume Hugging Face downloads.
echo An incomplete LaMa file will be downloaded again.
echo If models already exist, check IMAGE_EDIT_MODEL_DIR and IMAGE_EDIT_EXTRA_MODEL_DIR.
if not defined IMAGEEDIT_NO_PAUSE pause
popd
exit /b 1
