@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUTF8=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not errorlevel 0 exit /b 1
if not exist ".venv\Scripts\python.exe" (
  echo Create a local .venv and install dependencies first. See README.md.
  if not defined IMAGEEDIT_NO_PAUSE pause
  popd
  exit /b 1
)
".venv\Scripts\python.exe" app.py
set "TASK_EXIT=%ERRORLEVEL%"
if not "%TASK_EXIT%"=="0" (
  echo Web app failed. Run the install BAT first and check the error above.
  if not defined IMAGEEDIT_NO_PAUSE pause
)
popd
exit /b %TASK_EXIT%
