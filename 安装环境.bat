@echo off
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUTF8=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "ENV_PY=%CD%\.venv\Scripts\python.exe"

echo ==================================================
echo 图像编辑网页  安装运行环境
echo 本脚本仅创建或更新当前项目的 .venv，不下载模型。
echo 需要已安装 64 位 Python 3.10 至 3.13。
echo ==================================================

if not "%~1"=="" if /i not "%~1"=="cpu" goto :usage
if not exist "requirements.txt" goto :missing_project
if not exist "bootstrap.py" goto :missing_project

echo [1/5] 检查或创建虚拟环境...
if exist "%ENV_PY%" goto :environment_ready
where py >nul 2>nul
if errorlevel 1 goto :try_path_python
for %%V in (3.12 3.11 3.10 3.13) do (
    py -%%V -c "import struct,sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
    if not errorlevel 1 (
        py -%%V -m venv ".venv"
        if errorlevel 1 goto :fail
        goto :environment_ready
    )
)

:try_path_python
where python >nul 2>nul
if errorlevel 1 goto :missing_python
python -c "import struct,sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) and struct.calcsize('P') == 8 else 1)" >nul 2>nul
if errorlevel 1 goto :missing_python
python -m venv ".venv"
if errorlevel 1 goto :fail

:environment_ready
"%ENV_PY%" -c "import struct,sys; assert (3,10) <= sys.version_info[:2] <= (3,13), '需要 Python 3.10 至 3.13'; assert struct.calcsize('P') == 8, '需要 64 位 Python'; assert sys.prefix != sys.base_prefix, '当前 .venv 无效'; print('Python',sys.version.split()[0]); print('环境',sys.executable)"
if errorlevel 1 goto :bad_environment

echo [2/5] 更新安装工具...
"%ENV_PY%" -m pip install --upgrade pip setuptools wheel --retries 3 --timeout 60
if errorlevel 1 goto :fail

set "USE_CUDA=0"
set "TASK_TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if /i "%~1"=="cpu" goto :torch_selected
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :torch_selected
nvidia-smi -L >nul 2>nul
if errorlevel 1 goto :torch_selected
set "USE_CUDA=1"
set "TASK_TORCH_INDEX=https://download.pytorch.org/whl/cu130"
:torch_selected
if defined TORCH_INDEX_URL set "TASK_TORCH_INDEX=%TORCH_INDEX_URL%"
echo [3/5] 检查 PyTorch，安装源为 %TASK_TORCH_INDEX%
if "%USE_CUDA%"=="0" echo 当前使用 CPU 模式，生成图片可能较慢。
"%ENV_PY%" -c "import bootstrap; import torch,torchvision; assert %USE_CUDA% or torch.version.cuda is None, '需要 CPU 构建'; device='cuda' if %USE_CUDA% else 'cpu'; x=torch.ones((16,16),device=device); assert float((x@x).sum().cpu()) == 4096.0; print('已有 PyTorch 可用',torch.__version__,device)" >nul 2>nul
if not errorlevel 1 goto :dependencies
"%ENV_PY%" -m pip install --upgrade --force-reinstall torch torchvision --index-url "%TASK_TORCH_INDEX%" --retries 3 --timeout 60
if errorlevel 1 goto :fail

:dependencies
echo [4/5] 安装项目依赖...
"%ENV_PY%" -m pip install -r "requirements.txt" --retries 3 --timeout 60
if errorlevel 1 goto :fail
"%ENV_PY%" -m pip check
if errorlevel 1 goto :fail

echo [5/5] 验证导入与实际运算，不加载模型...
"%ENV_PY%" -c "import bootstrap; import torch,torchvision,gradio,diffusers,transformers,accelerate,safetensors,huggingface_hub,cv2,lpips,numpy,PIL,pycocotools; import app; device='cuda' if %USE_CUDA% else 'cpu'; x=torch.ones((16,16),device=device); assert float((x@x).sum().cpu()) == 4096.0; print('PyTorch',torch.__version__,'运行模式',device); print('网页代码与依赖导入通过')"
if errorlevel 1 goto :fail
echo.
echo 安装完成。接下来双击 下载模型.bat，完成后双击 start.bat。
if not defined IMAGEEDIT_NO_PAUSE pause
popd
exit /b 0

:missing_project
echo [错误] 缺少项目文件。请先完整解压仓库，再运行本脚本。
goto :fail
:missing_python
echo [错误] 未找到可用的 64 位 Python 3.10 至 3.13。
echo 请从 https://www.python.org/downloads/windows/ 安装 Python，并启用 PATH 或 Python Launcher。
goto :fail
:bad_environment
echo [错误] 当前 .venv 无法使用。请先将它重命名备份，再重新运行本脚本。
goto :fail
:usage
echo 用法：双击安装环境，或在命令行执行 安装环境.bat cpu 使用 CPU 模式。
goto :fail
:fail
echo.
echo [失败] 环境安装未完成，请查看上方的具体错误。
echo 网络中断后可重新运行；本脚本不会自动删除已有环境。
echo PyTorch 运算失败时，请按官方说明选择安装源：https://pytorch.org/get-started/locally/
echo 可通过 TORCH_INDEX_URL 指定安装源，或运行 安装环境.bat cpu。
if not defined IMAGEEDIT_NO_PAUSE pause
popd
exit /b 1
