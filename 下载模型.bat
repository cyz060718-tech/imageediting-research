@echo off
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUTF8=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "ENV_PY=%CD%\.venv\Scripts\python.exe"
set "MODEL_NAMES=clipseg inpaint ip2p clip"
if /i "%~1"=="minimal" set "MODEL_NAMES=inpaint"
if not "%~1"=="" if /i not "%~1"=="minimal" goto :usage

echo ==================================================
echo 图像编辑网页  下载模型
echo 默认下载全部网页模式的模型，需预留数 GB 空间。
echo 已缓存的 Hugging Face 文件会复用，请耐心等待。
echo ==================================================
if not exist "%ENV_PY%" (
    echo [错误] 未找到项目环境，请先双击 安装环境.bat。
    goto :fail
)
if not exist "download_models.py" goto :missing_project
if not exist "tools\download_advanced_models.py" goto :missing_project
if not exist "tools\download_lama.py" goto :missing_project
"%ENV_PY%" -c "import bootstrap; import huggingface_hub,torch,numpy,PIL; import config" >nul 2>nul
if errorlevel 1 (
    echo [错误] 项目依赖不可用，请先重新运行 安装环境.bat。
    goto :fail
)
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"
echo Hugging Face 下载地址：%HF_ENDPOINT%
echo 如需直连官网，请在命令行设置 HF_ENDPOINT=https://huggingface.co。
echo.

echo [1/4] 下载基础模型：%MODEL_NAMES%
"%ENV_PY%" "download_models.py" %MODEL_NAMES%
if errorlevel 1 goto :fail
echo [2/4] 下载 Grounding DINO 和 SAM 选区模型...
"%ENV_PY%" "tools\download_advanced_models.py"
if errorlevel 1 goto :fail
echo [3/4] 下载并校验 LaMa 删除模型...
"%ENV_PY%" "tools\download_lama.py"
if errorlevel 1 goto :fail
echo [4/4] 检查模型目录...
"%ENV_PY%" -c "import config; names='%MODEL_NAMES%'.split()+list(config.EXTRA_MODEL_REPOS); [print(n,config.model_path(n)) for n in names]; assert config.LAMA_MODEL_PATH.is_file(), 'LaMa 文件缺失'; print('模型目录检查通过')"
if errorlevel 1 goto :fail
echo.
echo 下载完成。现在双击 start.bat，打开终端显示的网址。
if not defined IMAGEEDIT_NO_PAUSE pause
popd
exit /b 0

:missing_project
echo [错误] 缺少模型下载脚本，请完整解压仓库。
goto :fail
:usage
echo 用法：双击下载全部模型；或执行 下载模型.bat minimal 仅下载默认保真模式所需模型。
goto :fail
:fail
echo.
echo [失败] 模型下载未完成，请查看上方的具体错误。
echo Hugging Face 下载可在重试时复用缓存；LaMa 未完成的文件会重新下载。
echo 如已有模型，请检查 IMAGE_EDIT_MODEL_DIR 和 IMAGE_EDIT_EXTRA_MODEL_DIR 路径。
if not defined IMAGEEDIT_NO_PAUSE pause
popd
exit /b 1
