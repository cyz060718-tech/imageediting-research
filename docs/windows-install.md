# Windows 安装与兼容性说明

## 使用条件

支持 Windows 10/11 x64 上的标准 64 位 CPython 3.10、3.11、3.12、3.13。Python 可来自 Python 官网或已有 Conda；安装器会验证实际执行的版本、架构和虚拟环境状态，不依赖开发者电脑的用户名、盘符或 Conda 路径。ARM64 原生解释器、32 位解释器和自由线程 Python 不属于本安装器的支持范围。

没有合适的 Python 时，脚本会停止并给出安装地址。已经有兼容版本时不要求再安装 3.12，也不会安装或卸载系统 Python。需要 Microsoft Visual C++ x64 运行库；若缺失，可从 [Microsoft 官方地址](https://aka.ms/vc14/vc_redist.x64.exe)安装。

## 正常安装顺序

完整解压仓库，进入同时包含 app.py、requirements.txt、bootstrap.py 和 tools 文件夹的目录：

1. 双击 `安装环境.bat`，等待出现 `SUCCESS`。
2. 双击 `下载模型.bat`，等待模型下载完成。
3. 双击 `start.bat`，打开终端显示的本地网址。

脚本必须放在项目目录中执行，不要只下载两个 BAT 后直接双击。批处理使用 ASCII 文本与 CRLF 换行，窗口操作提示为英文；中文说明见本文。

## Python 与环境处理

有效的项目 `.venv` 会直接复用。新建时先尝试 `py` 启动器中的 3.12、3.11、3.10、3.13，再尝试 PATH 中的 `python`。新版 Python 启动器返回的负数错误码同样是失败，只有 0 才表示成功。

如果希望明确指定解释器，可在命令提示符中执行：

```bat
set "IMAGEEDIT_PYTHON=D:\你的Python目录\python.exe"
安装环境.bat
```

这个变量只用于需要新建或重建环境的情况。若现有 `.venv` 有效，仍会复用它。失效环境会在找到可用解释器后重命名为 `.venv.backup_时间戳`，随后新建 `.venv`；备份不自动删除，并已列入 Git 忽略规则。没有可用解释器时，不移动原环境。

## 显卡与 CPU 选择

自动模式读取 `nvidia-smi` 中的驱动版本和计算能力，根据实际信息选择安装源，不根据显卡品牌名称猜测，也不因发现 NVIDIA 驱动就固定安装 CUDA 13。

| 检测情况 | 自动处理 |
| --- | --- |
| 没有 NVIDIA 驱动或查询失败 | 选择 CPU |
| 较早的支持范围内显卡 | 使用 cu118 与匹配的 Torch 2.7.1、torchvision 0.22.1 |
| 较新显卡，驱动满足所选构建条件 | 按条件选择 cu126、cu128 或 cu130，匹配 Torch 2.11.0、torchvision 0.26.0 |
| Blackwell 等新架构而驱动不足 | 使用 CPU，避免误装不支持新架构的旧 CUDA 构建 |
| 自动 GPU 安装或运算验证失败 | 明确提示并尝试 CPU 构建 |

这些选择是保守的初筛。每次都会运行矩阵乘法、卷积和 torchvision NMS；CUDA 路径还会验证半精度卷积。必须实际执行成功才算可用。已有能通过验证的 PyTorch 直接复用，不要求把它降级成默认安装版本。显卡显存、模型大小和任务复杂度仍会影响运行速度及是否能够加载模型。

需要强制 CPU 时：

```bat
安装环境.bat cpu
```

需要自行指定官方构建时：

```bat
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
安装环境.bat
```

手动指定源表示自行选择构建，失败后会停止，不会自动换成 CPU 覆盖你的选择。指定 CPU 模式时只能同时使用 CPU 安装源。版本组合依据 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/)，驱动兼容性依据 [NVIDIA 官方说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 网络与模型

pip 的网络设置遵循本机 pip 配置和 `PIP_INDEX_URL`。下载依赖只接受二进制 wheel，不要求普通使用者配置 C++ 或 Rust 编译环境。网络失败会停止，可以稍后重试；不会把下载失败当作安装成功。

模型默认经 `hf-mirror.com` 下载；直连 Hugging Face 时，在命令提示符中设置：

```bat
set "HF_ENDPOINT=https://huggingface.co"
下载模型.bat
```

`下载模型.bat minimal` 只准备默认保真流程需要的 Inpaint、Grounding DINO、SAM 和 LaMa。模型路径从当前项目出发；可通过 `IMAGE_EDIT_MODEL_DIR` 和 `IMAGE_EDIT_EXTRA_MODEL_DIR` 使用已有模型目录。项目权重不会随 GitHub 仓库一起分发。

## 怎样确认安装情况

如果只有 LaMa 下载失败，已经完成的其他模型可以复用。脚本会保留 TLS 校验并重试 LaMa 下载；也可以从另一台电脑复制已校验的模型，再在项目目录执行以下命令导入，路径替换为你实际的文件位置：

```bat
.venv\Scripts\python.exe tools\download_lama.py --source "E:\已有模型\big-lama.pt"
```

离线导入也必须通过官方摘要校验。直接把文件复制到 `models_extra\lama\big-lama.pt` 后，运行下载脚本同样会检查摘要。

`安装环境.bat` 的最后一步导入网页与全部必要依赖，并执行实际计算；出现 `SUCCESS` 表示这次环境检查完成。它不代表已经下载模型或验证过所有编辑效果。下载脚本也必须成功完成，才能启动对应网页模式。

仓库的 `Windows clean installation` 自动检查会在干净的 GitHub Windows runner 上，用 3.10、3.11、3.12、3.13 分别运行公开安装 BAT 的 CPU 路径，随后执行回归测试、启动真实网页并验证本地 HTTP、校验开发样本；另设 Windows Server 2022 与 Python 3.11 组合。结果见仓库 Actions。云端 runner 没有 NVIDIA 显卡，所以 GPU 选择与回退另有离线测试，真实 CUDA 验证仅覆盖实际执行过的硬件，不能冒充所有显卡实测。

维护者检查命令：

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe dataset.py validate --split dev
```

中文路径、空格、`&` 与 `!` 路径、缺少 Python 版本、负数错误码、失效环境备份、CPU 模式和模型下载中断，都有离线批处理回归检查。
