# 图像局部语义编辑研究原型

大学生创新训练项目。输入原图和一句明确指令，进行局部改色、删除或对象替换。此仓库以已验证的改进版代码为主体，保留对照方法、测试、数据和最终汇报资料。

## 当前流程

1. 规则解析指令，得到动作、目标与新内容；可选接入兼容接口的语言模型。
2. Grounding DINO 检测目标，SAM 生成实例选区；可切换 CLIPSeg 或手绘选区。
3. 改色采用 Lab 颜色变换，删除采用 LaMa 背景修补，替换采用逐实例局部重绘。
4. 替换结果通过目标检测作诊断，未确认区域最多重试两轮，并保存候选与选择依据。
5. Gradio 页面展示选区与结果，运行记录写入本地 outputs/。

基础模型来自公开预训练模型。本项目的工作是流程改进与评价，没有训练新的基础模型。

## 快速开始

### Windows 双击运行

适用于 Windows 10/11 x64 和标准 64 位 CPython 3.10 至 3.13，建议选 3.11 或 3.12。不是 ARM64 或自由线程 Python 环境。完整解压仓库后依次双击：

1. **安装环境.bat**：创建当前项目的 `.venv`，安装 PyTorch 和项目依赖，并检查实际运算和网页代码导入。不下载模型，不修改系统 Python。
2. **下载模型.bat**：下载基础编辑、选区和删除模型，并进行目录检查；LaMa 下载脚本还会校验文件摘要。默认包含 CLIPSeg、Inpaint、IP2P、CLIP、Grounding DINO、SAM 和 LaMa，以便使用网页各模式。
3. **start.bat**：启动网页，打开终端显示的本地网址。

两个安装脚本都会在错误时停止，窗口保留具体提示；网络中断后可重新运行。Hugging Face 文件复用下载缓存，LaMa 未完成的文件重新下载。脚本不会下载或重装 Python 本身。批处理内使用英文提示，避免不同 Windows 编码设置把中文拆成命令；操作说明以本页中文为准。

安装脚本会依次尝试已安装的 Python 3.12、3.11、3.10、3.13，再尝试 PATH 中的 Python。**不需要专门安装 3.12**，只有真正执行成功的解释器才会被选用，新版 Python 启动器的负数错误码也会被视为失败。有效的 `.venv` 会直接复用；若环境已失效，脚本会先找到可用解释器，再把原环境重命名为 `.venv.backup_时间戳`，随后创建新环境。不会自动删除备份；若没有可用解释器，原环境保持原样。

安装环境根据 NVIDIA 驱动和显卡计算能力选择官方 cu118、cu126、cu128、cu130 或 CPU 构建。较早的显卡使用 Torch 2.7.1 与 torchvision 0.22.1，新路径使用 Torch 2.11.0 与 torchvision 0.26.0；已经能通过运算验证的环境直接复用。验证包括矩阵乘法、卷积和 torchvision NMS，而不只是能否导入。自动模式下 GPU 安装或验证失败会明确提示并尝试 CPU；手动指定 `TORCH_INDEX_URL` 时失败会停止，保留用户的构建选择。安装源和版本组合参考 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/)。也可以在命令提示符中使用：

```bat
安装环境.bat cpu
下载模型.bat minimal
```

`cpu` 强制选择 CPU 模式。`minimal` 只下载默认保真流程所需的 Inpaint、Grounding DINO、SAM 和 LaMa；其他网页模式需要再次运行不带参数的下载脚本。已有模型路径环境变量和 `HF_ENDPOINT` 均会保留。自动化运行可设置 `IMAGEEDIT_NO_PAUSE=1` 关闭末尾暂停。

如果同时安装了多个 Python，可先设置 `IMAGEEDIT_PYTHON` 为所需解释器的完整路径。依赖使用已发布的二进制 wheel，避免要求使用者额外安装编译器。新电脑若缺少 Visual C++ 运行库，需要安装 [Microsoft 官方 x64 运行库](https://aka.ms/vc14/vc_redist.x64.exe)。更详细的条件、故障定位和测试范围见 [Windows 安装说明](docs/windows-install.md)。

### 手动安装

建议使用独立的 Python 3.11 或 3.12 环境。以下为 Windows PowerShell 示例；其他系统使用对应的虚拟环境激活命令。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装适合当前平台的 torch 与 torchvision，再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
python download_models.py inpaint
python tools/download_advanced_models.py
python tools/download_lama.py
python app.py
```

打开终端给出的本地 URL。也可在环境安装完成后双击 start.bat。权重不随 Git 仓库上传，下载脚本会将其放到 models/ 和 models_extra/。

默认保真流程需要 inpaint、grounding_dino、sam、LaMa。选择 CLIPSeg、IP2P 或学习型指标时，还需要按需下载相应模型：

```powershell
python download_models.py clipseg ip2p clip
```

下载脚本默认使用既有的 Hugging Face 镜像配置；需要直连官方站点时，先设置 `$env:HF_ENDPOINT='https://huggingface.co'`。已有本地权重可以通过 IMAGE_EDIT_MODEL_DIR 和 IMAGE_EDIT_EXTRA_MODEL_DIR 指向各自目录，无需复制。

依赖使用固定版本，Windows 安装另有多 Python 版本的全新安装自动检查。是否通过请以仓库 Actions 中对应提交的检查结果为准；CPU 云端测试不能替代每种实体显卡的实测。可选语言模型功能还需单独安装 openai 并设置 LLM_API_KEY 等环境变量；默认规则模式不需要密钥。不要将密钥提交到 Git。

## 使用方式

上传图片，输入明确指令，例如“把外套改成蓝色”“去掉停车标志”“把猫换成老虎”。先预览选区，确认红色范围，再执行编辑。已有 20 个开发示例位于“选择示例图片”折叠区域，点击后填入中文编辑指令。中文展示指令记录于开发样本的 `example_instruction_zh` 字段，实验原始指令仍由 `instruction` 字段记录。

自动选区不准时，使用手绘或同尺寸二值掩码，白色编辑、黑色保留。原照片不能直接作为黑白掩码。更换图片、指令或选区方式后会清除旧结果。

## 验证与实验

```powershell
python -m unittest discover -s tests -v
python dataset.py validate --split dev
python eval.py --split dev --settings experiment.json
```

数据保留 20 张开发图和 120 张预留测试图。开发集用于调试；测试集不参与调参。正式测试前先审核任务范围与 GT 标注、固定版本和参数，再执行冻结及评价。已有 COCO 提供方标注不等于本地独立复核已完成。

```powershell
python dataset.py freeze --settings experiment.json
python eval.py --split test --settings experiment.json
```

test_lock.json 包含本机解析后的文件路径，应在实际实验环境重新生成，不提交到 Git。正式实验应保留失败分母并组织独立人工评价。

## 已有证据与限制

历史开发验证中，20 例通过真实接口完成生成，输入一致、原尺寸保持和融合权重为零位置保持均为 20/20，41 项回归检查通过。这些是开发阶段工程检查，不是 100% 语义编辑成功率，也不是本次发布新增的模型推理结果。

密集杯子、远处动物、透明材质、原对象残留和删除支撑物后场景不合理等问题仍存在。检测到新类别不能证明数量、形态和自然度正确。120 张预留测试图尚未完成正式生成评价与独立盲评。

## 目录

```text
app.py                 网页入口
config.py              参数和可移植路径
core/                  解析、选区、三类编辑、结果检查和指标
tests/                 现有回归检查
tools/                 模型下载、对照与评价辅助
data/                  数据清单、原图、参考掩码与来源
docs/reports/          最终 PPT 和研究路径手册
docs/meeting/          五位成员及组长的会议汇报
docs/evidence/         开发阶段汇总与检查图
docs/STATUS.md         当前研究状态
docs/CLEANUP.md        精简范围与本地保留说明
```

原始开发目录、环境、模型、完整候选和历史结果仍保留在原位置。仓库未打包这些可再生成内容，也未改变当前编辑算法。模型、数据和原项目代码的来源说明见 [NOTICE.md](NOTICE.md)。
