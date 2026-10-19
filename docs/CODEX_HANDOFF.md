# 在另一台电脑接手项目

这份说明帮助新的 Codex 会话了解项目。代码以当前仓库为准，说明不代表另一台电脑已装好环境、模型或具备相同硬件条件。

## 获取代码与打开项目

仓库：https://github.com/cyz060718-tech/imageediting-research

长期在两台电脑间开发时，推荐分别克隆同一仓库：

```powershell
git clone https://github.com/cyz060718-tech/imageediting-research.git
```

在另一台电脑的 Codex 中添加本地项目文件夹，选克隆出的 `imageediting-research`，即直接包含 `app.py`、`requirements.txt`、`bootstrap.py` 的那一层，并设为主要文件夹。

没有 Git 时，也可在 GitHub 页面选择 Code → Download ZIP，完整解压后打开该目录。ZIP 副本不会通过 Git 自动拉取后续改动。

对话同步与文件访问分开；同步会话不会把整个本地文件夹传到另一台电脑。参见 [OpenAI 官方项目说明](https://learn.chatgpt.com/docs/projects)。

## 给新 Codex 的第一条消息

> 请先阅读 README.md、docs/CODEX_HANDOFF.md、docs/windows-install.md 和当前测试代码，确认工作目录与 Git 状态。这个项目已经经过多轮调试，请在现有实现上继续。先核对本机 Python、运行环境、模型路径和网页启动，保留已有的实例选区、食物保护、改色纹理保留、局部编辑和选区外保留逻辑。当前目标是让本机可靠运行项目，再按我后续的具体要求修改。请区分实际验证与推测。

阅读、解释或编辑代码不必先下载全部模型。执行推理时需要对应权重。

## 当前项目事实

- 大学生创新训练的局部图像语义编辑原型，使用公开预训练权重，没有训练新的基础模型。
- 指令得到动作、目标与新内容；默认规则解析，可选接入兼容语言模型接口。
- Grounding DINO 定位目标，SAM 生成实例掩码；保留实例、清理断开小块、处理重复区域。容器改色时扣除合理的食物或汤区域。
- 默认保真改色采用 Lab 颜色处理，删除采用 LaMa，替换采用逐实例局部重绘。保真改色使用保护后的原始选区，扩大边界时不能把保护内容选回去。
- 网页已有 20 个中文开发示例。开发与预留测试数据分开；正式评价依赖独立标注，自动选区不能作为自己的真实标注。
- 更换图片、指令或选区方式后清除旧结果；人工掩码与图片同尺寸，不能把原照片当掩码。
- 用户是成员 B，关注自动选区与内容保护的原理、原因和证据。个人实际贡献按真实参与说明，不能从分工推断独立完成了全部代码。

## 环境和模型迁移

Windows 10/11 x64 和标准 x64 Python 3.10—3.13 的步骤见 `docs/windows-install.md`。先运行 `安装环境.bat`，再运行 `下载模型.bat`，最后运行 `start.bat`。

`.venv` 在新机器重新创建，它关联原机器的解释器、路径和二进制包。已经下载并校验过的模型可通过移动硬盘或局域网复制，默认结构如下：

```text
imageediting-research/
  models/
    clipseg/
    inpaint/
    ip2p/
    clip/
  models_extra/
    grounding_dino/
    sam/
    lama/
      big-lama.pt
```

也可设置 `IMAGE_EDIT_MODEL_DIR` 与 `IMAGE_EDIT_EXTRA_MODEL_DIR` 指向新机器已有的模型目录。模型、`.venv`、输出与本地密钥不随 Git 迁移。macOS/Linux 也可打开仓库阅读和改代码，运行依赖需要按系统重新验证，不能执行 Windows BAT。

## 验证范围与下载问题

2026 年 10 月 19 日的 [安装验证记录](https://github.com/cyz060718-tech/imageediting-research/actions/runs/39255372441)覆盖 5 个独立 Windows runner 组合：Python 3.10—3.13，另含 Windows Server 2022 与 Python 3.11。每组通过全新 CPU 安装、67 项检查和真实网页 HTTP 启动。历史的 67 项不含之后新增的 LaMa 下载检查；最新数量和状态以实际执行和仓库 Actions 为准。

云端没有下载全部模型，也没有对不同 NVIDIA 显卡做实体推理测试。开发机 CUDA 运算曾通过，其他机器仍需实际检查。模型齐全与编辑效果正确也需另外验证。

曾收到 LaMa 下载的 `urllib` TLS 握手报 `FileNotFoundError` 日志，其余模型已完成。随后同一地址 HTTPS 重试成功，无法据此断言根因。下载脚本改用 HTTPX，保留 TLS 校验，增加有限重试、流式摘要校验和 `--source` 离线导入；不应关闭证书验证或重下全部模型。

常用验证命令：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools/check_web_start.py
.venv\Scripts\python.exe dataset.py validate --split dev
```

## 两台电脑同步

开始前检查 `git status`，处理未保存的修改后，再获取 GitHub 最新代码。完成修改后检查差异、做相关验证，并按用户要求提交、推送；另一台电脑拉取后即可获得这些改动。尚未推送的内容不会自动出现在另一台电脑，同时修改同一文件时可能需要合并冲突。
