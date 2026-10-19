# 来源与使用说明

本仓库基于用户提供的 Imageediting-for-personal-use 项目及其本地改进版整理。由于当前工作副本不含原始 Git 历史或明确许可证文件，此次整理不擅自为全部代码添加开源许可证。公开再分发或另行授权前应核对原项目及第三方权利信息。

模型权重不包含在 Git 中。模型来源与许可应分别在提供方核对：

- CIDAS/clipseg-rd64-refined
- stable-diffusion-v1-5/stable-diffusion-inpainting
- timbrooks/instruct-pix2pix
- openai/clip-vit-base-patch32
- IDEA-Research/grounding-dino-tiny
- facebook/sam-vit-base
- LaMa 检查点由 Sanster/models 发布，下载地址与摘要记录于 config.py 及下载后生成的 provenance.json；适配参考 Sanster/IOPaint 的 LaMa 实现。

新增 137 张图像及参考实例掩码来自 COCO val2017。各图片的来源地址、Flickr 地址、许可证和标注编号记录在 data/manifest.json；整体提取记录见 data/coco_provenance.json。COCO 参考标注由提供方生成，尚不表示本地逐图复核已完成。

原有 cat、street、person 三张演示图继承自原项目，尚未独立核验其授权来源；它们保留用于项目内部研究演示。项目代码的许可不自动覆盖任何第三方图片、模型或文档中使用的素材。
