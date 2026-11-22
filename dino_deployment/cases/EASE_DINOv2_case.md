# EASE 中 DINOv2 稠密特征使用案例审计

## 分析范围

本案例审阅工作区 `EASE-main` 的源码，说明 DINOv2 如何从通用视觉 backbone 进入一个具体项目。EASE 的研究任务是无监督伪装目标检测，但本案例不展开该任务本身，只保留与 DINOv2 输入、稠密特征、原型和检索有关的工程链路。

课程案例的复现基于 EASE 官方仓库。本仓库不重复发布第三方源码、模型权重和公开数据集，只保留与 DINOv2 特征链路有关的源码审计、参数说明和可复现命令。下述尺寸由代码与公式核对得到，不应当作 GPU 性能实测结果。

官方项目：<https://github.com/xiaohainku/EASE>

## 复现入口

仓库提供 `reproduce_ease.sh` 作为命令入口。准备好 EASE 官方代码、Stable Diffusion 环境图像、测试集和可选的 HQ-SAM 权重后，可按以下方式运行：

```bash
bash reproduce_ease.sh /path/to/EASE \
  /path/to/stable-diffusion-images \
  /path/to/TestDataset \
  /path/to/hq-sam-checkpoint.pth
```

脚本依次生成环境原型、运行无监督检索，并在提供 HQ-SAM 权重时执行边界精修。实际依赖、数据组织和权重许可应以 EASE 官方说明为准。

## DINOv2 特征链路

1. `gen-envir-proto.py` 和两个检索脚本都从本地 `hubconf.py` 构造 `dinov2_vitl14()`。该构造函数默认下载 LVD-142M 预训练权重。
2. 图像被缩放到 476×476，并使用 ImageNet 均值和标准差归一化。476 能被 patch size 14 整除，因此每边得到 34 个 patch。
3. `get_intermediate_layers(img_tensor, reshape=True)[0]` 返回最后一层 patch 特征。ViT-L/14 的输出形状为 1×1024×34×34，共 1156 个 1024 维局部描述子。
4. 原型生成脚本对空间维求平均，保存 1024 维全局向量；同时把 34×34 的特征图展开为 1156×1024 的局部原型并保存为 NPY 文件。
5. 检索脚本对向量做 L2 归一化，再用 FAISS `IndexFlatIP` 搜索。归一化后的内积等价于余弦相似度，因此同一组 DINOv2 特征可支持全局原型筛选和局部 patch 匹配。
6. 相似度图经过归一化、KDE 自适应阈值和插值后交给后续任务。这个链路说明 DINOv2 提供的是可迁移的稠密表示，不是直接输出检测结果的端到端检测器。

## 尺寸与存储估算

- 单张图的 patch 数：34×34 = 1156。
- 单张图的稠密特征元素数：1156×1024 = 1,183,744。
- 单张图以 float32 保存局部特征约占 4.52 MiB。
- 单张图的 1024 维全局原型以 float32 保存约占 0.0039 MiB。
- 默认每类使用 32 张图构建原型，单类局部特征文件合计约 144.5 MiB，不含 NPY 头和文件系统开销。

数值明细见 `results/ease_dinov2_case.csv`。

## 复现风险与改进建议

- 三个顶层脚本虽然提供 `--dino` 参数，但模型构造仍固定为 `dinov2_vitl14()`；若只修改参数，通道数可能与实际模型不一致。应使用模型字典按参数选择构造函数。
- 模型创建后没有显式调用 `eval()`。特征提取阶段应使用 `model.eval()` 和 `torch.inference_mode()`，并固定权重来源与 SHA-256。
- 默认输入尺寸、特征维度和本地原型 reshape 分别由多个变量计算。运行前应断言输入边长可被 patch size 整除，并根据实际特征张量读取空间尺寸，避免参数改动后静默错位。
- 完整 EASE 还依赖 FAISS、生成图像、测试数据和其他基础模型。课程报告把该仓库作为 DINOv2 稠密特征迁移案例，不把论文指标写成本机复现结果。
