# DINO 部署分析代码

本目录对应课程报告中的 DINO 系列部署与资源分析部分。

运行：

```bash
python src/dino_deployment_demo.py
```

输入为 `data/model_registry.csv`，输出写入 `results/`。脚本生成：

- 不同精度下的理论权重内存；
- 不同输入分辨率和 patch size 下的 token 数及注意力矩阵相对规模；
- 基于显存、模型用途和许可约束的候选模型列表；
- EASE 使用 DINOv2-ViT-L/14 时的特征尺寸与存储估算。

这些数据属于透明、可复核的公式计算和源码审计结果，不代表本机 GPU 的真实推理延迟或峰值显存。

EASE 案例详见 `cases/EASE_DINOv2_case.md`。
