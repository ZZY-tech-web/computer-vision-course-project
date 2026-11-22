# Python 图像处理课后作业

本目录包含三个可以独立运行的实验：

1. `task1_hist_rotate_scale.py`：直方图均衡化、旋转和缩放；
2. `task2_smooth_sharpen.py`：图像平滑、锐化和边缘响应；
3. `task3_color_spaces.py`：不同颜色空间转换及通道可视化。

一次运行全部任务：

```bash
python src/run_all.py
```

程序会先生成可重复的示例图像 `input/sample_image.png`，再将结果保存到 `results/`。也可以替换输入图片后分别运行三个任务脚本。
