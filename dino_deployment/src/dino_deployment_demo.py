#!/usr/bin/env python3
"""Reproducible resource and deployment analysis for the DINO course project.

The script does not download model weights and does not claim measured inference
latency. It reproduces three report artifacts from a small, auditable registry:
theoretical weight memory, attention-scaling estimates, and transparent model
selection suggestions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "model_registry.csv"
RESULTS = ROOT / "results"


def load_rows() -> list[dict[str, str]]:
    with REGISTRY.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def resource_estimates(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result = []
    for row in rows:
        p = float(row["parameters_m"])
        result.append(
            {
                "model": row["model"],
                "model_type": row["model_type"],
                "parameters_m": f"{p:g}",
                "fp32_weight_mb": f"{p * 4:.1f}",
                "fp16_weight_mb": f"{p * 2:.1f}",
                "int8_weight_mb": f"{p:.1f}",
                "license": row["license"],
            }
        )
    return result


def scaling_estimates() -> list[dict[str, object]]:
    result = []
    shared_base_tokens = (224 // 16) ** 2
    for patch in (14, 16):
        base_tokens = (224 // patch) ** 2
        for image_size in (224, 448, 896):
            grid = image_size // patch
            tokens = grid * grid
            result.append(
                {
                    "patch_size": patch,
                    "image_size": image_size,
                    "patch_grid": f"{grid}x{grid}",
                    "patch_tokens": tokens,
                    "attention_matrix_elements": tokens * tokens,
                    "relative_attention_vs_224": f"{(tokens / base_tokens) ** 2:.1f}",
                    "relative_attention_vs_vit16_224": f"{(tokens / shared_base_tokens) ** 2:.1f}",
                }
            )
    return result


def recommendation_rows(
    rows: list[dict[str, str]], gpu_memory_gb: float, multitask: bool, apache_only: bool
) -> list[dict[str, object]]:
    candidates = []
    for row in rows:
        params = float(row["parameters_m"])
        fp16_gb = params * 2 / 1000
        # A conservative three-times weight multiplier leaves room for activations,
        # the runtime, and a small task head. It is an engineering filter, not a
        # measured peak-memory prediction.
        estimated_runtime_gb = fp16_gb * 3
        if estimated_runtime_gb > gpu_memory_gb:
            continue
        if apache_only and row["license"] != "Apache-2.0":
            continue
        if multitask and row["model_type"] != "backbone":
            continue
        if not multitask and row["model_type"] == "detector":
            fit = "优先用于固定类别实时检测"
        elif row["family"] == "DINOv2":
            fit = "许可宽松且工具链成熟的通用特征基线"
        elif row["family"] == "DINOv3":
            fit = "适合关注密集特征和跨域能力的验证"
        else:
            fit = "适合作为轻量或历史对照"
        candidates.append(
            {
                "model": row["model"],
                "estimated_runtime_memory_gb": f"{estimated_runtime_gb:.2f}",
                "license": row["license"],
                "suggested_fit": fit,
            }
        )
    candidates.sort(key=lambda item: float(item["estimated_runtime_memory_gb"]))
    return candidates


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_bar_chart(resource_rows: list[dict[str, object]], path: Path) -> None:
    chosen = [r for r in resource_rows if r["model_type"] == "backbone" and float(r["parameters_m"]) <= 1100]
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "DINO代表模型理论权重内存", fill="#111111", font=font(42, True))
    draw.text((70, 95), "仅按参数量换算，不含激活、任务头与运行时开销", fill="#555555", font=font(24))
    left, top, right, bottom = 290, 160, 1430, 820
    draw.line((left, top, left, bottom), fill="#777777", width=2)
    max_value = max(float(r["fp16_weight_mb"]) for r in chosen)
    bar_h = (bottom - top) / len(chosen)
    colors = {"DINO v1": "#6C8EBF", "DINOv2": "#2F5597", "DINOv3": "#1F7A8C"}
    for index, row in enumerate(chosen):
        y0 = top + index * bar_h + 10
        y1 = y0 + bar_h - 18
        value = float(row["fp16_weight_mb"])
        family = str(row["model"]).split(" ")[0]
        if str(row["model"]).startswith("DINO v1"):
            family = "DINO v1"
        color = colors.get(family, "#2F5597")
        x1 = left + (right - left - 150) * value / max_value
        draw.rounded_rectangle((left, y0, x1, y1), radius=8, fill=color)
        draw.text((20, y0 + 4), str(row["model"]), fill="#222222", font=font(21))
        draw.text((x1 + 12, y0 + 4), f"{value:.0f} MB", fill="#222222", font=font(20, True))
    draw.text((70, 850), "数据来源：公开参数量；计算公式：参数量 × 每参数字节数", fill="#555555", font=font(20))
    image.save(path)


def draw_scaling_chart(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "输入分辨率与注意力矩阵规模", fill="#111111", font=font(42, True))
    draw.text((70, 95), "相对ViT-S/16的224输入；标准自注意力主要矩阵项按token数平方增长", fill="#555555", font=font(24))
    left, top, right, bottom = 140, 180, 1380, 770
    draw.line((left, bottom, right, bottom), fill="#777777", width=2)
    draw.line((left, top, left, bottom), fill="#777777", width=2)
    x_values = [224, 448, 896]
    for tick in (1, 4, 16, 64, 256, 512):
        y = bottom - math.log2(tick) / 9 * (bottom - top)
        draw.line((left, y, right, y), fill="#E0E0E0", width=1)
        draw.text((70, y - 12), f"{tick}x", fill="#444444", font=font(19))
    for patch, color in ((14, "#2F5597"), (16, "#1F7A8C")):
        points = []
        for size in x_values:
            item = next(r for r in rows if r["patch_size"] == patch and r["image_size"] == size)
            value = float(item["relative_attention_vs_vit16_224"])
            x = left + (x_values.index(size) + 0.5) * (right - left) / len(x_values)
            y = bottom - math.log2(value) / 9 * (bottom - top)
            points.append((x, y))
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)
            draw.text((x - 24, y - 42), f"{value:.0f}x", fill=color, font=font(20, True))
        draw.line(points, fill=color, width=5)
    for index, size in enumerate(x_values):
        x = left + (index + 0.5) * (right - left) / len(x_values)
        draw.text((x - 35, bottom + 18), str(size), fill="#222222", font=font(22))
    draw.rectangle((1020, 190, 1060, 215), fill="#2F5597")
    draw.text((1075, 184), "patch 14", fill="#222222", font=font(21))
    draw.rectangle((1020, 230, 1060, 255), fill="#1F7A8C")
    draw.text((1075, 224), "patch 16", fill="#222222", font=font(21))
    draw.text((620, 830), "输入边长", fill="#444444", font=font(22))
    image.save(path)


def run(gpu_memory_gb: float, multitask: bool, apache_only: bool) -> None:
    rows = load_rows()
    resources = resource_estimates(rows)
    scaling = scaling_estimates()
    recommendations = recommendation_rows(rows, gpu_memory_gb, multitask, apache_only)
    save_csv(
        RESULTS / "resource_estimates.csv",
        ["model", "model_type", "parameters_m", "fp32_weight_mb", "fp16_weight_mb", "int8_weight_mb", "license"],
        resources,
    )
    save_csv(
        RESULTS / "resolution_scaling.csv",
        ["patch_size", "image_size", "patch_grid", "patch_tokens", "attention_matrix_elements", "relative_attention_vs_224", "relative_attention_vs_vit16_224"],
        scaling,
    )
    save_csv(
        RESULTS / "model_selection.csv",
        ["model", "estimated_runtime_memory_gb", "license", "suggested_fit"],
        recommendations,
    )
    draw_bar_chart(resources, RESULTS / "resource_estimates.png")
    draw_scaling_chart(scaling, RESULTS / "resolution_scaling.png")
    print(f"Generated results in {RESULTS}")
    print(f"Candidates under current constraints: {len(recommendations)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DINO resource and deployment analysis demo")
    parser.add_argument("--gpu-memory-gb", type=float, default=8.0)
    parser.add_argument("--multitask", action="store_true", help="Keep general-purpose backbones only")
    parser.add_argument("--apache-only", action="store_true", help="Keep Apache-2.0 candidates only")
    args = parser.parse_args()
    run(args.gpu_memory_gb, args.multitask, args.apache_only)


if __name__ == "__main__":
    main()
