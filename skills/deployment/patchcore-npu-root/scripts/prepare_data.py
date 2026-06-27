#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore 合成数据生成器
======================
生成 MVTec AD 格式的合成工业图像数据，用于推理验证。
目录结构: data/mvtec/{category}/train/good/ | test/{good,defect_type}/

用法:
    python prepare_data.py --output data/mvtec --num_train 50 --num_test 20
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CATEGORIES = {
    "bottle": {
        "base_color": (100, 140, 180),
        "shape": "circle",
        "defects": ["broken", "contamination"],
    },
    "cable": {
        "base_color": (60, 60, 60),
        "shape": "lines",
        "defects": ["bent_wire", "cable_swap"],
    },
    "capsule": {
        "base_color": (200, 180, 140),
        "shape": "pill",
        "defects": ["crack", "poke"],
    },
}


def generate_base_image(category, size=224, seed=None):
    """生成正常的基础图像。"""
    rng = np.random.RandomState(seed)
    cfg = CATEGORIES[category]
    base = np.full((size, size, 3), cfg["base_color"], dtype=np.uint8)
    noise = rng.randint(-15, 16, (size, size, 3)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    if cfg["shape"] == "circle":
        cx, cy, r = size // 2, size // 2, size // 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(70, 110, 150), width=3)
        draw.ellipse([cx - r + 10, cy - r + 10, cx + r - 10, cy + r - 10],
                      fill=(cfg["base_color"][0] + 20, cfg["base_color"][1] + 20, cfg["base_color"][2] + 20))
    elif cfg["shape"] == "lines":
        for i in range(5):
            y = size // 6 + i * size // 6
            draw.line([(20, y), (size - 20, y)], fill=(80, 80, 80), width=2)
    elif cfg["shape"] == "pill":
        cx, cy = size // 2, size // 2
        w, h = size // 3, size // 5
        draw.rounded_rectangle([cx - w, cy - h, cx + w, cy + h], radius=h,
                                fill=(220, 200, 160), outline=(180, 160, 120), width=2)
        draw.line([(cx, cy - h), (cx, cy + h)], fill=(180, 160, 120), width=1)

    return img


def add_defect(img, defect_type, seed=None):
    """在正常图像上添加缺陷。"""
    rng = np.random.RandomState(seed)
    draw = ImageDraw.Draw(img)
    size = img.size[0]

    if defect_type == "broken" or defect_type == "crack":
        x1, y1 = rng.randint(30, size // 3, 2)
        x2, y2 = rng.randint(size // 2, size - 30, 2)
        draw.line([(x1, y1), (x2, y2)], fill=(30, 30, 30), width=rng.randint(2, 5))
    elif defect_type == "contamination" or defect_type == "poke":
        cx, cy = rng.randint(40, size - 40, 2)
        r = rng.randint(8, 20)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(40, 40, 40))
    elif defect_type == "bent_wire":
        points = [(rng.randint(20, size - 20), rng.randint(20, size - 20)) for _ in range(5)]
        draw.line(points, fill=(200, 50, 50), width=3)
    elif defect_type == "cable_swap":
        cx, cy = rng.randint(40, size - 40, 2)
        draw.rectangle([cx - 15, cy - 15, cx + 15, cy + 15], fill=(200, 100, 50))

    return img


def generate_mask(size, defect_type, seed=None):
    """生成缺陷 mask（白色=缺陷）。"""
    rng = np.random.RandomState(seed)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    if defect_type in ("broken", "crack"):
        x1, y1 = rng.randint(30, size // 3, 2)
        x2, y2 = rng.randint(size // 2, size - 30, 2)
        draw.line([(x1, y1), (x2, y2)], fill=255, width=rng.randint(4, 10))
    elif defect_type in ("contamination", "poke"):
        cx, cy = rng.randint(40, size - 40, 2)
        r = rng.randint(10, 25)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif defect_type == "bent_wire":
        points = [(rng.randint(20, size - 20), rng.randint(20, size - 20)) for _ in range(5)]
        draw.line(points, fill=255, width=6)
    elif defect_type == "cable_swap":
        cx, cy = rng.randint(40, size - 40, 2)
        draw.rectangle([cx - 18, cy - 18, cx + 18, cy + 18], fill=255)

    return mask


def main():
    parser = argparse.ArgumentParser(description="PatchCore 合成数据生成")
    parser.add_argument("--output", type=str, default="data/mvtec", help="输出目录")
    parser.add_argument("--num_train", type=int, default=50, help="每类训练图数量")
    parser.add_argument("--num_test", type=int, default=20, help="每类每种缺陷测试图数量")
    parser.add_argument("--size", type=int, default=224, help="图像尺寸")
    args = parser.parse_args()

    total_images = 0
    for cat_name, cfg in CATEGORIES.items():
        print(f"[INFO] 生成 {cat_name} ...")

        # 训练集（正常图）
        train_dir = os.path.join(args.output, cat_name, "train", "good")
        os.makedirs(train_dir, exist_ok=True)
        for i in range(args.num_train):
            img = generate_base_image(cat_name, args.size, seed=i * 1000 + hash(cat_name) % 10000)
            img.save(os.path.join(train_dir, f"{i:04d}.png"))
            total_images += 1

        # 测试集（正常图）
        test_good_dir = os.path.join(args.output, cat_name, "test", "good")
        os.makedirs(test_good_dir, exist_ok=True)
        for i in range(args.num_test):
            img = generate_base_image(cat_name, args.size, seed=i * 2000 + hash(cat_name) % 10000)
            img.save(os.path.join(test_good_dir, f"{i:04d}.png"))
            total_images += 1

        # 测试集（缺陷图）+ ground truth mask
        for defect in cfg["defects"]:
            test_defect_dir = os.path.join(args.output, cat_name, "test", defect)
            gt_dir = os.path.join(args.output, cat_name, "ground_truth", defect)
            os.makedirs(test_defect_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            for i in range(args.num_test):
                seed = i * 3000 + hash(f"{cat_name}{defect}") % 10000
                img = generate_base_image(cat_name, args.size, seed=seed)
                img = add_defect(img, defect, seed=seed)
                img.save(os.path.join(test_defect_dir, f"{i:04d}.png"))
                mask = generate_mask(args.size, defect, seed=seed)
                mask.save(os.path.join(gt_dir, f"{i:04d}_mask.png"))
                total_images += 1

    print(f"\n[INFO] 共生成 {total_images} 张图像")
    print(f"[INFO] 数据目录: {os.path.abspath(args.output)}")
    # 打印目录结构
    for cat_name in CATEGORIES:
        base = os.path.join(args.output, cat_name)
        for root, dirs, files in os.walk(base):
            level = root.replace(base, "").count(os.sep)
            indent = "  " * level
            print(f"  {indent}{os.path.basename(root)}/ ({len(files)} files)")


if __name__ == "__main__":
    main()
