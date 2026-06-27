#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 Paddle2ONNX 导出的 ONNX 模型：将 Constant 节点提取为 Initializer。

Paddle2ONNX 常将权重以 Constant 节点形式存储，导致 onnx2torch 解析失败。
此脚本遍历所有 Constant 节点，将其中的 tensor/floats/ints 提取为 graph.initializer。

用法：
    python fix_constant_nodes.py --input model.onnx --output model_fixed.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def fix_constant_nodes(input_path: str, output_path: str):
    model = onnx.load(input_path)
    graph = model.graph
    new_nodes = []

    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    arr = numpy_helper.to_array(attr.t)
                    tensor = numpy_helper.from_array(arr, name=node.output[0])
                    graph.initializer.append(tensor)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    arr = np.array(attr.floats, dtype=np.float32)
                    tensor = numpy_helper.from_array(arr, name=node.output[0])
                    graph.initializer.append(tensor)
                elif attr.type == onnx.AttributeProto.INTS:
                    arr = np.array(attr.ints, dtype=np.int64)
                    tensor = numpy_helper.from_array(arr, name=node.output[0])
                    graph.initializer.append(tensor)
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.save(model, output_path)
    print(f"Fixed: {input_path} -> {output_path} (initializers: {len(graph.initializer)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 ONNX 文件路径")
    parser.add_argument("--output", required=True, help="输出修复后的 ONNX 文件路径")
    args = parser.parse_args()
    fix_constant_nodes(args.input, args.output)


if __name__ == "__main__":
    main()
