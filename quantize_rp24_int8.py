#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RP24_YOLO 0526.onnx -> OpenVINO INT8 后训练量化（PTQ）

用法（在装有 openvino / nncf / opencv-python 的机器上）：
    pip install openvino nncf opencv-python
    python3 quantize_rp24_int8.py --video InputVideo/infantry_blue.mp4

产出：
    src/shared_files/models/RP24_YOLO/0526_int8.xml
    src/shared_files/models/RP24_YOLO/0526_int8.bin
    src/shared_files/models/RP24_YOLO/0526_int8.onnx   (占位副本，让 C++ 侧路径解析通过)

然后把 config.yaml 里 RP24_YOLO_model_relative_path 改成：
    "src/shared_files/models/RP24_YOLO/0526_int8.onnx"
（C++ 端会检测同名 .xml 已存在，直接加载 INT8 IR，无需重编译）

说明：0526.onnx 的输入约定为 RGB float [0,1] NCHW（C++ 端 PrePostProcessor
负责 u8 BGR -> f32 RGB /255），校准数据必须与之匹配。
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def require(pkg):
    try:
        return __import__(pkg)
    except ImportError:
        sys.exit(f"[ERROR] 缺少依赖 {pkg}，请先安装：pip install openvino nncf opencv-python")


def build_calibration_gen(video_path, count):
    """从视频抽帧构造 NNCF 校准数据：RGB f32 [0,1] NCHW [1,3,640,640]"""
    cv2 = require("cv2")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] 打不开视频：{video_path}")

    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 视频循环取帧
            continue
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # HWC -> CHW
        frames.append(img[np.newaxis, ...])         # [1,3,640,640]
    cap.release()
    print(f"[INFO] 校准帧数：{len(frames)}")

    def gen():
        for f in frames:
            yield {"images": f}

    return gen()


def main():
    ap = argparse.ArgumentParser(description="RP24_YOLO ONNX -> OpenVINO INT8")
    ap.add_argument("--onnx", default="src/shared_files/models/RP24_YOLO/0526.onnx",
                    help="原始 ONNX 模型路径")
    ap.add_argument("--video", default="InputVideo/infantry_blue.mp4",
                    help="用于校准的视频路径")
    ap.add_argument("--frames", type=int, default=300,
                    help="校准抽样帧数")
    ap.add_argument("--output", default=None,
                    help="输出前缀（默认 <onnx 同目录>_int8）")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        sys.exit(f"[ERROR] 找不到 ONNX：{onnx_path}")

    out_prefix = Path(args.output) if args.output else onnx_path.with_name(
        onnx_path.stem + "_int8")

    ov = require("openvino")
    nncf = require("nncf")

    print(f"[INFO] 加载 ONNX：{onnx_path}")
    # 注意：用 read_model 而不是 ov.convert_model——与 C++ 端 FP32 路径一致，
    # convert_model 可能改变输出布局（如 [1,25200,22] -> [1,22,25200]），
    # 会导致 C++ 解码读到错位的关键点（表现为“Diagonals are parallel”）。
    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))

    print("[INFO] 构建校准数据集 ...")
    calib_gen = build_calibration_gen(args.video, args.frames)

    print("[INFO] NNCF INT8 量化中（小模型约几十秒）...")
    quantized_model = nncf.quantize(
        ov_model,
        nncf.Dataset(calib_gen),
        preset=nncf.quantization.QuantizationPreset.MIXED,
    )

    # 自检：输出形状 + 数值是否有限 + 关键点列量级（提前发现量化把模型搞坏）
    try:
        compiled = ov.compile_model(quantized_model, "CPU")
        probe = next(build_calibration_gen(args.video, 1))
        raw = compiled(probe)
        out_tensor = list(raw.values())[0]
        arr = np.asarray(out_tensor)
        print(f"[CHECK] 量化模型输出形状：{list(arr.shape)}")
        print(f"[CHECK] 数值有限：{bool(np.isfinite(arr).all())}  "
              f"min={float(np.min(arr)):.4f} max={float(np.max(arr)):.4f}")
        if arr.ndim == 3 and arr.shape[1] < arr.shape[2]:
            print("[WARN] 输出疑似转置布局 [N, 22, 25200]，C++ 解码会出错！")
    except Exception as e:
        print(f"[WARN] 自检失败（不影响保存）：{e}")

    out_xml = out_prefix.with_suffix(".xml")
    ov.save_model(quantized_model, str(out_xml))
    print(f"[OK] INT8 模型已保存：{out_xml}")
    print(f"     bin 文件：{out_prefix.with_suffix('.bin')}")

    # C++ 端路径解析要求 .onnx 存在（会 strip 后缀后查同名 .xml），复制一份占位
    dummy_onnx = out_prefix.with_suffix(".onnx")
    if not dummy_onnx.exists():
        import shutil
        shutil.copy2(onnx_path, dummy_onnx)
        print(f"[OK] 占位 onnx（供 C++ 路径解析）：{dummy_onnx}")

    print("\n下一步：把 config.yaml 的 RP24_YOLO_model_relative_path 改为：")
    print(f'    "{dummy_onnx}"')
    print("重启节点即可，无需重新编译 C++。")


if __name__ == "__main__":
    main()
