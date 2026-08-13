# 北航Transistor战队RM2026算法视觉组代码

## 0、安装

* 需提前安装好ROS2 Humble和MVS（海康机器人工业相机驱动），可参考下列链接安装：

```
https://fishros.com/d2lros2/#/humble/chapt1/get_started/3.%E5%8A%A8%E6%89%8B%E5%AE%89%E8%A3%85ROS2
https://www.hikrobotics.com/cn/machinevision/service/download/
```

* 最外层目录（即该README.md所在目录）需保持命名为"transistor_rm2026_algorithm_visual_ws"，配置文件读取依赖此路径名。
* 下列命令的执行路径均为上述最外层文件夹。
* 运行下命令自动安装必要依赖（该命令会在~目录下建立"makeInstall"文件夹，所有需要编译安装的依赖保存于此）：

```
sudo bash ./pkgInstall/pkgInstall.bash
```

* 如有Intel的GPU（包括核显），可执行下命令安装驱动：

```
sudo bash ./pkgInstall/intelGpuDriverInstall.bash
```

* 安装失败可尝试按照`./pkgInstall/pkgInstall.txt`中的命令和操作手动安装。

## 1、配置

* 主要配置文件为下列两个文件：

```
src/shared_files/config.yaml
src/auto_aim/include/macro/AutoAimMacro.h
```

* 其中config.yaml更改后无需重新编译，AutoAimMacro.h更改后需重新编译才能生效。
* 无摄像头时可在`AutoAimMacro.h`中定义`USE_VIDEO`或`USE_IMAGES`后重新编译，切换至使用视频或图片文件夹输入，使用的视频或图片文件夹路径在`config.yaml`中。

## 2、编译

* 运行下命令编译：

```
colcon build
```

## 3、运行

* 运行下命令导入环境（每次打开终端后该命令仅需运行一次，多次运行无需重复执行）：

```
source ./install/setup.bash
```

* 运行下命令运行：

```
ros2 launch auto_aim auto_aim_launch.py
```

* 运行后可能没有画面，需定义`AutoAimMacro.h`中的`SHOW_WINDOWS`后重新编译。

## 4、自启动

* 运行下命令安装自启动服务：

```
sudo ./auto_launch/serviceInstall.py
```

* 该命令会产生自启动服务文件`/etc/systemd/system/auto_aim_auto_launch.service`。
* 使用自启动时需取消定义`AutoAimMacro.h`中的`SHOW_WINDOWS`后重新编译，否则无法自启动。
* 自启动使用`auto_launch`中的看门狗启动主程序，而直接运行不使用看门狗。可手动运行`./auto_launch/watchdog.py`来使用看门狗启动主程序。
* 使用下列命令管理自瞄程序自启动：

```
sudo systemctl enable auto_aim_auto_launch.service # 启用自启动
sudo systemctl disable auto_aim_auto_launch.service # 取消自启动
sudo systemctl start auto_aim_auto_launch.service # 单次运行自启动脚本
sudo systemctl stop auto_aim_auto_launch.service # 关闭当前运行中的自启动程序
sudo systemctl status auto_aim_auto_launch.service # 查看自启动服务状态
```
## 修稿
对比完成。先说整体关系：old 目录是 `main` 分支（团队旧状态），当前是 `feature/thread-pool`——两者差异包含**团队前期工作 + 本次会话**。本次会话实际只改了 **11 个文件（+385/-69）**，全部是性能链路。

## 本次会话改动（按类别）

**1. 事件驱动唤醒**（[AutoAimPipeline.cpp](C:/Users/Lenovo/ourscode/transistor_rm2026_algorithm_visual_ws/src/auto_aim/src/pipeline/AutoAimPipeline.cpp) / [RP24_YOLO_Wrapper.h](C:/Users/Lenovo/ourscode/transistor_rm2026_algorithm_visual_ws/src/auto_aim/include/RP24_YOLO/RP24_YOLO_Wrapper.h)）
- Stage1 拉 YOLO 结果从 5ms 轮询改为"结果就绪回调置位 + 条件变量唤醒"（`result_wakeup_` + `setResultNotify`），砍掉最多 5ms 帧延迟。

**2. 推理配置化 + 线程调优**（[OpenvinoInfer.cpp](C:/Users/Lenovo/ourscode/transistor_rm2026_algorithm_visual_ws/src/auto_aim/src/RP24_YOLO/OpenvinoInfer.cpp) / config.yaml）
- `inference_num_threads / num_streams` 变成配置项，经隔离基准确认 **8 线程（7.5ms）> 4 线程（12.5ms）**，最终 `threads:8 / streams:1`。

**3. 计时可视化**（[PerformanceMonitor.cpp](C:/Users/Lenovo/ourscode/transistor_rm2026_algorithm_visual_ws/src/auto_aim/src/utils/PerformanceMonitor.cpp)）
- 拆分 `yolo_infer`（纯推理）与 `yolo_infer_wait`（排队）——之前等锁时间混在推理里，误导了两轮判断；
- 报告加时间戳 + 帧号区间 + avg/min/max，yolo 三阶段补进打印列表；
- 新增 `stage1_classify_track` 计时；YOLO 路径总延迟改名 `stage1_yolo_latency` 与传统视觉区分；
- 启动日志打印实际加载的模型路径。

**4. 多帧并行推理（实现后被数据否决）**（[OpenvinoInfer.h](C:/Users/Lenovo/ourscode/transistor_rm2026_algorithm_visual_ws/src/auto_aim/include/RP24_YOLO/OpenvinoInfer.h)）
- 做了 N 个 `InferRequest` 请求池、删掉 `infer_mutex_`、解码全部局部化（线程安全）；
- CPU 实测双流仅 +12% 且单帧推理翻倍 → 配置回 `streams:1` 退化为单请求（代码保留）。

**5. INT8 量化（完整走通后被数据否决）**
- 写了 [quantize_rp24_int8.py](C:/Users/Lenovo/ourscode/transistor_rm2026_algorithm_visual_ws/quantize_rp24_int8.py)（NNCF PTQ、read_model 替代 convert_model 防输出转置、输出自检）；
- postprocess 加"退化装甲板跳过而非丢帧"容错；
- AMD CPU 实测 INT8 仅 +15%（6.4 vs 7.5ms）→ 弃用回 FP32。

**6. 环境减负（尝试后按要求撤回）**
- launch 加 `visualizer:=true/false` 开关默认关闭 + taskset 注释参考——你撤回后 launch 恢复原样，但结论留下：**visualizer 是环境最大 CPU 小偷**。

**7. 杂项**：`.gitignore` 补 `__pycache__/*.pyc`，清理误提交缓存。

## 性能成果

| 指标 | 会话开始（4线程+轮询+串行） | 最终（8线程+事件唤醒+无visualizer） |
|---|---|---|
| yolo_infer | 16.7ms | **8.9ms** |
| yolo_infer_wait | 46.8ms | **0.02ms** |
| 端到端 Total | 116ms | **13.5ms** |

## 被验证否决的路线（数据驱动）

- 8 线程单流"更慢"（17ms）→ 脏测量，拆分计时后反转；
- 2 流跨帧并行推理（+12%）→ CPU 算力瓶颈，并发只是分摊；
- INT8（+15%）→ AMD 无 VNNI 红利；
- 输入 512 → 模型头部硬编码 `[1,3,22,80,80]`，只认 640。

最终定位：**AMD CPU 上"单帧内多线程 + 流水线任务重叠 + 环境减负"是有效杠杆，"跨帧并行推理"和"量化"都是死路**。当前状态 13.5ms 端到端、wait 0.02ms，余量充足，可以直接上场。
========== Auto Aim Performance ==========
Report time: 09:54:27 (uptime 17.671 s)
Frames: 90 (id 1260 .. 1349)
stage1_yolo_latency: avg 9.588 ms [min 8.420, max 14.953]
yolo_preprocess: avg 0.567 ms [min 0.410, max 1.303]
yolo_infer: avg 8.886 ms [min 7.834, max 14.217]
yolo_infer_wait: avg 0.019 ms [min 0.001, max 1.611]
yolo_postprocess: avg 0.084 ms [min 0.059, max 0.162]
stage2_3d_solve_transform: avg 0.301 ms [min 0.183, max 0.490]
stage3_predict_command: avg 2.961 ms [min 2.369, max 5.641]
stage4_visualize_log: avg 0.015 ms [min 0.008, max 0.029]
stage1_classify_track: avg 0.013 ms [min 0.008, max 0.034]
------------------------------------------
Total(from data init): 13.529 ms
FPS: 73.914
==========================================

