cmake_minimum_required(VERSION 3.5)
project(auto_aim)

# 防御：清除环境变量 CXXFLAGS / CMake 缓存可能带入的 -flto。
# 原因：LTO 目标文件配合 Ubuntu 默认 --as-needed 链接时，链接器会把
# rclcpp/OpenCV/yaml-cpp/g2o 等共享库从最终链接中丢弃，表现为大量
# undefined reference（即使 CMakeLists 里已删除 -flto，环境变量仍会带入）。
string(REPLACE "-flto=auto" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-flto" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 默认 RelWithDebInfo（带调试符号 + 优化，便于 uProf/性能分析定位函数）；
# 可通过命令行 -DCMAKE_BUILD_TYPE=xxx 覆盖
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(TBB REQUIRED)
find_package(angles REQUIRED) # <<--- 确保这一行存在

# ba solver
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)

# OpenVINO：优先从 Python pip 安装位置自动定位 CMake 配置目录。
# pip 版 openvino 不会进入 CMAKE_PREFIX_PATH，全新构建目录（无 CMakeCache）
# 下 find_package(OpenVINO) 会失败；这里自动探测 openvino 包内的 cmake/ 目录，
# 用户也可用 -DOpenVINO_DIR=xxx 显式覆盖。
if(NOT DEFINED OpenVINO_DIR)
    # 1) 从 python 探测 pip 版 openvino 的 cmake 目录
    foreach(_py IN ITEMS python3 python)
        execute_process(
            COMMAND ${_py} -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'cmake'))"
            OUTPUT_VARIABLE _openvino_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_openvino_cmake_dir AND EXISTS "${_openvino_cmake_dir}/OpenVINOConfig.cmake")
            set(OpenVINO_DIR "${_openvino_cmake_dir}" CACHE PATH "OpenVINO CMake config directory (pip install)")
            break()
        endif()
    endforeach()
    # 2) 兜底：直接扫描 ~/.local/lib/python*/site-packages/openvino/cmake，
    #    不依赖 python 是否能 import openvino
    if(NOT DEFINED OpenVINO_DIR)
        file(GLOB _openvino_pip_cmake_dirs
            "$ENV{HOME}/.local/lib/python*/site-packages/openvino/cmake")
        foreach(_d IN LISTS _openvino_pip_cmake_dirs)
            if(EXISTS "${_d}/OpenVINOConfig.cmake")
                set(OpenVINO_DIR "${_d}" CACHE PATH "OpenVINO CMake config directory (pip install)")
                break()
            endif()
        endforeach()
    endif()
endif()
if(DEFINED OpenVINO_DIR)
    message(STATUS "OpenVINO_DIR: ${OpenVINO_DIR}")
endif()
find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Point2f.msg"
  "msg/DebugLight.msg"
  "msg/DebugArmor.msg"
  "msg/DebugArmorResult.msg"
  "msg/VisualizerDebugData.msg"
  DEPENDENCIES std_msgs
)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")

# 查找 MvCameraControl 库
find_library(MVCAMERA_LIB MvCameraControl PATHS /opt/MVS/lib/64)

# 设置包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    /opt/MVS/include
    /opt/ros/humble/include
    /opt/ros/humble/include/cv_bridge
    ${sensor_msgs_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}
)

# 添加装甲板检测相关的库
add_library(armor_detector SHARED
    src/2d_armor_detector/Armor.cpp
    src/2d_armor_detector/LightBarDetector.cpp
    src/2d_armor_detector/ArmorDetector.cpp
    src/2d_armor_detector/ArmorClassifier.cpp
    src/2d_armor_detector/ArmorTracker.cpp
    src/2d_armor_detector/UnwarpUtils.cpp
    # src/2d_armor_detector/YOLOPoseArmorDetector.cpp
    src/3d_processing/ArmorSolver.cpp
    src/3d_processing/BallisticSolver.cpp
    src/3d_processing/RestFrame.cpp
    src/camera/cameraConnect.cpp  # 添加这一行
    src/other_input/VideoInput.cpp
    src/other_input/ImagesInput.cpp
    src/predictor/PositionPredictor2D.cpp
    # src/predictor/PredictionTrans.cpp
    # src/predictor/PositionPredictor3D.cpp
    src/predictor/PeriodicDataPredictor.cpp
    src/predictor/RotationMotionModel.cpp
    src/predictor/DirectMotionModel.cpp
    src/predictor/AllPredictor.cpp
    src/predictor/PredictorSwitcher.cpp
    src/predictor/PredictorMain.cpp
    # src/predictor/RotationJudge.cpp
    src/visualizer/DataVisualizer.cpp
    src/communication/SharedMemoryClassifier.cpp
    # src/communication/SharedMemoryYOLOPose.cpp
    src/communication/Com.cpp
    src/communication/TorqueBridge.cpp
    src/communication/WatchdogClient.cpp
    src/communication/HeadIMU.cpp
    src/utils/FrameRateCounter.cpp
    src/utils/PerformanceMonitor.cpp
    src/utils/SimpleDataFilter.cpp
    src/utils/DataProcessFuncs.cpp
    src/utils/PeriodFunctions.cpp
    # EKF
    # src/EKF/Tracker.cpp
    # ba solver
    src/ba_solver/ba_solver.cpp
    src/ba_solver/graph_optimizer.cpp
    src/ba_solver/utils.cpp
    src/logger/TwoVideoLogger.cpp
    src/logger/MkvWriter.cpp
    src/RP24_YOLO/OpenvinoInfer.cpp
    src/RP24_YOLO/RP24_YOLO_Wrapper.cpp
    src/pipeline/AutoAimPipeline.cpp
)

# ============================================================================
# TorqueController（力矩控制：MCU+IMU 串口 / 融合 / MPC / 100Hz 下行）
# ============================================================================
set(TORQUE_SOURCES
    src/torque_controller/CRC.cpp
    src/torque_controller/FusionFilter.cpp
    src/torque_controller/McuDataPreprocessor.cpp
    src/torque_controller/mpc_controller.cpp
    src/torque_controller/yaw_mpc_controller.cpp
    src/torque_controller/mcu_mpc_controller.cpp
    src/torque_controller/RobotController.cpp
)
add_library(torque_controller STATIC ${TORQUE_SOURCES})
set_target_properties(torque_controller PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(torque_controller PUBLIC
    ${PROJECT_SOURCE_DIR}/include/torque_controller
    ${CERES_INCLUDE_DIRS}
)
# 用 Ceres::ceres 目标而不是 ${CERES_LIBRARIES}：目标会传递 glog/gflags/SuiteSparse
# 等传递依赖，否则链接 libarmor_detector.so 时会出现
# google::base::CheckOpMessageBuilder::NewString 等 undefined reference
target_link_libraries(torque_controller PUBLIC udev Ceres::ceres Eigen3::Eigen)
#add_library(crc_lib SHARED
#    src/nodes/CRC.cpp
#)
# 添加可执行文件
add_executable(armor_detect_node src/nodes/ArmorDetect_Node.cpp)
#add_executable(com_node src/nodes/Com_Node_New.cpp)

# 为装甲板检测器添加依赖
target_link_libraries(armor_detector
    ${OpenCV_LIBS}
    Eigen3::Eigen
    rclcpp::rclcpp   # 添加这一行
    yaml-cpp
#    "${cpp_typesupport_target}"
#    crc_lib
    pthread

    # ba solver
    cv_bridge::cv_bridge
    g2o_core
    g2o_stuff
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_types_slam3d
    g2o_types_sba
    g2o_solver_dense
    cxsparse
    Sophus::Sophus
    fmt::fmt
    openvino::runtime
    udev
    torque_controller
)

# 为装甲板检测节点添加依赖
target_link_libraries(armor_detect_node
    armor_detector
    ${OpenCV_LIBS}
    ${MVCAMERA_LIB}
    Eigen3::Eigen
    pthread
    rclcpp::rclcpp
    ${sensor_msgs_LIBRARIES}
    ${cv_bridge_LIBRARIES}
    "${cpp_typesupport_target}"
    yaml-cpp
    TBB::tbb
    openvino::runtime
)

# 强制链接器保留所有显式链接的库，防止 --as-needed 把它们丢弃
target_link_options(armor_detector PRIVATE -Wl,--no-as-needed)
target_link_options(armor_detect_node PRIVATE -Wl,--no-as-needed)

# 为通信节点添加依赖
#target_link_libraries(com_node
#    crc_lib
#    pthread
#    rclcpp::rclcpp
#    "${cpp_typesupport_target}"
#)

ament_target_dependencies(armor_detector yaml-cpp angles)

ament_target_dependencies(armor_detect_node yaml-cpp)

# 编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 不要启用 -flto / -flto=auto：
# 默认构建类型改为 RelWithDebInfo 后，编译侧 -flto（仅 Release 生效）与链接侧
# -flto=auto 不一致，配合 Ubuntu 默认 --as-needed 会把共享库（rclcpp/OpenCV/
# yaml-cpp/g2o 等）从链接中丢弃，导致大量 undefined reference
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")

# 添加库的路径
link_directories(/opt/MVS/lib/64)

# 安装
install(TARGETS
    armor_detect_node
#    com_node
    armor_detector  # 添加这行
#    crc_lib        # 添加这行
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
