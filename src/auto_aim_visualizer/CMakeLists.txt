cmake_minimum_required(VERSION 3.5)
project(auto_aim_visualizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 默认 RelWithDebInfo（带调试符号 + 优化）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(auto_aim REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(auto_aim_visualizer_node
    src/AutoAimVisualizer_Node.cpp
)

target_link_libraries(auto_aim_visualizer_node
    ${OpenCV_LIBS}
    rclcpp::rclcpp
    cv_bridge::cv_bridge
    yaml-cpp
)

ament_target_dependencies(auto_aim_visualizer_node
    rclcpp
    sensor_msgs
    std_msgs
    cv_bridge
    auto_aim
    yaml-cpp
)

install(TARGETS
    auto_aim_visualizer_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
