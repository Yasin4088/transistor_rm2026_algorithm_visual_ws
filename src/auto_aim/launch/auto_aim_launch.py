# auto_aim_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# 用法：
#   比赛/性能测试（默认，不开可视化，性能最好）：
#     ros2 launch auto_aim auto_aim_launch.py
#   需要看画面调试：
#     ros2 launch auto_aim auto_aim_launch.py visualizer:=true
#
# 核绑定参考（8 物理核 / 16 逻辑核 SMT 机器）：
#   armor_detect_node 绑物理核 0-7、visualizer 绑 8-15，可缓解互相抢核：
#     taskset -c 0-7  ros2 run auto_aim armor_detect_node
#     taskset -c 8-15 ros2 run auto_aim_visualizer auto_aim_visualizer_node
#   注意：SMT 下 8-15 与 0-7 共享物理核，只能缓解不能根治；
#   性能敏感场景（比赛）建议直接不开 visualizer（本 launch 默认即如此）。

def generate_launch_description():
    visualizer_arg = DeclareLaunchArgument(
        'visualizer',
        default_value='false',
        description='是否启动可视化节点（默认 false 比赛态；调试时 visualizer:=true）')

    visualizer_node = Node(
        package='auto_aim_visualizer',
        executable='auto_aim_visualizer_node',
        name='auto_aim_visualizer_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('visualizer')),
    )

    # openvino_core = ov.Core()
    # if "GPU" in openvino_core.available_devices:
    #     shm_yolo_pose_node = ExecuteProcess(
    #         cmd=['taskset', '-c', "0,1", 'ros2', 'run', 'shm_python_processor_pkg', 'shm_yolo_pose_node'],
    #         output='screen'
    #     )
    # else:
    #     shm_yolo_pose_node = Node(
    #         package='shm_python_processor_pkg',
    #         executable='shm_yolo_pose_node',
    #         name='shm_yolo_pose_node',
    #         #arguments = ['--ros-args', '--log-level', 'DEBUG']
    #     )

    return LaunchDescription([
        visualizer_arg,
        #Node(
        #    package='auto_aim',
        #    executable='com_node',  # 改为 com_node
        #    name='com_node',        # 改为 com_node
        #    parameters=[{
        #        'serial_port': '/dev/ttyACM0',
        #        'baudrate': 115200
        #    }]
        #),
        Node(
            package='auto_aim',
            executable='armor_detect_node',
            name='armor_detect_node',
            output='screen',  # <--- 加上这一行
            #arguments = ['--ros-args', '--log-level', 'DEBUG']
        ),
        visualizer_node,
        Node(
            package='shm_python_processor_pkg',
            executable='shm_classifier_node',
            name='shm_classifier_node',
            #arguments = ['--ros-args', '--log-level', 'DEBUG']
        ),
        # shm_yolo_pose_node
    ])
