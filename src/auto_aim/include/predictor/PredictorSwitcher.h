#ifndef PREDICTOR_SWITCHER_H
#define PREDICTOR_SWITCHER_H
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <deque>

#include "utils/DataProcessFuncs.h"
#include "3d_processing/RestFrame.h"
// #include "predictor/RotationJudge.h"

namespace PredictorType {
    enum PredictorType {
        None = 0,   // 直接瞄准装甲板
        RotationMotionModel,
        DirectModel, // 直接运动模型：自由 3D 点目标（卡尔曼平滑），适合手持板/无旋转假设场景
        AutoSwitch
    };

    extern std::vector<std::string> PredictorTypeStrings;
}

class PredictorSwitcher {
public:
    PredictorSwitcher(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);
    PredictorType::PredictorType step();
    void clearHistory();
    
private:
    std::shared_ptr<YAML::Node> config_file_ptr; 
    rclcpp::Node* node;
    PredictorType::PredictorType predictor_type_ = PredictorType::RotationMotionModel;
};

#endif
