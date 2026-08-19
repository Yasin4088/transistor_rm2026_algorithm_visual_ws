#include "predictor/PredictorSwitcher.h"

namespace PredictorType {
    std::vector<std::string> PredictorTypeStrings = {
        "None",
        "RMM",
        "DirectModel",
        "AutoSwitch(should not be used)"
    };
}

PredictorSwitcher::PredictorSwitcher(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node)
    : config_file_ptr(config_file_ptr), node(node) {
    // 从配置文件读取预测器类型，缺失时默认 RMM（保持原行为）
    std::string predictor_type_str = "RMM";
    if ((*config_file_ptr)["predictor_type"]) {
        predictor_type_str = (*config_file_ptr)["predictor_type"].as<std::string>();
    }
    if (predictor_type_str == "None") {
        predictor_type_ = PredictorType::None;
    } else if (predictor_type_str == "DirectModel") {
        predictor_type_ = PredictorType::DirectModel;
    } else {
        predictor_type_ = PredictorType::RotationMotionModel;
    }
}

void PredictorSwitcher::clearHistory() {
}


PredictorType::PredictorType PredictorSwitcher::step() {
    return predictor_type_;
}
