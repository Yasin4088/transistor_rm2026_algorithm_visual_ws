// ArmorDetector.h
#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "LightBar.h"
#include "2d_armor_detector/Armor.h"
#include <vector>
#include <yaml-cpp/yaml.h>
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>
#include <algorithm>
#include <execution>
#include <thread>
#include <rclcpp/rclcpp.hpp>

class ArmorDetector {
public:
    ArmorDetector(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node)
    : node(node), config_file_ptr(config_file_ptr) {
        const YAML::Node& armor_cfg = (*config_file_ptr)["armor_detector"];
        max_angle_diff = armor_cfg["max_angle_diff"].as<float>();
        max_height_diff_ratio = armor_cfg["max_height_diff_ratio"].as<float>();
        min_light_distance = armor_cfg["min_light_distance"].as<float>();
        max_light_distance = armor_cfg["max_light_distance"].as<float>();
        min_armor_confidence = armor_cfg["min_armor_confidence"].as<float>();
        max_expected_small_distance_mismatch_ratio = armor_cfg["max_expected_small_distance_mismatch_ratio"].as<float>();
        max_expected_large_distance_mismatch_ratio = armor_cfg["max_expected_large_distance_mismatch_ratio"].as<float>();
        max_length_direction_mismatch_ratio = armor_cfg["max_length_direction_mismatch_ratio"].as<float>();
    }
    std::vector<Armor> detectArmors(const std::vector<Light>& lights);
private:
    float max_angle_diff;
    float max_height_diff_ratio;
    float min_light_distance;
    float max_light_distance;
    float min_armor_confidence;
    float max_expected_small_distance_mismatch_ratio;
    float max_expected_large_distance_mismatch_ratio;
    float max_length_direction_mismatch_ratio;
    
    bool isArmorPair(const Light& l1, const Light& l2);
    float getArmorConfidence(const Light& l1, const Light& l2);
    
    std::shared_ptr<YAML::Node> config_file_ptr;
    rclcpp::Node* node;
};

#endif // ARMOR_DETECTOR_H