// Desired-feature helpers shared by all controllers: builds s* from the
// target geometry and computes per-axis pixel errors from a FeatureTarget.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ibvs_msgs/msg/feature_target.hpp"

namespace ibvs_control {

struct DesiredFeatures {
  Eigen::MatrixXd s_star;       // 4x2 normalized (TL, TR, BR, BL)
  Eigen::VectorXd s_star_flat;  // 8x1
  cv::Point2f pixels[4];        // desired pixel positions (for viz/recording)

  // A fronto-parallel rectangular target of size (w x h) at depth Z* projects
  // to corners at +-(w/2)/Z*, +-(h/2)/Z* in normalized coordinates.
  void init(double cube_w, double cube_h, double desired_Z, double fx, double fy, double cx,
            double cy) {
    const double hx = (cube_w / 2.0) / desired_Z;
    const double hy = (cube_h / 2.0) / desired_Z;
    s_star.resize(4, 2);
    s_star << -hx, -hy, hx, -hy, hx, hy, -hx, hy;
    s_star_flat.resize(8);
    for (int i = 0; i < 4; i++) {
      s_star_flat(2 * i) = s_star(i, 0);
      s_star_flat(2 * i + 1) = s_star(i, 1);
      pixels[i] = cv::Point2f(s_star(i, 0) * fx + cx, s_star(i, 1) * fy + cy);
    }
  }

  // Overwrites s* with externally tracked desired features. Used in the
  // eye-to-hand configuration where the reference generator projects the
  // marker goal pose (above the cube) into the fixed scene camera each frame.
  void setFromMsg(const ibvs_msgs::msg::FeatureTarget& msg) {
    if (s_star.rows() != 4) s_star.resize(4, 2);
    if (s_star_flat.size() != 8) s_star_flat.resize(8);
    for (int i = 0; i < 4; i++) {
      s_star(i, 0) = msg.features_normalized[2 * i];
      s_star(i, 1) = msg.features_normalized[2 * i + 1];
      s_star_flat(2 * i) = s_star(i, 0);
      s_star_flat(2 * i + 1) = s_star(i, 1);
      pixels[i] = cv::Point2f(msg.corner_pixels[2 * i], msg.corner_pixels[2 * i + 1]);
    }
  }
};

// Optional dynamic desired features (eye-to-hand). When the
// 'desired_feature_topic' parameter is non-empty the controller tracks an
// external reference (the eth_reference node) instead of the static centered
// square: every incoming message overwrites s_star, and 'received' flips true
// after the first one so the controller can hold still until a reference
// exists. Returns nullptr when the topic is empty (classic eye-in-hand).
// Shared by all four controllers so the eye-to-hand wiring lives in one place.
inline rclcpp::Subscription<ibvs_msgs::msg::FeatureTarget>::SharedPtr
subscribeDynamicDesired(rclcpp::Node* node, DesiredFeatures& s_star, bool& received) {
  const auto topic = node->declare_parameter<std::string>("desired_feature_topic", "");
  if (topic.empty()) return nullptr;
  RCLCPP_INFO(node->get_logger(), "Tracking dynamic desired features from %s",
              topic.c_str());
  return node->create_subscription<ibvs_msgs::msg::FeatureTarget>(
      topic, rclcpp::SensorDataQoS(),
      [&s_star, &received](ibvs_msgs::msg::FeatureTarget::SharedPtr msg) {
        if (!msg->detected) return;  // keep the last known reference
        s_star.setFromMsg(*msg);
        received = true;
      });
}

// Worst per-axis pixel error over the 8 normalized error components; the
// convergence criterion requires EVERY component below the threshold.
inline double maxPixelError(const Eigen::VectorXd& e, double fx, double fy) {
  double m = 0.0;
  for (int i = 0; i < 4; i++) {
    m = std::max(m, std::fabs(e(2 * i)) * fx);
    m = std::max(m, std::fabs(e(2 * i + 1)) * fy);
  }
  return m;
}

// Extracts corners/features from the message into working types.
inline void unpackFeatures(const ibvs_msgs::msg::FeatureTarget& msg, Eigen::VectorXd& s,
                           std::vector<cv::Point2f>& corners) {
  s.resize(8);
  corners.resize(4);
  for (int i = 0; i < 4; i++) {
    s(2 * i) = msg.features_normalized[2 * i];
    s(2 * i + 1) = msg.features_normalized[2 * i + 1];
    corners[i] =
        cv::Point2f(msg.corner_pixels[2 * i], msg.corner_pixels[2 * i + 1]);
  }
}

// Depth selection: prefer the aligned depth-image sample, fall back to the
// PnP/area estimate, clamped to the model's validity range.
inline double selectDepth(const ibvs_msgs::msg::FeatureTarget& msg, double Z_min,
                          double Z_max) {
  const double Z = msg.depth_valid ? msg.z_depth : msg.z_pnp;
  return std::clamp(Z, Z_min, Z_max);
}

}  // namespace ibvs_control
