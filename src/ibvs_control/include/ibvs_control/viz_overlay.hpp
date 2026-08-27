// Controller-side visualization: overlays desired corners, current corners,
// per-corner trajectories and a status HUD on the perception node's
// detection image. Shared by all controllers (ROC: no duplication).
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace ibvs_control {

class VizOverlay {
 public:
  VizOverlay(rclcpp::Node* node, const std::string& input_topic =
                                     "/cube_detector/detection_image")
      : node_(node) {
    pub_ = node->create_publisher<sensor_msgs::msg::Image>("~/visualization", 10);
    sub_ = node->create_subscription<sensor_msgs::msg::Image>(
        input_topic, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::Image::SharedPtr msg) { latest_image_ = msg; });
  }

  void publish(const std::vector<cv::Point2f>& corners, const cv::Point2f* desired,
               bool converged, const std::string& hud_line) {
    if (!latest_image_ || pub_->get_subscription_count() == 0) return;
    cv::Mat viz;
    try {
      viz = cv_bridge::toCvCopy(latest_image_, "bgr8")->image;
    } catch (const cv_bridge::Exception&) {
      return;
    }

    const cv::Scalar kDesired(0, 255, 0), kCurrent(0, 0, 255), kLink(255, 255, 0);
    for (int i = 0; i < 4; i++) {
      cv::rectangle(viz, cv::Point(desired[i].x - 8, desired[i].y - 8),
                    cv::Point(desired[i].x + 8, desired[i].y + 8), kDesired, 2);
      cv::line(viz, desired[i], desired[(i + 1) % 4], kDesired, 1);
    }
    if (corners.size() == 4) {
      if (trajectories_.size() != 4) trajectories_.resize(4);
      for (int i = 0; i < 4; i++) {
        cv::circle(viz, corners[i], 6, kCurrent, -1);
        cv::line(viz, corners[i], corners[(i + 1) % 4], kCurrent, 2);
        cv::line(viz, corners[i], desired[i], kLink, 1, cv::LINE_AA);
        trajectories_[i].push_back(corners[i]);
        if (trajectories_[i].size() > kMaxTrajectory)
          trajectories_[i].erase(trajectories_[i].begin());
      }
      for (int i = 0; i < 4; i++) {
        for (size_t j = 1; j < trajectories_[i].size(); j++) {
          const float a = static_cast<float>(j) / trajectories_[i].size();
          cv::line(viz, trajectories_[i][j - 1], trajectories_[i][j],
                   cv::Scalar(255 * a, 100, 100), 1, cv::LINE_AA);
        }
      }
    }
    cv::putText(viz, converged ? "CONVERGED" : "SERVOING", {10, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                converged ? kDesired : cv::Scalar(255, 165, 0), 2);
    cv::putText(viz, hud_line, {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);

    auto out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", viz).toImageMsg();
    out->header.stamp = node_->now();
    pub_->publish(*out);
  }

 private:
  static constexpr size_t kMaxTrajectory = 200;
  rclcpp::Node* node_;
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  std::vector<std::vector<cv::Point2f>> trajectories_;
};

}  // namespace ibvs_control
