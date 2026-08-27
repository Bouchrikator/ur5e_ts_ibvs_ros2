// Publishes camera-frame velocity screws as TwistStamped for MoveIt Servo,
// with optional re-expression in an output frame (e.g. tool0). Only the
// rotation part of the TF is used: the twist is a free vector pair, and
// Servo expects the command expressed in (not translated to) the frame.
#pragma once

#include <string>

#include <Eigen/Dense>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <moveit_msgs/srv/servo_command_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace ibvs_control {

class TwistCommander {
 public:
  TwistCommander(rclcpp::Node* node) : node_(node), tf_buffer_(node->get_clock()),
                                       tf_listener_(tf_buffer_) {
    servo_topic_ = node->declare_parameter<std::string>("servo_topic",
                                                        "/servo_node/delta_twist_cmds");
    // Frame the raw control law is computed in (camera optical frame).
    command_frame_ =
        node->declare_parameter<std::string>("command_frame", "d435_color_optical_frame");
    // Frame in which the twist is re-expressed before publishing; empty
    // keeps the camera frame (simulation default).
    output_frame_ = node->declare_parameter<std::string>("output_frame", "");
    // Compensate the reference-point offset between camera and output frame:
    // v_out = R v - (R w) x p. Needed on the real robot where the camera is
    // offset from tool0 (used by QMM-MPC).
    compensate_reference_point_ =
        node->declare_parameter<bool>("compensate_reference_point", false);

    pub_ = node->create_publisher<geometry_msgs::msg::TwistStamped>(servo_topic_, 10);

    // Jazzy servo_node starts with no command type set and drops twists
    // until /servo_node/switch_command_type is called with TWIST.
    const auto slash = servo_topic_.rfind('/');
    const std::string servo_ns = slash == std::string::npos ? "" : servo_topic_.substr(0, slash);
    switch_client_ = node->create_client<moveit_msgs::srv::ServoCommandType>(
        servo_ns + "/switch_command_type");
  }

  void ensureTwistMode() {
    if (twist_mode_set_ || switch_pending_) return;
    if (!switch_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                           "Waiting for servo switch_command_type service");
      return;
    }
    auto req = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
    req->command_type = moveit_msgs::srv::ServoCommandType::Request::TWIST;
    switch_pending_ = true;
    switch_client_->async_send_request(
        req, [this](rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedFuture f) {
          switch_pending_ = false;
          twist_mode_set_ = f.get()->success;
          if (twist_mode_set_) {
            RCLCPP_INFO(node_->get_logger(), "Servo switched to TWIST command type");
          }
        });
  }

  void publish(const Eigen::Matrix<double, 6, 1>& u_cam) {
    ensureTwistMode();
    Eigen::Vector3d v = u_cam.head<3>();
    Eigen::Vector3d w = u_cam.tail<3>();
    std::string frame = command_frame_;

    if (!output_frame_.empty() && output_frame_ != command_frame_) {
      try {
        auto tfm =
            tf_buffer_.lookupTransform(output_frame_, command_frame_, tf2::TimePointZero);
        const auto& q = tfm.transform.rotation;
        const Eigen::Quaterniond R(q.w, q.x, q.y, q.z);
        v = R * v;
        w = R * w;
        if (compensate_reference_point_) {
          const Eigen::Vector3d p(tfm.transform.translation.x, tfm.transform.translation.y,
                                  tfm.transform.translation.z);
          v -= w.cross(p);
        }
        frame = output_frame_;
      } catch (const tf2::TransformException& e) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                             "TF %s->%s unavailable (%s); publishing in %s",
                             command_frame_.c_str(), output_frame_.c_str(), e.what(),
                             command_frame_.c_str());
      }
    }

    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = node_->now();
    cmd.header.frame_id = frame;
    cmd.twist.linear.x = v.x();
    cmd.twist.linear.y = v.y();
    cmd.twist.linear.z = v.z();
    cmd.twist.angular.x = w.x();
    cmd.twist.angular.y = w.y();
    cmd.twist.angular.z = w.z();
    pub_->publish(cmd);
  }

  void publishZero() { publish(Eigen::Matrix<double, 6, 1>::Zero()); }

 private:
  rclcpp::Node* node_;
  std::string servo_topic_, command_frame_, output_frame_;
  bool compensate_reference_point_;
  bool twist_mode_set_ = false;
  bool switch_pending_ = false;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pub_;
  rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_client_;
};

}  // namespace ibvs_control
