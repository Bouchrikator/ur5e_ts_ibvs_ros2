// CSV experiment recorder shared by all IBVS controllers (ROS 2 port of the
// ROS 1 DataRecorder). One row per control cycle; the file is flushed on
// every write and closed in the destructor so data survives Ctrl+C.
#pragma once

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace ibvs_control {

class DataRecorder {
 public:
  // mode: short controller tag written into every row and the default file
  // name (e.g. "ts_lmi_d", "ibvs_classic", "qmm_mpc").
  DataRecorder(rclcpp::Node* node, const std::string& mode)
      : node_(node), mode_(mode), tf_buffer_(node->get_clock()), tf_listener_(tf_buffer_) {
    enabled_ = node->declare_parameter<bool>("record_data", true);
    std::string csv_path = node->declare_parameter<std::string>("record_csv", "");
    base_link_ = node->declare_parameter<std::string>("record_base_link", "base_link");
    ee_link_ = node->declare_parameter<std::string>("record_ee_link", "tool0");

    if (!enabled_) return;

    if (csv_path.empty()) {
      // Default under $HOME/ibvs_logs so the container bind-mount can expose
      // recordings to the host (no hardcoded user paths).
      const char* home = std::getenv("HOME");
      std::string dir = std::string(home ? home : "/tmp") + "/ibvs_logs";
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      char stamp[32];
      std::time_t t = std::time(nullptr);
      std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&t));
      csv_path = dir + "/recorder_" + mode_ + "_" + stamp + ".csv";
    }

    file_.open(csv_path);
    if (!file_.is_open()) {
      RCLCPP_ERROR(node->get_logger(), "[Recorder] Cannot open %s; recording disabled",
                   csv_path.c_str());
      enabled_ = false;
      return;
    }

    file_ << "t,mode,converged,err_norm,V,x_norm,dV,Z_used,Z_desired,"
             "v_x,v_y,v_z,w_x,w_y,w_z,"
             "u0,v0,u1,v1,u2,v2,u3,v3,"
             "u0_des,v0_des,u1_des,v1_des,u2_des,v2_des,u3_des,v3_des,"
             "ee_x,ee_y,ee_z,ee_qx,ee_qy,ee_qz,ee_qw,"
             "q0,q1,q2,q3,q4,q5,qd0,qd1,qd2,qd3,qd4,qd5,"
             "h_far,h_close,solver_time_ms,solver_feasible\n";

    joint_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mutex_);
          for (size_t i = 0; i < msg->name.size(); i++) {
            joint_pos_[msg->name[i]] = i < msg->position.size() ? msg->position[i] : 0.0;
            joint_vel_[msg->name[i]] = i < msg->velocity.size() ? msg->velocity[i] : 0.0;
          }
        });

    t0_ = node->now();
    RCLCPP_INFO(node->get_logger(), "[Recorder] Writing to %s", csv_path.c_str());
  }

  ~DataRecorder() {
    if (file_.is_open()) {
      file_.flush();
      file_.close();
    }
  }

  void record(bool converged, double err_norm, double V, double x_norm, double dV,
              double Z_used, double Z_desired, double vx, double vy, double vz, double wx,
              double wy, double wz, const std::vector<cv::Point2f>& corners,
              const cv::Point2f* desired_pixels, double h_far = 0.0, double h_close = 0.0,
              double solver_time_ms = 0.0, bool solver_feasible = true) {
    if (!enabled_ || !file_.is_open()) return;

    const double t = (node_->now() - t0_).seconds();

    // End-effector pose (best effort — leave zeros if TF is not up yet).
    double ee[7] = {0, 0, 0, 0, 0, 0, 1};
    try {
      auto tfm = tf_buffer_.lookupTransform(base_link_, ee_link_, tf2::TimePointZero);
      ee[0] = tfm.transform.translation.x;
      ee[1] = tfm.transform.translation.y;
      ee[2] = tfm.transform.translation.z;
      ee[3] = tfm.transform.rotation.x;
      ee[4] = tfm.transform.rotation.y;
      ee[5] = tfm.transform.rotation.z;
      ee[6] = tfm.transform.rotation.w;
    } catch (const tf2::TransformException&) {
    }

    // UR joint order.
    static const char* kJoints[6] = {"shoulder_pan_joint", "shoulder_lift_joint",
                                     "elbow_joint",        "wrist_1_joint",
                                     "wrist_2_joint",      "wrist_3_joint"};
    double q[6] = {0}, qd[6] = {0};
    {
      std::lock_guard<std::mutex> lk(mutex_);
      for (int i = 0; i < 6; i++) {
        auto ip = joint_pos_.find(kJoints[i]);
        if (ip != joint_pos_.end()) q[i] = ip->second;
        auto iv = joint_vel_.find(kJoints[i]);
        if (iv != joint_vel_.end()) qd[i] = iv->second;
      }
    }

    std::lock_guard<std::mutex> lk(write_mutex_);
    file_ << t << ',' << mode_ << ',' << (converged ? 1 : 0) << ',' << err_norm << ',' << V
          << ',' << x_norm << ',' << dV << ',' << Z_used << ',' << Z_desired;
    file_ << ',' << vx << ',' << vy << ',' << vz << ',' << wx << ',' << wy << ',' << wz;
    for (int i = 0; i < 4; i++) {
      if (i < static_cast<int>(corners.size()))
        file_ << ',' << corners[i].x << ',' << corners[i].y;
      else
        file_ << ",0,0";
    }
    for (int i = 0; i < 4; i++)
      file_ << ',' << desired_pixels[i].x << ',' << desired_pixels[i].y;
    for (int i = 0; i < 7; i++) file_ << ',' << ee[i];
    for (int i = 0; i < 6; i++) file_ << ',' << q[i];
    for (int i = 0; i < 6; i++) file_ << ',' << qd[i];
    file_ << ',' << h_far << ',' << h_close << ',' << solver_time_ms << ','
          << (solver_feasible ? 1 : 0) << '\n';
    file_.flush();
  }

 private:
  rclcpp::Node* node_;
  std::string mode_, base_link_, ee_link_;
  bool enabled_ = false;
  std::ofstream file_;
  rclcpp::Time t0_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  std::mutex mutex_, write_mutex_;
  std::map<std::string, double> joint_pos_, joint_vel_;
};

}  // namespace ibvs_control
