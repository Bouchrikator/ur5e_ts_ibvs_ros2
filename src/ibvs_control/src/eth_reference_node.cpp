// Eye-to-hand reference generator.
//
// In the eye-to-hand configuration a FIXED scene camera observes both the
// goal object (red cube, tracked by one cube_detector instance) and a green
// marker on the gripper (tracked by a second instance). The controllers servo
// the MARKER features; this node supplies their desired values s*:
//
//   cube centroid --> back-project at Z from the detector's PnP/area estimate
//   --> offset by approach_height along the world +Z axis (via TF)
//   --> project the marker square at that goal point --> publish as a
//       FeatureTarget on ~/feature_target.
//
// The result: the controller drives the gripper marker to hover
// approach_height above the cube. Keeping this in its own node preserves the
// separation detector != reference != controller and lets all four
// controllers (TS-LMI d/c, classic, QMM) share the exact same logic.
//
// Port of the desired-feature section of the ROS1 ts_ibvs_eth_node.

#include <algorithm>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "ibvs_msgs/msg/feature_target.hpp"

namespace ibvs_control {

class EthReferenceNode : public rclcpp::Node {
 public:
  EthReferenceNode() : Node("eth_reference"), tf_buffer_(get_clock()),
                       tf_listener_(tf_buffer_) {
    // Frame of the fixed scene camera (optical convention: Z forward).
    camera_frame_ =
        declare_parameter<std::string>("camera_frame", "overview_optical_frame");
    // Robot base frame; its +Z is the world "up" used for the approach offset.
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    // Height of the marker goal pose above the cube (m).
    approach_height_ = declare_parameter<double>("approach_height", 0.12);
    // Side length of the square gripper marker (m) — sets the desired scale.
    marker_size_ = declare_parameter<double>("marker_size", 0.05);
    // Cube depth estimate validity range (m).
    Z_cube_min_ = declare_parameter<double>("Z_cube_min", 0.1);
    Z_cube_max_ = declare_parameter<double>("Z_cube_max", 2.0);

    pub_ = create_publisher<ibvs_msgs::msg::FeatureTarget>("~/feature_target",
                                                           rclcpp::SensorDataQoS());
    cube_sub_ = create_subscription<ibvs_msgs::msg::FeatureTarget>(
        "/cube_detector/feature_target", rclcpp::SensorDataQoS(),
        std::bind(&EthReferenceNode::cubeCb, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "ETH reference: goal = marker (%.0f mm) %.0f mm above cube, camera=%s",
                marker_size_ * 1e3, approach_height_ * 1e3, camera_frame_.c_str());
  }

 private:
  void cubeCb(const ibvs_msgs::msg::FeatureTarget::SharedPtr msg) {
    // Cube lost: publish nothing — controllers hold the last known s*.
    if (!msg->detected || msg->fx <= 0.0) return;

    // Back-project the cube centroid using the detector's depth estimate.
    const double Z_cube = std::clamp(msg->z_pnp, Z_cube_min_, Z_cube_max_);
    const double x_n = (msg->centroid_u - msg->cx) / msg->fx;
    const double y_n = (msg->centroid_v - msg->cy) / msg->fy;
    const Eigen::Vector3d P_cube(x_n * Z_cube, y_n * Z_cube, Z_cube);

    // World up direction expressed in the camera frame = 3rd column of the
    // rotation of base_frame expressed in camera_frame.
    Eigen::Vector3d up_cam;
    try {
      const auto tfm =
          tf_buffer_.lookupTransform(camera_frame_, base_frame_, tf2::TimePointZero);
      const auto& q = tfm.transform.rotation;
      up_cam = Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix().col(2);
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "TF %s->%s unavailable (%s); no reference published",
                           camera_frame_.c_str(), base_frame_.c_str(), e.what());
      return;
    }

    // Marker goal point: approach_height above the cube, kept in front of
    // the camera (Z >= 5 cm) so the projection stays well-defined.
    Eigen::Vector3d P_goal = P_cube + approach_height_ * up_cam;
    P_goal.z() = std::max(P_goal.z(), 0.05);

    // Project the fronto-parallel marker square centered at the goal point.
    const double x_star = P_goal.x() / P_goal.z();
    const double y_star = P_goal.y() / P_goal.z();
    const double h = (marker_size_ / 2.0) / P_goal.z();

    ibvs_msgs::msg::FeatureTarget out;
    out.header = msg->header;
    out.header.frame_id = camera_frame_;
    out.detected = true;
    out.image_width = msg->image_width;
    out.image_height = msg->image_height;
    // Corner order TL, TR, BR, BL — matches DesiredFeatures::init and the
    // detector's orderCorners convention.
    const double xs[4] = {x_star - h, x_star + h, x_star + h, x_star - h};
    const double ys[4] = {y_star - h, y_star - h, y_star + h, y_star + h};
    for (int i = 0; i < 4; i++) {
      out.features_normalized[2 * i] = xs[i];
      out.features_normalized[2 * i + 1] = ys[i];
      out.corner_pixels[2 * i] = xs[i] * msg->fx + msg->cx;
      out.corner_pixels[2 * i + 1] = ys[i] * msg->fy + msg->cy;
    }
    out.centroid_u = x_star * msg->fx + msg->cx;
    out.centroid_v = y_star * msg->fy + msg->cy;
    out.area = 0.0;
    out.pose_valid = true;
    out.z_pnp = P_goal.z();
    out.depth_valid = false;
    out.z_depth = 0.0;
    out.fx = msg->fx;
    out.fy = msg->fy;
    out.cx = msg->cx;
    out.cy = msg->cy;
    pub_->publish(out);
  }

  std::string camera_frame_, base_frame_;
  double approach_height_, marker_size_, Z_cube_min_, Z_cube_max_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Subscription<ibvs_msgs::msg::FeatureTarget>::SharedPtr cube_sub_;
  rclcpp::Publisher<ibvs_msgs::msg::FeatureTarget>::SharedPtr pub_;
};

}  // namespace ibvs_control

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ibvs_control::EthReferenceNode>());
  rclcpp::shutdown();
  return 0;
}
