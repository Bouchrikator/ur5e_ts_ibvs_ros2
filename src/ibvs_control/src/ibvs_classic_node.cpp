// Classic IBVS (Chaumette) with desired-configuration interaction matrix and
// adaptive Levenberg-Marquardt damping — the baseline controller.
//
// Control law:
//   L* = L(s*, Z*)                       (constant 8x6, desired features)
//   mu = mu_min + mu0 exp(-||e|| / eps)  (adaptive LM damping: more damping
//                                         near the goal for smooth landing)
//   v  = -lambda (L*' L* + mu I)^-1 L*' e
//
// Anti-divergence: near the goal (||e|| < scale_radius) the command is scaled
// down linearly to avoid overshoot from the constant-L* approximation.
// No velocity saturation: MoveIt Servo applies its own limits (faithful to
// the ROS 1 baseline used for the paper comparisons).

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "ibvs_control/convergence_monitor.hpp"
#include "ibvs_control/data_recorder.hpp"
#include "ibvs_control/feature_utils.hpp"
#include "ibvs_control/ibvs_math.hpp"
#include "ibvs_control/twist_commander.hpp"
#include "ibvs_control/viz_overlay.hpp"
#include "ibvs_msgs/msg/feature_target.hpp"

namespace ibvs_control {

class IbvsClassicNode : public rclcpp::Node {
 public:
  IbvsClassicNode() : Node("ibvs_classic") {
    cube_w_ = declare_parameter<double>("cube_size_x", 0.05);
    cube_h_ = declare_parameter<double>("cube_size_y", 0.05);
    desired_Z_ = declare_parameter<double>("desired_Z", 0.4);
    Z_min_ = declare_parameter<double>("Z_min", 0.1);
    Z_max_ = declare_parameter<double>("Z_max", 2.0);

    fx_ = declare_parameter<double>("cam_px", 554.254691191187);
    fy_ = declare_parameter<double>("cam_py", 554.254691191187);
    cx_ = declare_parameter<double>("cam_u0", 320.5);
    cy_ = declare_parameter<double>("cam_v0", 240.5);

    lambda_ = declare_parameter<double>("lambda", 0.5);
    lm_mu0_ = declare_parameter<double>("lm_mu0", 0.02);
    lm_mu_min_ = declare_parameter<double>("lm_mu_min", 0.001);
    lm_eps_ = declare_parameter<double>("lm_eps", 0.10);
    scale_radius_ = declare_parameter<double>("error_scale_radius", 0.10);
    edge_margin_ = declare_parameter<double>("edge_margin", 50.0);
    max_lost_frames_ = declare_parameter<int>("max_lost_frames", 10);

    conv_thresh_px_ = declare_parameter<double>("conv_thresh_px", 3.0);
    const int conv_frames = declare_parameter<int>("convergence_frames", 1);
    monitor_ = std::make_unique<ConvergenceMonitor>(conv_thresh_px_, conv_frames);

    commander_ = std::make_unique<TwistCommander>(this);
    recorder_ = std::make_unique<DataRecorder>(this, "ibvs_classic");
    viz_ = std::make_unique<VizOverlay>(this);

    error_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/error", 10);
    lyap_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/lyapunov", 10);

    s_star_.init(cube_w_, cube_h_, desired_Z_, fx_, fy_, cx_, cy_);
    L_star_ = stackedInteractionMatrix(s_star_.s_star, desired_Z_);

    feature_sub_ = create_subscription<ibvs_msgs::msg::FeatureTarget>(
        "/cube_detector/feature_target", rclcpp::SensorDataQoS(),
        std::bind(&IbvsClassicNode::featureCb, this, std::placeholders::_1));
    desired_sub_ = subscribeDynamicDesired(this, s_star_, desired_received_);

    RCLCPP_INFO(get_logger(),
                "Classic IBVS: lambda=%.2f mu=(%.3f+%.3f exp(-e/%.2f)) conv=%.1fpx",
                lambda_, lm_mu_min_, lm_mu0_, lm_eps_, conv_thresh_px_);
  }

 private:
  void featureCb(const ibvs_msgs::msg::FeatureTarget::SharedPtr msg) {
    if (!msg->detected) {
      if (++lost_count_ >= max_lost_frames_) commander_->publishZero();
      return;
    }
    lost_count_ = 0;

    // Eye-to-hand: hold still until the reference generator has seen the cube.
    if (desired_sub_ && !desired_received_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Waiting for desired features (eye-to-hand)");
      commander_->publishZero();
      return;
    }

    if (msg->fx > 0 && std::fabs(msg->fx - fx_) > 1e-9) {
      fx_ = msg->fx;
      fy_ = msg->fy;
      cx_ = msg->cx;
      cy_ = msg->cy;
      // Dynamic s* comes from the topic; L* stays the fixed estimate at the
      // nominal desired pose (same policy as the ROS1 eye-to-hand node).
      if (!desired_sub_) {
        s_star_.init(cube_w_, cube_h_, desired_Z_, fx_, fy_, cx_, cy_);
        L_star_ = stackedInteractionMatrix(s_star_.s_star, desired_Z_);
      }
    }

    Eigen::VectorXd s_cur;
    std::vector<cv::Point2f> corners;
    unpackFeatures(*msg, s_cur, corners);

    const Eigen::VectorXd e = s_cur - s_star_.s_star_flat;
    const double e_norm = e.norm();
    const double max_px_err = maxPixelError(e, fx_, fy_);

    // Adaptive LM damping + damped least-squares control law.
    const double mu = lm_mu_min_ + lm_mu0_ * std::exp(-e_norm / lm_eps_);
    const Eigen::MatrixXd H =
        L_star_.transpose() * L_star_ + mu * Eigen::MatrixXd::Identity(6, 6);
    Eigen::VectorXd v = -lambda_ * H.ldlt().solve(L_star_.transpose() * e);

    // Anti-divergence scaling near the goal.
    if (e_norm < scale_radius_) v *= 0.1 + 0.9 * (e_norm / scale_radius_);

    const double V = 0.5 * e.squaredNorm();
    const double t = now().seconds();
    double Vd = 0.0;
    if (!first_lyap_ && prev_time_ > 0 && t - prev_time_ > 1e-6)
      Vd = (V - V_prev_) / (t - prev_time_);
    first_lyap_ = false;

    const bool hold = monitor_->update(max_px_err);
    if (monitor_->justConverged())
      RCLCPP_INFO(get_logger(), "*** CONVERGED: max|e_axis|=%.2f px ***", max_px_err);
    if (monitor_->justRearmed())
      RCLCPP_WARN(get_logger(), "Re-arming: max|e_axis|=%.2f px", max_px_err);

    char hud[128];
    std::snprintf(hud, sizeof(hud), "||e||:%.4f max_px:%.1f mu:%.4f", e_norm, max_px_err,
                  mu);

    if (hold) {
      commander_->publishZero();
      publishDebug(e, Eigen::VectorXd::Zero(6), V, Vd);
      viz_->publish(corners, s_star_.pixels, true, hud);
      finishCycle(t, V);
      return;
    }

    v *= edgeFactor(msg->centroid_u, msg->centroid_v, msg->image_width, msg->image_height,
                    edge_margin_);

    Eigen::Matrix<double, 6, 1> cmd = v;
    commander_->publish(cmd);

    const double Z_used = selectDepth(*msg, Z_min_, Z_max_);
    publishDebug(e, v, V, Vd);
    viz_->publish(corners, s_star_.pixels, false, hud);
    recorder_->record(false, e_norm, V, e_norm, Vd, Z_used, desired_Z_, v(0), v(1), v(2),
                      v(3), v(4), v(5), corners, s_star_.pixels);
    finishCycle(t, V);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                         "||e||=%.4f max_px=%.1f mu=%.4f v=(%.3f,%.3f,%.3f)", e_norm,
                         max_px_err, mu, v(0), v(1), v(2));
  }

  void publishDebug(const Eigen::VectorXd& e, const Eigen::VectorXd& v, double V,
                    double Vd) {
    {
      // 14 fields: 8 error components + 6 velocity components (ROS 1 layout).
      std_msgs::msg::Float64MultiArray m;
      m.data.resize(14);
      for (int i = 0; i < 8; i++) m.data[i] = e(i);
      for (int i = 0; i < 6; i++) m.data[8 + i] = v(i);
      error_pub_->publish(m);
    }
    {
      std_msgs::msg::Float64MultiArray m;
      m.data = {V, Vd, e.norm(), (Vd < 0) ? 1.0 : 0.0};
      lyap_pub_->publish(m);
    }
  }

  void finishCycle(double t, double V) {
    prev_time_ = t;
    V_prev_ = V;
  }

  double cube_w_, cube_h_, desired_Z_, Z_min_, Z_max_;
  double fx_, fy_, cx_, cy_;
  DesiredFeatures s_star_;
  Eigen::MatrixXd L_star_;

  double lambda_, lm_mu0_, lm_mu_min_, lm_eps_, scale_radius_, edge_margin_;
  double conv_thresh_px_;
  int max_lost_frames_, lost_count_ = 0;

  double prev_time_ = 0.0, V_prev_ = 0.0;
  bool first_lyap_ = true;

  std::unique_ptr<ConvergenceMonitor> monitor_;
  std::unique_ptr<TwistCommander> commander_;
  std::unique_ptr<DataRecorder> recorder_;
  std::unique_ptr<VizOverlay> viz_;

  rclcpp::Subscription<ibvs_msgs::msg::FeatureTarget>::SharedPtr feature_sub_;
  rclcpp::Subscription<ibvs_msgs::msg::FeatureTarget>::SharedPtr desired_sub_;
  bool desired_received_ = false;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr error_pub_, lyap_pub_;
};

}  // namespace ibvs_control

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ibvs_control::IbvsClassicNode>());
  rclcpp::shutdown();
  return 0;
}
