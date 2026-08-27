// Discrete-time TS-PDC IBVS controller with LMI-designed gains.
//
// Theory (Tanaka & Wang 2001, Chaumette reduced error):
//   x(k)   = L_hat^+ e(k)              (reduced state, 6D)
//   x(k+1) = x(k) + Ts B(Z) u(k),      B(Z) = L_hat^+ L_e(Z)
//   TS:    B(Z) = h_far B1 + h_close B2   on a = 1/Z
//   PDC:   u(k) = -gain_scale (h_far F1 + h_close F2) x(k)
//   Stability: G_ij' P G_ij - P < 0 via offline LMI synthesis
//   (solve_ts_pdc_lmi_discrete.py); verified again at startup.
//
// Vision is delegated to ibvs_perception/cube_detector (separation of
// concerns): this node consumes ibvs_msgs/FeatureTarget only.

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "ibvs_control/convergence_monitor.hpp"
#include "ibvs_control/data_recorder.hpp"
#include "ibvs_control/feature_utils.hpp"
#include "ibvs_control/gains_loader.hpp"
#include "ibvs_control/ibvs_math.hpp"
#include "ibvs_control/twist_commander.hpp"
#include "ibvs_control/viz_overlay.hpp"
#include "ibvs_msgs/msg/feature_target.hpp"

namespace ibvs_control {

class TsIbvsDiscreteNode : public rclcpp::Node {
 public:
  TsIbvsDiscreteNode() : Node("ts_ibvs_lmi_discrete") {
    // --- Target / model ---
    cube_w_ = declare_parameter<double>("cube_size_x", 0.05);
    cube_h_ = declare_parameter<double>("cube_size_y", 0.05);
    desired_Z_ = declare_parameter<double>("desired_Z", 0.4);
    Z0_ = declare_parameter<double>("Z0", 0.5);
    Z_min_ = declare_parameter<double>("Z_min", 0.2);
    Z_max_ = declare_parameter<double>("Z_max", 1.0);
    Ts_ = declare_parameter<double>("Ts", 0.033);

    // Intrinsics used for desired-pixel computation until the first
    // FeatureTarget provides live values.
    fx_ = declare_parameter<double>("cam_px", 554.254691191187);
    fy_ = declare_parameter<double>("cam_py", 554.254691191187);
    cx_ = declare_parameter<double>("cam_u0", 320.5);
    cy_ = declare_parameter<double>("cam_v0", 240.5);

    // --- Control ---
    gain_scale_ = declare_parameter<double>("gain_scale", 1.0);
    max_lin_ = declare_parameter<double>("max_linear_vel", 0.1);
    max_ang_ = declare_parameter<double>("max_angular_vel", 0.3);
    edge_margin_ = declare_parameter<double>("edge_margin", 50.0);
    lyap_alpha_ = declare_parameter<double>("lyapunov_ema_alpha", 0.05);
    // Detection-glitch rejection: reject frames where the reduced state
    // jumps by more than this ratio while converged (corner-order flips).
    state_jump_ratio_ = declare_parameter<double>("state_jump_ratio", 10.0);

    // --- Convergence (per-axis pixel criterion) ---
    conv_thresh_px_ = declare_parameter<double>("conv_thresh_px", 3.0);
    const int conv_frames = declare_parameter<int>("convergence_frames", 3);
    monitor_ = std::make_unique<ConvergenceMonitor>(conv_thresh_px_, conv_frames);

    max_lost_frames_ = declare_parameter<int>("max_lost_frames", 10);

    // --- Gains ---
    const std::string gains_file = declare_parameter<std::string>("gains_file", "");
    initializeMatrices();
    TsGains gains;
    if (!gains_file.empty() && loadTsGains(gains_file, gains)) {
      F1_ = gains.F1;
      F2_ = gains.F2;
      P_ = gains.P;
      if (gains.L_hat_pinv.size() > 0 && (L_hat_pinv_ - gains.L_hat_pinv).norm() > 1e-6) {
        L_hat_pinv_ = gains.L_hat_pinv;
        B1_ = L_hat_pinv_ * stackedInteractionMatrix(s_star_.s_star, Z_max_);
        B2_ = L_hat_pinv_ * stackedInteractionMatrix(s_star_.s_star, Z_min_);
      }
      RCLCPP_INFO(get_logger(), "Loaded LMI gains from %s", gains_file.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "gains_file missing/invalid — using defaults (NOT LMI-optimal)");
      F1_ = 0.3 * Eigen::MatrixXd::Identity(6, 6);
      F2_ = 0.7 * Eigen::MatrixXd::Identity(6, 6);
      P_ = Eigen::MatrixXd::Identity(6, 6);
    }
    verifyStabilityOffline();

    // --- I/O ---
    commander_ = std::make_unique<TwistCommander>(this);
    recorder_ = std::make_unique<DataRecorder>(this, "ts_lmi_d");
    viz_ = std::make_unique<VizOverlay>(this);

    error_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/error", 10);
    state_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/reduced_state", 10);
    ts_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/ts_weights", 10);
    lyap_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/lyapunov", 10);

    feature_sub_ = create_subscription<ibvs_msgs::msg::FeatureTarget>(
        "/cube_detector/feature_target", rclcpp::SensorDataQoS(),
        std::bind(&TsIbvsDiscreteNode::featureCb, this, std::placeholders::_1));
    desired_sub_ = subscribeDynamicDesired(this, s_star_, desired_received_);

    RCLCPP_INFO(get_logger(),
                "TS-PDC IBVS Discrete: cube=%.3fx%.3f Z*=%.2f Z=[%.2f,%.2f] Ts=%.3f "
                "gain_scale=%.2f conv=%.1fpx",
                cube_w_, cube_h_, desired_Z_, Z_min_, Z_max_, Ts_, gain_scale_,
                conv_thresh_px_);
  }

 private:
  void initializeMatrices() {
    s_star_.init(cube_w_, cube_h_, desired_Z_, fx_, fy_, cx_, cy_);
    L_hat_ = stackedInteractionMatrix(s_star_.s_star, Z0_);
    L_hat_pinv_ = L_hat_.completeOrthogonalDecomposition().pseudoInverse();
    B1_ = L_hat_pinv_ * stackedInteractionMatrix(s_star_.s_star, Z_max_);
    B2_ = L_hat_pinv_ * stackedInteractionMatrix(s_star_.s_star, Z_min_);
  }

  // Startup sanity check: sweep Z and verify the 3 discrete LMI conditions
  // G' P G - P < 0 with the loaded gains. A failed check means the gains
  // file does not match the model parameters (Z range, Ts).
  void verifyStabilityOffline() {
    const int kSamples = 50;
    double worst = -1e10;
    for (int k = 0; k < kSamples; k++) {
      const double Z = Z_min_ + k * (Z_max_ - Z_min_) / (kSamples - 1);
      double hf, hc;
      tsMemberships(Z, Z_min_, Z_max_, hf, hc);
      const Eigen::MatrixXd G =
          Eigen::MatrixXd::Identity(6, 6) -
          Ts_ * (hf * B1_ + hc * B2_) * gain_scale_ * (hf * F1_ + hc * F2_);
      const Eigen::MatrixXd Q = G.transpose() * P_ * G - P_;
      worst = std::max(
          worst, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Q).eigenvalues().maxCoeff());
    }
    if (worst < 0)
      RCLCPP_INFO(get_logger(), "Discrete stability sweep OK: max eig(G'PG-P) = %.3e", worst);
    else
      RCLCPP_WARN(get_logger(), "Discrete stability sweep FAILED: max eig = %.3e >= 0", worst);
  }

  void publishDebug(const Eigen::VectorXd& e, const Eigen::VectorXd& x,
                    const Eigen::VectorXd& u, double hf, double hc, double Z, double V,
                    double dV) {
    {
      std_msgs::msg::Float64MultiArray m;
      m.data.assign(e.data(), e.data() + 8);
      error_pub_->publish(m);
    }
    {
      std_msgs::msg::Float64MultiArray m;
      m.data.resize(7);
      for (int i = 0; i < 6; i++) m.data[i] = x(i);
      m.data[6] = x.norm();
      state_pub_->publish(m);
    }
    {
      std_msgs::msg::Float64MultiArray m;
      m.data = {Z,        hc,      hf,       u(0),     u(1),    u(2),   u(3), u(4),
                u(5),     u.norm(), e.norm(), x.norm(), 1.0 / Z, Z_min_, Z_max_};
      ts_pub_->publish(m);
    }
    {
      std_msgs::msg::Float64MultiArray m;
      m.data = {V, dV, x.norm(), (dV < 0) ? 1.0 : 0.0};
      lyap_pub_->publish(m);
    }
  }

  // EMA-filtered discrete Lyapunov difference: raw dV = V(k) - V(k-1) is
  // noisy at camera rate; the EMA exposes the trend for PlotJuggler.
  double lyapunovDifference(double V, double& raw) {
    if (first_lyap_) {
      first_lyap_ = false;
      dV_ema_ = 0.0;
      raw = 0.0;
      return 0.0;
    }
    raw = V - V_prev_;
    dV_ema_ = lyap_alpha_ * raw + (1.0 - lyap_alpha_) * dV_ema_;
    return dV_ema_;
  }

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

    // Refresh desired pixels with live intrinsics (sim vs real camera).
    // With a dynamic reference s* comes from the topic instead.
    if (msg->fx > 0 && std::fabs(msg->fx - fx_) > 1e-9) {
      fx_ = msg->fx;
      fy_ = msg->fy;
      cx_ = msg->cx;
      cy_ = msg->cy;
      if (!desired_sub_) s_star_.init(cube_w_, cube_h_, desired_Z_, fx_, fy_, cx_, cy_);
    }

    Eigen::VectorXd s_cur;
    std::vector<cv::Point2f> corners;
    unpackFeatures(*msg, s_cur, corners);
    const double Z_used = selectDepth(*msg, Z_min_, Z_max_);

    const Eigen::VectorXd e = s_cur - s_star_.s_star_flat;
    const double max_px_err = maxPixelError(e, fx_, fy_);
    const Eigen::VectorXd x = L_hat_pinv_ * e;

    // Detection-glitch rejection (corner-order flips while holding).
    if (monitor_->converged() && prev_x_.norm() > 1e-6 &&
        (x - prev_x_).norm() / prev_x_.norm() > state_jump_ratio_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "Detection glitch rejected (state jump)");
      commander_->publishZero();
      return;
    }

    double hf, hc;
    tsMemberships(Z_used, Z_min_, Z_max_, hf, hc);
    Eigen::VectorXd u = -gain_scale_ * ((hf * F1_ + hc * F2_) * x);

    const double V = (x.transpose() * P_ * x)(0, 0);
    double raw_dV;
    const double dV = lyapunovDifference(V, raw_dV);

    const bool hold = monitor_->update(max_px_err);
    if (monitor_->justConverged())
      RCLCPP_INFO(get_logger(), "*** CONVERGED: max|e_axis|=%.2f < %.2f px ***", max_px_err,
                  conv_thresh_px_);
    if (monitor_->justRearmed())
      RCLCPP_WARN(get_logger(), "Re-arming: max|e_axis|=%.2f px", max_px_err);

    char hud[128];
    std::snprintf(hud, sizeof(hud), "Z:%.3f |x|:%.5f max_px:%.1f h=(%.2f,%.2f)", Z_used,
                  x.norm(), max_px_err, hf, hc);

    if (hold) {
      commander_->publishZero();
      publishDebug(e, x, u, hf, hc, Z_used, V, dV);
      viz_->publish(corners, s_star_.pixels, true, hud);
      prev_x_ = x;
      V_prev_ = V;
      return;
    }

    u *= edgeFactor(msg->centroid_u, msg->centroid_v, msg->image_width, msg->image_height,
                    edge_margin_);

    Eigen::Matrix<double, 6, 1> u_sat;
    for (int i = 0; i < 3; i++) u_sat(i) = clampAbs(u(i), max_lin_);
    for (int i = 3; i < 6; i++) u_sat(i) = clampAbs(u(i), max_ang_);
    commander_->publish(u_sat);

    publishDebug(e, x, u_sat, hf, hc, Z_used, V, dV);
    viz_->publish(corners, s_star_.pixels, false, hud);
    recorder_->record(false, e.norm(), V, x.norm(), dV, Z_used, desired_Z_, u_sat(0),
                      u_sat(1), u_sat(2), u_sat(3), u_sat(4), u_sat(5), corners,
                      s_star_.pixels, hf, hc);

    prev_x_ = x;
    V_prev_ = V;

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                         "Z=%.2f h=(%.2f,%.2f) ||x||=%.4f V=%.4f dV=%.4f max_px=%.1f",
                         Z_used, hf, hc, x.norm(), V, dV, max_px_err);
  }

  // Model / gains
  double cube_w_, cube_h_, desired_Z_, Z0_, Z_min_, Z_max_, Ts_;
  double fx_, fy_, cx_, cy_;
  DesiredFeatures s_star_;
  Eigen::MatrixXd L_hat_, L_hat_pinv_, B1_, B2_, F1_, F2_, P_;

  // Control params
  double gain_scale_, max_lin_, max_ang_, edge_margin_, lyap_alpha_, state_jump_ratio_;
  double conv_thresh_px_;
  int max_lost_frames_, lost_count_ = 0;

  // State
  Eigen::VectorXd prev_x_ = Eigen::VectorXd::Zero(6);
  double V_prev_ = 0.0, dV_ema_ = 0.0;
  bool first_lyap_ = true;

  std::unique_ptr<ConvergenceMonitor> monitor_;
  std::unique_ptr<TwistCommander> commander_;
  std::unique_ptr<DataRecorder> recorder_;
  std::unique_ptr<VizOverlay> viz_;

  rclcpp::Subscription<ibvs_msgs::msg::FeatureTarget>::SharedPtr feature_sub_;
  rclcpp::Subscription<ibvs_msgs::msg::FeatureTarget>::SharedPtr desired_sub_;
  bool desired_received_ = false;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr error_pub_, state_pub_,
      ts_pub_, lyap_pub_;
};

}  // namespace ibvs_control

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ibvs_control::TsIbvsDiscreteNode>());
  rclcpp::shutdown();
  return 0;
}
