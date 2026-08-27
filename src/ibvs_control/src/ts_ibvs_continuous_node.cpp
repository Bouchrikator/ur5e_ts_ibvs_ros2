// Continuous-time TS-PDC IBVS controller with LMI-designed gains.
//
// Theory (Chaumette reduced error):
//   x = L_hat^+ e,  x_dot = B(Z) u,  B(Z) = L_hat^+ L_e(Z)
//   TS:  x_dot = (h_far B1 + h_close B2) u   on a = 1/Z
//   PDC: u = -(h_far F1 + h_close F2) x
//   Stability: sym(P G_ij) < 0 via offline LMI synthesis
//   (solve_ts_lmi_reduced.py); verified again at startup.
//
// Convergence uses the reduced-state norm ||x|| (continuous formulation),
// unlike the discrete node which uses a per-axis pixel criterion.

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "ibvs_control/data_recorder.hpp"
#include "ibvs_control/feature_utils.hpp"
#include "ibvs_control/gains_loader.hpp"
#include "ibvs_control/ibvs_math.hpp"
#include "ibvs_control/twist_commander.hpp"
#include "ibvs_control/viz_overlay.hpp"
#include "ibvs_msgs/msg/feature_target.hpp"

namespace ibvs_control {

class TsIbvsContinuousNode : public rclcpp::Node {
 public:
  TsIbvsContinuousNode() : Node("ts_ibvs_lmi") {
    cube_w_ = declare_parameter<double>("cube_size_x", 0.05);
    cube_h_ = declare_parameter<double>("cube_size_y", 0.05);
    desired_Z_ = declare_parameter<double>("desired_Z", 0.4);
    Z0_ = declare_parameter<double>("Z0", 0.5);
    Z_min_ = declare_parameter<double>("Z_min", 0.2);
    Z_max_ = declare_parameter<double>("Z_max", 1.0);

    fx_ = declare_parameter<double>("cam_px", 554.254691191187);
    fy_ = declare_parameter<double>("cam_py", 554.254691191187);
    cx_ = declare_parameter<double>("cam_u0", 320.5);
    cy_ = declare_parameter<double>("cam_v0", 240.5);

    gain_scale_ = declare_parameter<double>("gain_scale", 1.0);
    max_lin_ = declare_parameter<double>("max_linear_vel", 0.1);
    max_ang_ = declare_parameter<double>("max_angular_vel", 0.3);
    edge_margin_ = declare_parameter<double>("edge_margin", 50.0);
    error_threshold_ = declare_parameter<double>("error_threshold", 0.005);
    conv_frames_ = declare_parameter<int>("convergence_frames", 3);
    state_jump_ratio_ = declare_parameter<double>("state_jump_ratio", 10.0);
    max_lost_frames_ = declare_parameter<int>("max_lost_frames", 10);

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

    commander_ = std::make_unique<TwistCommander>(this);
    recorder_ = std::make_unique<DataRecorder>(this, "ts_lmi_c");
    viz_ = std::make_unique<VizOverlay>(this);

    error_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/error", 10);
    state_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/reduced_state", 10);
    ts_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/ts_weights", 10);
    lyap_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/lyapunov", 10);

    feature_sub_ = create_subscription<ibvs_msgs::msg::FeatureTarget>(
        "/cube_detector/feature_target", rclcpp::SensorDataQoS(),
        std::bind(&TsIbvsContinuousNode::featureCb, this, std::placeholders::_1));
    desired_sub_ = subscribeDynamicDesired(this, s_star_, desired_received_);

    RCLCPP_INFO(get_logger(),
                "TS-PDC IBVS Continuous: cube=%.3fx%.3f Z*=%.2f Z=[%.2f,%.2f] thresh=%.4f",
                cube_w_, cube_h_, desired_Z_, Z_min_, Z_max_, error_threshold_);
  }

 private:
  void initializeMatrices() {
    s_star_.init(cube_w_, cube_h_, desired_Z_, fx_, fy_, cx_, cy_);
    L_hat_ = stackedInteractionMatrix(s_star_.s_star, Z0_);
    L_hat_pinv_ = L_hat_.completeOrthogonalDecomposition().pseudoInverse();
    B1_ = L_hat_pinv_ * stackedInteractionMatrix(s_star_.s_star, Z_max_);
    B2_ = L_hat_pinv_ * stackedInteractionMatrix(s_star_.s_star, Z_min_);
  }

  void verifyStabilityOffline() {
    // Eigenvalue sweep of the closed loop over the Z range.
    const int kSamples = 50;
    double max_re = -1e10;
    for (int k = 0; k < kSamples; k++) {
      const double Z = Z_min_ + k * (Z_max_ - Z_min_) / (kSamples - 1);
      double hf, hc;
      tsMemberships(Z, Z_min_, Z_max_, hf, hc);
      const Eigen::MatrixXd Acl = -(hf * B1_ + hc * B2_) * (hf * F1_ + hc * F2_);
      Eigen::EigenSolver<Eigen::MatrixXd> es(Acl);
      for (int i = 0; i < es.eigenvalues().size(); i++)
        max_re = std::max(max_re, es.eigenvalues()(i).real());
    }
    if (max_re < 0)
      RCLCPP_INFO(get_logger(), "STABLE: max Re(lambda) = %.3e < 0", max_re);
    else
      RCLCPP_WARN(get_logger(), "UNSTABLE sweep: max Re(lambda) = %.3e >= 0", max_re);

    // The 3 continuous LMI conditions sym(P G) < 0.
    auto check = [&](const char* name, const Eigen::MatrixXd& G) {
      const Eigen::MatrixXd Q = P_ * G + G.transpose() * P_;
      const double me =
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Q).eigenvalues().maxCoeff();
      RCLCPP_INFO(get_logger(), "  %s: max_eig = %.3e %s", name, me,
                  me < 0 ? "< 0 OK" : ">= 0 FAIL");
    };
    check("sym(P G11)", -B1_ * F1_);
    check("sym(P G22)", -B2_ * F2_);
    check("sym(P Gcross)", -0.5 * (B1_ * F2_ + B2_ * F1_));
  }

  void publishDebug(const Eigen::VectorXd& e, const Eigen::VectorXd& x,
                    const Eigen::VectorXd& u, double hf, double hc, double Z, double V,
                    double Vd) {
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
      m.data = {Z,    hc,       hf,       u(0),     u(1),    u(2),   u(3), u(4),
                u(5), u.norm(), e.norm(), x.norm(), 1.0 / Z, Z_min_, Z_max_};
      ts_pub_->publish(m);
    }
    {
      std_msgs::msg::Float64MultiArray m;
      m.data = {V, Vd, x.norm(), (Vd < 0) ? 1.0 : 0.0};
      lyap_pub_->publish(m);
    }
  }

  void featureCb(const ibvs_msgs::msg::FeatureTarget::SharedPtr msg) {
    const double t = now().seconds();
    const double dt = (prev_time_ > 0) ? (t - prev_time_) : 0.033;

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
      if (!desired_sub_) s_star_.init(cube_w_, cube_h_, desired_Z_, fx_, fy_, cx_, cy_);
    }

    Eigen::VectorXd s_cur;
    std::vector<cv::Point2f> corners;
    unpackFeatures(*msg, s_cur, corners);
    const double Z_used = selectDepth(*msg, Z_min_, Z_max_);

    const Eigen::VectorXd e = s_cur - s_star_.s_star_flat;
    const Eigen::VectorXd x = L_hat_pinv_ * e;
    const double xn = x.norm();

    if (converged_ && prev_x_.norm() > 1e-6 &&
        (x - prev_x_).norm() / prev_x_.norm() > state_jump_ratio_) {
      commander_->publishZero();
      prev_time_ = t;
      return;
    }

    double hf, hc;
    tsMemberships(Z_used, Z_min_, Z_max_, hf, hc);
    Eigen::VectorXd u = -gain_scale_ * ((hf * F1_ + hc * F2_) * x);

    const double V = (x.transpose() * P_ * x)(0, 0);
    double Vd = 0.0;
    if (first_lyap_) {
      first_lyap_ = false;
    } else if (dt > 1e-6) {
      Vd = (V - V_prev_) / dt;
    }

    char hud[128];
    std::snprintf(hud, sizeof(hud), "Z:%.3f |x|:%.6f h=(%.2f,%.2f)", Z_used, xn, hf, hc);

    // Norm-based convergence with 3x hysteresis (continuous formulation).
    const double kHysteresis = 3.0;
    bool hold = false;
    if (converged_) {
      if (xn < kHysteresis * error_threshold_) {
        hold = true;
      } else {
        converged_ = false;
        conv_count_ = 0;
      }
    } else if (xn < error_threshold_) {
      if (++conv_count_ >= conv_frames_) {
        converged_ = true;
        hold = true;
        RCLCPP_INFO(get_logger(), "*** CONVERGED: ||x||=%.6f ***", xn);
      }
    } else {
      conv_count_ = 0;
    }

    if (hold) {
      commander_->publishZero();
      publishDebug(e, x, u, hf, hc, Z_used, V, Vd);
      viz_->publish(corners, s_star_.pixels, true, hud);
      finishCycle(t, x, V);
      return;
    }

    u *= edgeFactor(msg->centroid_u, msg->centroid_v, msg->image_width, msg->image_height,
                    edge_margin_);

    Eigen::Matrix<double, 6, 1> u_sat;
    for (int i = 0; i < 3; i++) u_sat(i) = clampAbs(u(i), max_lin_);
    for (int i = 3; i < 6; i++) u_sat(i) = clampAbs(u(i), max_ang_);
    commander_->publish(u_sat);

    publishDebug(e, x, u_sat, hf, hc, Z_used, V, Vd);
    viz_->publish(corners, s_star_.pixels, false, hud);
    recorder_->record(false, e.norm(), V, xn, Vd, Z_used, desired_Z_, u_sat(0), u_sat(1),
                      u_sat(2), u_sat(3), u_sat(4), u_sat(5), corners, s_star_.pixels, hf,
                      hc);
    finishCycle(t, x, V);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                         "Z=%.2f h=(%.2f,%.2f) ||x||=%.4f V=%.4f Vdot~%.4f", Z_used, hf, hc,
                         xn, V, Vd);
  }

  void finishCycle(double t, const Eigen::VectorXd& x, double V) {
    prev_time_ = t;
    prev_x_ = x;
    V_prev_ = V;
  }

  double cube_w_, cube_h_, desired_Z_, Z0_, Z_min_, Z_max_;
  double fx_, fy_, cx_, cy_;
  DesiredFeatures s_star_;
  Eigen::MatrixXd L_hat_, L_hat_pinv_, B1_, B2_, F1_, F2_, P_;

  double gain_scale_, max_lin_, max_ang_, edge_margin_, error_threshold_, state_jump_ratio_;
  int conv_frames_, max_lost_frames_, lost_count_ = 0, conv_count_ = 0;
  bool converged_ = false;

  Eigen::VectorXd prev_x_ = Eigen::VectorXd::Zero(6);
  double prev_time_ = 0.0, V_prev_ = 0.0;
  bool first_lyap_ = true;

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
  rclcpp::spin(std::make_shared<ibvs_control::TsIbvsContinuousNode>());
  rclcpp::shutdown();
  return 0;
}
