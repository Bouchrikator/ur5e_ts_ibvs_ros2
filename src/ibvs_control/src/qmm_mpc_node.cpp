// Quasi-Min-Max MPC IBVS controller (Wang et al. 2014).
//
// Architecture: vision arrives asynchronously from the perception node; a
// control timer at Ts consumes the latest features, builds per-point pixel
// errors and Jacobians, and delegates the SDP (Eqs.17-23) to the Python
// solver node via the SolveMPC service. A fast timer republishes the last
// command between solves so MoveIt Servo never starves.
//
// Fair-comparison choices kept from the ROS 1 study:
//  - Jacobians evaluated at the DESIRED pixels with the MEASURED depth
//  - fixed Ts in the MPC prediction model (Theorem 1 requirement)
//  - EMA-filtered depth; corner EMA is done in the perception node

#include <atomic>
#include <cmath>
#include <mutex>
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
#include "ibvs_msgs/srv/solve_mpc.hpp"

namespace ibvs_control {

class QmmMpcNode : public rclcpp::Node {
 public:
  QmmMpcNode() : Node("qmm_ibvs_mpc") {
    cube_w_ = declare_parameter<double>("cube_size_x", 0.075);
    cube_h_ = declare_parameter<double>("cube_size_y", 0.060);
    desired_Z_ = declare_parameter<double>("desired_Z", 0.4);
    Z_min_ = declare_parameter<double>("Z_min", 0.2);
    Z_max_ = declare_parameter<double>("Z_max", 1.0);

    fx_ = declare_parameter<double>("cam_px", 554.254691191187);
    fy_ = declare_parameter<double>("cam_py", 554.254691191187);
    cx_ = declare_parameter<double>("cam_u0", 320.5);
    cy_ = declare_parameter<double>("cam_v0", 240.5);

    Ts_ = declare_parameter<double>("Ts", 0.04);
    v_max_linear_ = declare_parameter<double>("v_max_linear", 0.25);
    v_max_angular_ = declare_parameter<double>("v_max_angular", 0.25);
    v_max_lateral_ = declare_parameter<double>("v_max_lateral", 0.25);
    v_max_depth_ = declare_parameter<double>("v_max_depth", 0.25);

    u_min_vis_ = declare_parameter<double>("u_min_vis", 20.0);
    u_max_vis_ = declare_parameter<double>("u_max_vis", 620.0);
    v_min_vis_ = declare_parameter<double>("v_min_vis", 20.0);
    v_max_vis_ = declare_parameter<double>("v_max_vis", 460.0);

    // Command blending keeps the twist smooth despite solver-rate jitter.
    command_filter_alpha_ = declare_parameter<double>("command_filter_alpha", 0.45);
    stale_timeout_ = declare_parameter<double>("stale_command_timeout", 0.35);
    use_velocity_decay_ = declare_parameter<bool>("use_velocity_decay", false);
    velocity_decay_tau_ = declare_parameter<double>("velocity_decay_tau", 0.2);
    depth_ema_alpha_ = declare_parameter<double>("depth_ema_alpha", 0.7);
    fast_rate_ = declare_parameter<double>("fast_republish_rate", 30.0);
    edge_margin_ = declare_parameter<double>("edge_margin", 50.0);

    conv_thresh_px_ = declare_parameter<double>("convergence_threshold_px", 2.0);
    const int conv_frames = declare_parameter<int>("convergence_frames", 3);
    monitor_ = std::make_unique<ConvergenceMonitor>(conv_thresh_px_, conv_frames,
                                                    /*rearm_ratio=*/3.0);

    s_star_.init(cube_w_, cube_h_, desired_Z_, fx_, fy_, cx_, cy_);

    commander_ = std::make_unique<TwistCommander>(this);
    recorder_ = std::make_unique<DataRecorder>(this, "qmm_mpc");
    viz_ = std::make_unique<VizOverlay>(this);

    error_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/error", 10);
    info_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/mpc_info", 10);
    lyap_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/lyapunov", 10);

    feature_sub_ = create_subscription<ibvs_msgs::msg::FeatureTarget>(
        "/cube_detector/feature_target", rclcpp::SensorDataQoS(),
        std::bind(&QmmMpcNode::featureCb, this, std::placeholders::_1));
    // Eye-to-hand desired features: the subscription lives in the default
    // (mutually exclusive) callback group, so s_star_ writes are serialized
    // with controlTimerCb even under the MultiThreadedExecutor.
    desired_sub_ = subscribeDynamicDesired(this, s_star_, desired_received_);

    // The service client lives in its own callback group so a response can
    // be processed while the timers run (MultiThreadedExecutor).
    client_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    solver_client_ = create_client<ibvs_msgs::srv::SolveMPC>(
        "/qmm_mpc_solver_node/solve_mpc", rclcpp::ServicesQoS(), client_group_);

    control_timer_ = create_wall_timer(std::chrono::duration<double>(Ts_),
                                       std::bind(&QmmMpcNode::controlTimerCb, this));
    fast_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / fast_rate_),
                                    std::bind(&QmmMpcNode::fastTimerCb, this));

    RCLCPP_INFO(get_logger(),
                "QMM-MPC: Ts=%.3f cube=%.3fx%.3f Z*=%.2f vis=[%g,%g]x[%g,%g] conv=%.1fpx",
                Ts_, cube_w_, cube_h_, desired_Z_, u_min_vis_, u_max_vis_, v_min_vis_,
                v_max_vis_, conv_thresh_px_);
  }

 private:
  void featureCb(const ibvs_msgs::msg::FeatureTarget::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(vision_mutex_);
    latest_ = msg;
    if (msg->detected) {
      const double Z_raw = msg->depth_valid ? msg->z_depth : msg->z_pnp;
      // EMA depth: rejects single-frame depth noise on the real D435.
      Z_ema_ = first_depth_ ? Z_raw : depth_ema_alpha_ * Z_ema_ + (1.0 - depth_ema_alpha_) * Z_raw;
      first_depth_ = false;
      has_new_vision_ = true;
    }
  }

  void controlTimerCb() {
    // Skip if the previous SDP is still in flight (control_busy_ guard).
    if (busy_.exchange(true)) return;

    ibvs_msgs::msg::FeatureTarget::SharedPtr msg;
    double Z_used;
    {
      std::lock_guard<std::mutex> lk(vision_mutex_);
      if (!latest_ || !latest_->detected || !has_new_vision_) {
        busy_ = false;
        return;
      }
      msg = latest_;
      has_new_vision_ = false;
      Z_used = std::clamp(Z_ema_, Z_min_, Z_max_);
    }

    if (!solver_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "SolveMPC service not ready");
      busy_ = false;
      return;
    }

    if (msg->fx > 0 && std::fabs(msg->fx - fx_) > 1e-9) {
      fx_ = msg->fx;
      fy_ = msg->fy;
      cx_ = msg->cx;
      cy_ = msg->cy;
      if (!desired_sub_) s_star_.init(cube_w_, cube_h_, desired_Z_, fx_, fy_, cx_, cy_);
    }

    // Eye-to-hand: hold still until the reference generator has seen the cube.
    if (desired_sub_ && !desired_received_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Waiting for desired features (eye-to-hand)");
      commander_->publishZero();
      busy_ = false;
      return;
    }

    std::vector<cv::Point2f> corners(4);
    for (int i = 0; i < 4; i++)
      corners[i] = cv::Point2f(msg->corner_pixels[2 * i], msg->corner_pixels[2 * i + 1]);

    // Per-point pixel errors and worst per-axis error for convergence.
    Eigen::VectorXd e_px(8);
    double max_px_err = 0.0;
    for (int i = 0; i < 4; i++) {
      e_px(2 * i) = corners[i].x - s_star_.pixels[i].x;
      e_px(2 * i + 1) = corners[i].y - s_star_.pixels[i].y;
      max_px_err = std::max({max_px_err, std::fabs(e_px(2 * i)), std::fabs(e_px(2 * i + 1))});
    }

    const bool hold = monitor_->update(max_px_err);
    if (monitor_->justConverged())
      RCLCPP_INFO(get_logger(), "*** CONVERGED: max|e|=%.2f < %.2f px ***", max_px_err,
                  conv_thresh_px_);
    if (hold) {
      {
        std::lock_guard<std::mutex> lk(cmd_mutex_);
        last_vc_.setZero();
        last_cmd_time_ = now();
      }
      commander_->publishZero();
      publishDebug(e_px, Eigen::Matrix<double, 6, 1>::Zero(), Z_used, 0.0, true, 0.0, 0.0);
      viz_->publish(corners, s_star_.pixels, true, "CONVERGED (holding)");
      busy_ = false;
      return;
    }

    // Infeasibility diagnostic: warn when corners leave the visibility window.
    for (int i = 0; i < 4; i++) {
      if (corners[i].x < u_min_vis_ || corners[i].x > u_max_vis_ ||
          corners[i].y < v_min_vis_ || corners[i].y > v_max_vis_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Corner %d (%.0f,%.0f) outside visibility window", i,
                             corners[i].x, corners[i].y);
      }
    }

    auto req = std::make_shared<ibvs_msgs::srv::SolveMPC::Request>();
    req->errors_flat.assign(e_px.data(), e_px.data() + 8);
    req->jacobians_flat.reserve(48);
    for (int j = 0; j < 4; j++) {
      // Jacobian at desired pixels with measured Z (fair-comparison choice).
      const auto J = pixelJacobian(s_star_.pixels[j].x, s_star_.pixels[j].y, Z_used, fx_,
                                   fy_, cx_, cy_);
      for (int r = 0; r < 2; r++)
        for (int c = 0; c < 6; c++) req->jacobians_flat.push_back(J(r, c));
    }
    req->ts = Ts_;
    req->v_max_linear = v_max_linear_;
    req->v_max_angular = v_max_angular_;
    req->v_max_lateral = v_max_lateral_;
    req->v_max_depth = v_max_depth_;
    req->u_min = u_min_vis_;
    req->u_max = u_max_vis_;
    req->v_min = v_min_vis_;
    req->v_max = v_max_vis_;
    req->desired_positions_flat.reserve(8);
    for (int i = 0; i < 4; i++) {
      req->desired_positions_flat.push_back(s_star_.pixels[i].x);
      req->desired_positions_flat.push_back(s_star_.pixels[i].y);
    }

    const double edge = edgeFactor(msg->centroid_u, msg->centroid_v, msg->image_width,
                                   msg->image_height, edge_margin_);
    const Eigen::VectorXd e_for_cb = e_px;
    auto corners_cb = corners;
    // Snapshot the desired pixels: with a dynamic s* (eye-to-hand) the
    // response callback runs on another thread and must not read s_star_.
    const std::vector<cv::Point2f> des_px_cb(s_star_.pixels, s_star_.pixels + 4);
    solver_client_->async_send_request(
        req, [this, e_for_cb, corners_cb, des_px_cb, Z_used,
              edge](rclcpp::Client<ibvs_msgs::srv::SolveMPC>::SharedFuture future) {
          onSolverResponse(future.get(), e_for_cb, corners_cb, des_px_cb, Z_used, edge);
          busy_ = false;
        });
  }

  void onSolverResponse(const ibvs_msgs::srv::SolveMPC::Response::SharedPtr resp,
                        const Eigen::VectorXd& e_px, const std::vector<cv::Point2f>& corners,
                        const std::vector<cv::Point2f>& des_px, double Z_used, double edge) {
    Eigen::Matrix<double, 6, 1> vc = Eigen::Matrix<double, 6, 1>::Zero();
    if (resp->feasible && resp->vc.size() == 6) {
      for (int i = 0; i < 6; i++) vc(i) = resp->vc[i];
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "SDP infeasible (%s)",
                           resp->solver_status.c_str());
    }

    vc *= edge;

    Eigen::Matrix<double, 6, 1> cmd;
    {
      std::lock_guard<std::mutex> lk(cmd_mutex_);
      // Low-pass blend with the previous command, then client-side clamp as a
      // safety net (the SDP shapes velocity via Rw, not hard limits).
      cmd = command_filter_alpha_ * last_vc_ + (1.0 - command_filter_alpha_) * vc;
      cmd(0) = clampAbs(cmd(0), v_max_lateral_);
      cmd(1) = clampAbs(cmd(1), v_max_lateral_);
      cmd(2) = clampAbs(cmd(2), v_max_depth_);
      for (int i = 3; i < 6; i++) cmd(i) = clampAbs(cmd(i), v_max_angular_);
      last_vc_ = cmd;
      last_cmd_time_ = now();
    }
    commander_->publish(cmd);

    publishDebug(e_px, cmd, Z_used, resp->gamma, resp->feasible, resp->solver_time_ms,
                 resp->lyapunov_v);

    char hud[128];
    std::snprintf(hud, sizeof(hud), "Z:%.3f gamma:%.1f solve:%.0fms %s", Z_used, resp->gamma,
                  resp->solver_time_ms, resp->feasible ? "OK" : "INFEASIBLE");
    viz_->publish(corners, des_px.data(), false, hud);

    double max_e = 0.0;
    for (int i = 0; i < 8; i++) max_e = std::max(max_e, std::fabs(e_px(i)));
    recorder_->record(false, e_px.norm(), resp->lyapunov_v, e_px.norm(), 0.0, Z_used,
                      desired_Z_, cmd(0), cmd(1), cmd(2), cmd(3), cmd(4), cmd(5), corners,
                      des_px.data(), 0.0, 0.0, resp->solver_time_ms, resp->feasible);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                         "||e||=%.1fpx max=%.1fpx gamma=%.2f solve=%.0fms vc=(%.3f,%.3f,%.3f)",
                         e_px.norm(), max_e, resp->gamma, resp->solver_time_ms, cmd(0),
                         cmd(1), cmd(2));
  }

  // Republish between solves so Servo's command stream never goes stale.
  void fastTimerCb() {
    Eigen::Matrix<double, 6, 1> cmd;
    {
      std::lock_guard<std::mutex> lk(cmd_mutex_);
      const double age = (now() - last_cmd_time_).seconds();
      if (age > stale_timeout_) {
        last_vc_.setZero();
        cmd.setZero();
      } else if (use_velocity_decay_) {
        cmd = last_vc_ * std::exp(-age / velocity_decay_tau_);
      } else {
        cmd = last_vc_;
      }
    }
    commander_->publish(cmd);
  }

  void publishDebug(const Eigen::VectorXd& e_px, const Eigen::Matrix<double, 6, 1>& vc,
                    double Z, double gamma, bool feasible, double solve_ms, double V) {
    {
      std_msgs::msg::Float64MultiArray m;
      m.data.assign(e_px.data(), e_px.data() + 8);
      error_pub_->publish(m);
    }
    {
      double max_e = 0.0;
      for (int i = 0; i < 8; i++) max_e = std::max(max_e, std::fabs(e_px(i)));
      std_msgs::msg::Float64MultiArray m;
      m.data = {Z,     gamma,      feasible ? 1.0 : 0.0, vc(0),   vc(1),
                vc(2), vc(3),      vc(4),                vc(5),   vc.norm(),
                e_px.norm(), max_e, solve_ms,            Z_min_,  Z_max_};
      info_pub_->publish(m);
    }
    {
      std_msgs::msg::Float64MultiArray m;
      const double dV = first_V_ ? 0.0 : V - V_prev_;
      first_V_ = false;
      V_prev_ = V;
      m.data = {V, dV, e_px.norm(), (dV < 0) ? 1.0 : 0.0};
      lyap_pub_->publish(m);
    }
  }

  // Geometry / camera
  double cube_w_, cube_h_, desired_Z_, Z_min_, Z_max_;
  double fx_, fy_, cx_, cy_;
  DesiredFeatures s_star_;

  // MPC parameters
  double Ts_, v_max_linear_, v_max_angular_, v_max_lateral_, v_max_depth_;
  double u_min_vis_, u_max_vis_, v_min_vis_, v_max_vis_;
  double command_filter_alpha_, stale_timeout_, velocity_decay_tau_, depth_ema_alpha_;
  bool use_velocity_decay_;
  double fast_rate_, edge_margin_, conv_thresh_px_;

  // Vision state
  std::mutex vision_mutex_;
  ibvs_msgs::msg::FeatureTarget::SharedPtr latest_;
  bool has_new_vision_ = false, first_depth_ = true;
  double Z_ema_ = 0.0;

  // Command state
  std::mutex cmd_mutex_;
  Eigen::Matrix<double, 6, 1> last_vc_ = Eigen::Matrix<double, 6, 1>::Zero();
  rclcpp::Time last_cmd_time_{0, 0, RCL_ROS_TIME};
  std::atomic<bool> busy_{false};
  double V_prev_ = 0.0;
  bool first_V_ = true;

  std::unique_ptr<ConvergenceMonitor> monitor_;
  std::unique_ptr<TwistCommander> commander_;
  std::unique_ptr<DataRecorder> recorder_;
  std::unique_ptr<VizOverlay> viz_;

  rclcpp::Subscription<ibvs_msgs::msg::FeatureTarget>::SharedPtr feature_sub_;
  rclcpp::Subscription<ibvs_msgs::msg::FeatureTarget>::SharedPtr desired_sub_;
  bool desired_received_ = false;
  rclcpp::Client<ibvs_msgs::srv::SolveMPC>::SharedPtr solver_client_;
  rclcpp::CallbackGroup::SharedPtr client_group_;
  rclcpp::TimerBase::SharedPtr control_timer_, fast_timer_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr error_pub_, info_pub_,
      lyap_pub_;
};

}  // namespace ibvs_control

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ibvs_control::QmmMpcNode>();
  // Multi-threaded: service responses must be processed while timers run.
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 3);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
