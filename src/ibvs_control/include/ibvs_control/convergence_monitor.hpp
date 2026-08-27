// Pixel-error convergence monitor with hysteresis + dwell-time re-arm,
// shared by all controllers. Residual set Omega = {e : max|e_axis| < thresh}.
// Once converged, the controller only re-arms when the error leaves the
// dilated set (rearm_ratio * thresh) for rearm_frames consecutive frames —
// standard switched-systems hysteresis that prevents twist chattering.
#pragma once

namespace ibvs_control {

class ConvergenceMonitor {
 public:
  ConvergenceMonitor(double threshold_px, int convergence_frames, double rearm_ratio = 2.0,
                     int rearm_frames = 5)
      : thresh_(threshold_px), conv_frames_(convergence_frames), rearm_ratio_(rearm_ratio),
        rearm_frames_(rearm_frames) {}

  // Feed the current worst per-axis pixel error; returns true when the
  // controller should hold (publish zero twist).
  bool update(double max_px_err) {
    if (converged_) {
      if (max_px_err > rearm_ratio_ * thresh_) {
        if (++rearm_count_ >= rearm_frames_) {
          converged_ = false;
          conv_count_ = 0;
          rearm_count_ = 0;
          just_rearmed_ = true;
          return false;
        }
      } else {
        rearm_count_ = 0;
      }
      return true;
    }
    just_rearmed_ = false;
    if (max_px_err < thresh_) {
      if (++conv_count_ >= conv_frames_) {
        converged_ = true;
        just_converged_ = true;
        return true;
      }
    } else {
      conv_count_ = 0;
    }
    just_converged_ = false;
    return false;
  }

  bool converged() const { return converged_; }
  // One-shot flags for logging.
  bool justConverged() { bool f = just_converged_; just_converged_ = false; return f; }
  bool justRearmed() { bool f = just_rearmed_; just_rearmed_ = false; return f; }

 private:
  double thresh_;
  int conv_frames_;
  double rearm_ratio_;
  int rearm_frames_;
  bool converged_ = false;
  int conv_count_ = 0;
  int rearm_count_ = 0;
  bool just_converged_ = false;
  bool just_rearmed_ = false;
};

}  // namespace ibvs_control
