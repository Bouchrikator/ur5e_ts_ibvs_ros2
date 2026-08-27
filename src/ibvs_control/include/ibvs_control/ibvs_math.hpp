// Shared IBVS math used by all controllers: point-feature interaction
// matrices (normalized and pixel space), TS membership functions, and the
// image-edge damping factor. Header-only so unit tests exercise the exact
// production code.
#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

namespace ibvs_control {

// Chaumette point-feature interaction matrix in normalized coordinates:
// s_dot = L(x, y, Z) * v_c, with s = (x, y).
inline Eigen::Matrix<double, 2, 6> pointInteractionMatrix(double x, double y, double Z) {
  Eigen::Matrix<double, 2, 6> L;
  L << -1.0 / Z, 0.0, x / Z, x * y, -(1 + x * x), y,
       0.0, -1.0 / Z, y / Z, 1 + y * y, -x * y, -x;
  return L;
}

// Stacked 8x6 interaction matrix for 4 points at a common depth Z.
// pts is 4x2 (rows: x_i, y_i in normalized coordinates).
inline Eigen::MatrixXd stackedInteractionMatrix(const Eigen::MatrixXd& pts, double Z) {
  Eigen::MatrixXd L_e(8, 6);
  for (int i = 0; i < 4; i++) {
    L_e.block<2, 6>(2 * i, 0) = pointInteractionMatrix(pts(i, 0), pts(i, 1), Z);
  }
  return L_e;
}

// Pixel-coordinate image Jacobian at pixel (u, v) with intrinsics
// (fx, fy, cx, cy) and depth Z, as used by the QMM-MPC (Wang et al. 2014).
// uc = u - cx, vc = v - cy.
inline Eigen::Matrix<double, 2, 6> pixelJacobian(double u, double v, double Z, double fx,
                                                 double fy, double cx, double cy) {
  const double uc = u - cx;
  const double vc = v - cy;
  Eigen::Matrix<double, 2, 6> J;
  J << -fx / Z, 0.0, uc / Z, uc * vc / fy, -(fx + uc * uc / fx), fx * vc / fy,
       0.0, -fy / Z, vc / Z, fy + vc * vc / fy, -uc * vc / fx, -fy * uc / fx;
  return J;
}

// TS membership functions on the scheduling variable a = 1/Z:
// h_far -> 1 as Z -> Z_max, h_close -> 1 as Z -> Z_min; convex sum by design.
inline void tsMemberships(double Z, double Z_min, double Z_max, double& h_far,
                          double& h_close) {
  const double a_min = 1.0 / Z_max;
  const double a_max = 1.0 / Z_min;
  const double a = 1.0 / std::clamp(Z, Z_min, Z_max);
  h_far = std::clamp((a_max - a) / (a_max - a_min), 0.0, 1.0);
  h_close = 1.0 - h_far;
}

// Damps the command when the target centroid approaches the image border so
// the target is not pushed out of the field of view. Full gain beyond
// edge_margin px from the border, linearly reduced to 0.3 at the border.
inline double edgeFactor(double u, double v, double width, double height,
                         double edge_margin) {
  const double kMinFactor = 0.3;
  const double md = std::min({u, width - u, v, height - v});
  if (md >= edge_margin) return 1.0;
  if (md <= 0) return kMinFactor;
  return kMinFactor + (1.0 - kMinFactor) * (md / edge_margin);
}

inline double clampAbs(double x, double lim) { return std::clamp(x, -lim, lim); }

// Spatial velocity-twist transform cVe: maps a twist expressed in the
// end-effector frame to the camera frame, given the pose (R, t) of the EE
// frame expressed in the camera frame (Chaumette & Hutchinson tutorial,
// Part II, eq. (6.9)):  cVe = [[R, skew(t) R], [0, R]].
// Used for eye-to-hand servoing where the control law yields a virtual
// camera twist that must be realized by the opposite end-effector motion.
inline Eigen::Matrix<double, 6, 6> velocityTwistMatrix(const Eigen::Matrix3d& R,
                                                       const Eigen::Vector3d& t) {
  Eigen::Matrix3d tx;
  tx << 0.0, -t.z(), t.y(),
        t.z(), 0.0, -t.x(),
        -t.y(), t.x(), 0.0;
  Eigen::Matrix<double, 6, 6> V = Eigen::Matrix<double, 6, 6>::Zero();
  V.topLeftCorner<3, 3>() = R;
  V.topRightCorner<3, 3>() = tx * R;
  V.bottomRightCorner<3, 3>() = R;
  return V;
}

}  // namespace ibvs_control
