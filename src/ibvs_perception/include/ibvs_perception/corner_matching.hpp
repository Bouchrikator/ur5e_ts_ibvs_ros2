// Corner extraction and temporal matching utilities, header-only so both the
// detector node and the unit tests use the exact same implementation.
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>

namespace ibvs_perception {

// Orders the 4 minAreaRect corners as TL, TR, BR, BL. A stable geometric
// ordering is required so the feature vector s = (x0,y0,...,x3,y3) always
// refers to the same physical corners on the first detection.
inline std::vector<cv::Point2f> orderCorners(const cv::Point2f* rect_corners) {
  std::vector<cv::Point2f> corners(rect_corners, rect_corners + 4);

  std::sort(corners.begin(), corners.end(),
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

  if (corners[0].x > corners[1].x) std::swap(corners[0], corners[1]);
  if (corners[2].x > corners[3].x) std::swap(corners[2], corners[3]);

  return {corners[0], corners[1], corners[3], corners[2]};
}

// Matches the current corners against the previous frame by searching the
// full D4 corner-order group (4 cyclic rotations x 2 windings).
// cv::minAreaRect can flip CW/CCW winding as the target rotates; trying only
// cyclic rotations lets the feature vector become mirrored, which makes the
// controller translate instead of rotating.
//
// When even the best permutation has a large cost (fast motion or a detection
// glitch), the best match is still returned rather than resetting the
// tracking: resetting disables the next frame's matching, permutes the
// feature vector, and spikes the IBVS error by hundreds of pixels (this made
// the QMM-MPC problem infeasible on the real robot).
inline std::vector<cv::Point2f> matchCornersTemporal(
    const std::vector<cv::Point2f>& current,
    const std::vector<cv::Point2f>& previous, double reset_cost,
    bool* large_jump = nullptr) {
  if (large_jump) *large_jump = false;
  if (previous.size() != 4 || current.size() != 4) return current;

  std::vector<cv::Point2f> best_match = current;
  float best_cost = std::numeric_limits<float>::max();

  for (int reflect = 0; reflect < 2; reflect++) {
    for (int rot = 0; rot < 4; rot++) {
      float cost = 0;
      for (int i = 0; i < 4; i++) {
        int idx = reflect ? ((4 - i + rot) % 4) : ((i + rot) % 4);
        float dx = current[idx].x - previous[i].x;
        float dy = current[idx].y - previous[i].y;
        cost += dx * dx + dy * dy;
      }
      if (cost < best_cost) {
        best_cost = cost;
        best_match.clear();
        for (int i = 0; i < 4; i++) {
          int idx = reflect ? ((4 - i + rot) % 4) : ((i + rot) % 4);
          best_match.push_back(current[idx]);
        }
      }
    }
  }

  if (large_jump && best_cost > reset_cost) *large_jump = true;
  return best_match;
}

}  // namespace ibvs_perception
