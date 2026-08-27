// Cube detector node — the single vision concern of the IBVS pipeline.
//
// Pipeline: BGR -> HSV -> red threshold (dual hue range) -> morphology ->
// largest contour -> minAreaRect -> corner ordering + temporal matching (D4)
// -> optional EMA smoothing -> PnP depth (ViSP DEMENTHON + VIRTUAL_VS) ->
// aligned depth-image sampling -> ibvs_msgs/FeatureTarget.
//
// Controllers never touch pixels: they consume FeatureTarget only.

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/core/vpPoint.h>
#include <visp3/vision/vpPose.h>

#include "ibvs_msgs/msg/feature_target.hpp"
#include "ibvs_perception/corner_matching.hpp"

namespace ibvs_perception {

class CubeDetectorNode : public rclcpp::Node {
 public:
  CubeDetectorNode() : Node("cube_detector") {
    // --- Topics ---
    image_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    depth_topic_ = declare_parameter<std::string>(
        "depth_topic", "/camera/aligned_depth_to_color/image_raw");
    camera_info_topic_ =
        declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");

    // --- Camera intrinsics fallback (overridden by CameraInfo when it arrives) ---
    px_ = declare_parameter<double>("cam_px", 554.254691191187);
    py_ = declare_parameter<double>("cam_py", 554.254691191187);
    u0_ = declare_parameter<double>("cam_u0", 320.5);
    v0_ = declare_parameter<double>("cam_v0", 240.5);
    cam_.initPersProjWithoutDistortion(px_, py_, u0_, v0_);

    // --- Target geometry (rectangular target supported; real cube 7.5 x 6.0 cm) ---
    cube_width_ = declare_parameter<double>("cube_size_x", 0.05);
    cube_height_ = declare_parameter<double>("cube_size_y", 0.05);

    // --- HSV red segmentation ---
    h_low1_ = declare_parameter<int>("h_low1", 0);
    h_high1_ = declare_parameter<int>("h_high1", 10);
    h_low2_ = declare_parameter<int>("h_low2", 170);
    // Red needs two hue ranges (hue wraps at 180); single-hue targets like
    // the green eye-to-hand marker disable the second range.
    use_hue_range2_ = declare_parameter<bool>("use_hue_range2", true);
    h_high2_ = declare_parameter<int>("h_high2", 179);
    s_min_ = declare_parameter<int>("s_min", 100);
    v_min_ = declare_parameter<int>("v_min", 100);
    min_area_ = declare_parameter<double>("min_area", 100.0);

    // --- Temporal matching / smoothing ---
    corner_reset_cost_ = declare_parameter<double>("corner_reset_cost", 40000.0);
    // 0.0 disables smoothing (TS-LMI / classic); QMM uses 0.85 on the real robot.
    corner_ema_alpha_ = declare_parameter<double>("corner_ema_alpha", 0.0);

    // --- Depth sampling ---
    use_depth_ = declare_parameter<bool>("use_depth_for_Z", true);
    // Small/far targets (eye-to-hand scene camera) make PnP depth unreliable;
    // the area-based estimate sqrt(w*h)*f/sqrt(area) is much more robust there.
    prefer_area_depth_ = declare_parameter<bool>("prefer_area_depth", false);
    depth_window_ = declare_parameter<int>("depth_window", 4);
    depth_min_ = declare_parameter<double>("depth_min", 0.10);
    depth_max_ = declare_parameter<double>("depth_max", 2.0);
    Z_min_ = declare_parameter<double>("Z_min", 0.10);
    Z_max_ = declare_parameter<double>("Z_max", 2.0);

    feature_pub_ = create_publisher<ibvs_msgs::msg::FeatureTarget>("~/feature_target", 10);
    viz_pub_ = create_publisher<sensor_msgs::msg::Image>("~/detection_image", 10);

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&CubeDetectorNode::imageCb, this, std::placeholders::_1));
    if (use_depth_) {
      depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
          depth_topic_, rclcpp::SensorDataQoS(),
          std::bind(&CubeDetectorNode::depthCb, this, std::placeholders::_1));
    }
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, rclcpp::SensorDataQoS(),
        std::bind(&CubeDetectorNode::cameraInfoCb, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "[CubeDetector] image=%s depth=%s cube=%.3fx%.3f m depth_for_Z=%d",
                image_topic_.c_str(), depth_topic_.c_str(), cube_width_, cube_height_,
                use_depth_);
  }

 private:
  void cameraInfoCb(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    // K = [fx 0 cx; 0 fy cy; 0 0 1]. Take the first valid message: real
    // cameras (RealSense) publish calibrated intrinsics that must override
    // the launch-file defaults.
    if (msg->k[0] <= 0.0) return;
    px_ = msg->k[0];
    py_ = msg->k[4];
    u0_ = msg->k[2];
    v0_ = msg->k[5];
    cam_.initPersProjWithoutDistortion(px_, py_, u0_, v0_);
    if (!have_camera_info_) {
      RCLCPP_INFO(get_logger(), "[CubeDetector] CameraInfo: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                  px_, py_, u0_, v0_);
    }
    have_camera_info_ = true;
  }

  void depthCb(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      cv_bridge::CvImageConstPtr cv = cv_bridge::toCvShare(msg);
      std::lock_guard<std::mutex> lk(depth_mutex_);
      latest_depth_ = cv->image.clone();
      latest_depth_encoding_ = msg->encoding;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "depth cv_bridge: %s", e.what());
    }
  }

  // Median depth over a (2*depth_window+1)^2 patch. Supports RealSense 16UC1
  // (millimetres) and 32FC1 (metres).
  bool sampleDepthAt(double u, double v, double& Z) {
    cv::Mat depth;
    std::string enc;
    {
      std::lock_guard<std::mutex> lk(depth_mutex_);
      if (latest_depth_.empty()) return false;
      depth = latest_depth_;
      enc = latest_depth_encoding_;
    }

    const int iu = static_cast<int>(std::round(u));
    const int iv = static_cast<int>(std::round(v));
    const int w = depth_window_;
    const int u0p = std::max(0, iu - w);
    const int u1p = std::min(depth.cols - 1, iu + w);
    const int v0p = std::max(0, iv - w);
    const int v1p = std::min(depth.rows - 1, iv + w);
    if (u0p > u1p || v0p > v1p) return false;

    const bool is_16u = (enc == sensor_msgs::image_encodings::TYPE_16UC1) ||
                        (enc == "16UC1") || (enc == sensor_msgs::image_encodings::MONO16);
    const bool is_32f =
        (enc == sensor_msgs::image_encodings::TYPE_32FC1) || (enc == "32FC1");
    if (!is_16u && !is_32f) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Unsupported depth encoding: %s", enc.c_str());
      return false;
    }

    std::vector<double> samples;
    samples.reserve((u1p - u0p + 1) * (v1p - v0p + 1));
    for (int y = v0p; y <= v1p; y++) {
      for (int x = u0p; x <= u1p; x++) {
        double d_m = 0.0;
        if (is_16u) {
          uint16_t d = depth.at<uint16_t>(y, x);
          if (d == 0) continue;
          d_m = d * 0.001;
        } else {
          float d = depth.at<float>(y, x);
          if (!std::isfinite(d) || d <= 0.0f) continue;
          d_m = static_cast<double>(d);
        }
        if (d_m < depth_min_ || d_m > depth_max_) continue;
        samples.push_back(d_m);
      }
    }
    if (samples.empty()) return false;
    std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());
    Z = samples[samples.size() / 2];
    return true;
  }

  // Median over centroid + 4 corners: target edges on the background give
  // reliable depth, unlike the saturated red centre.
  bool sampleTargetDepth(const std::vector<cv::Point2f>& corners, double& Z) {
    std::vector<double> zs;
    zs.reserve(5);
    cv::Point2f centroid(0.f, 0.f);
    for (const auto& p : corners) centroid += p;
    centroid *= 0.25f;

    double zc = 0.0;
    if (sampleDepthAt(centroid.x, centroid.y, zc)) zs.push_back(zc);
    for (const auto& p : corners) {
      double zi = 0.0;
      if (sampleDepthAt(p.x, p.y, zi)) zs.push_back(zi);
    }
    if (zs.empty()) return false;
    std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
    Z = zs[zs.size() / 2];
    return true;
  }

  bool computePoseFromCorners(const std::vector<cv::Point2f>& corners,
                              vpHomogeneousMatrix& cMo) {
    if (corners.size() != 4) return false;

    vpPose pose;
    const double half_w = cube_width_ / 2.0;
    const double half_h = cube_height_ / 2.0;
    vpPoint pts[4];
    pts[0].setWorldCoordinates(-half_w, -half_h, 0);
    pts[1].setWorldCoordinates(half_w, -half_h, 0);
    pts[2].setWorldCoordinates(half_w, half_h, 0);
    pts[3].setWorldCoordinates(-half_w, half_h, 0);

    for (int i = 0; i < 4; i++) {
      double x = 0.0, y = 0.0;
      vpPixelMeterConversion::convertPoint(cam_, corners[i].x, corners[i].y, x, y);
      pts[i].set_x(x);
      pts[i].set_y(y);
      pose.addPoint(pts[i]);
    }

    try {
      pose.computePose(vpPose::DEMENTHON, cMo);
      pose.computePose(vpPose::VIRTUAL_VS, cMo);
      return true;
    } catch (const vpException& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "vpPose failed: %s", e.what());
      return false;
    }
  }

  // For a rectangular target the projected pixel area scales as
  // (w*h)*f^2/Z^2, hence Z = sqrt(w*h)*f / sqrt(area).
  double estimateDepthFromArea(double area) const {
    const double f_avg = (px_ + py_) / 2.0;
    const double Z_est = std::sqrt(cube_width_ * cube_height_) * f_avg / std::sqrt(area);
    return std::max(Z_min_, std::min(Z_max_, Z_est));
  }

  void publishLost(const std_msgs::msg::Header& header, uint32_t w, uint32_t h) {
    ibvs_msgs::msg::FeatureTarget msg;
    msg.header = header;
    msg.detected = false;
    msg.image_width = w;
    msg.image_height = h;
    msg.fx = px_;
    msg.fy = py_;
    msg.cx = u0_;
    msg.cy = v0_;
    feature_pub_->publish(msg);
  }

  void imageCb(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv::Mat bgr;
    try {
      bgr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "cv_bridge: %s", e.what());
      return;
    }
    if (bgr.empty()) return;

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // Red wraps around hue 0, so two ranges are needed.
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(h_low1_, s_min_, v_min_), cv::Scalar(h_high1_, 255, 255),
                mask1);
    mask = mask1;
    if (use_hue_range2_) {
      cv::inRange(hsv, cv::Scalar(h_low2_, s_min_, v_min_), cv::Scalar(h_high2_, 255, 255),
                  mask2);
      mask = mask1 | mask2;
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    size_t best = 0;
    double best_area = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
      const double a = std::fabs(cv::contourArea(contours[i]));
      if (a > best_area) {
        best_area = a;
        best = i;
      }
    }

    if (contours.empty() || best_area < min_area_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "No red region (best area=%.1f)", best_area);
      publishLost(msg->header, bgr.cols, bgr.rows);
      publishViz(msg->header, bgr, {}, false);
      return;
    }

    std::vector<cv::Point2f> contour_f(contours[best].begin(), contours[best].end());
    cv::RotatedRect rect = cv::minAreaRect(contour_f);
    cv::Point2f rect_corners[4];
    rect.points(rect_corners);

    std::vector<cv::Point2f> corners = orderCorners(rect_corners);
    if (!first_detection_) {
      bool large_jump = false;
      corners = matchCornersTemporal(corners, prev_corners_, corner_reset_cost_, &large_jump);
      if (large_jump) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Large corner motion; keeping best D4 match");
      }
      // EMA smoothing rejects single-frame pixel noise on the real camera.
      if (corner_ema_alpha_ > 0.0) {
        const float a = static_cast<float>(corner_ema_alpha_);
        for (int i = 0; i < 4; i++) {
          corners[i] = a * prev_corners_[i] + (1.0f - a) * corners[i];
        }
      }
    }
    prev_corners_ = corners;
    first_detection_ = false;

    // Depth from geometry (PnP) with area-based fallback; prefer_area_depth
    // skips PnP depth entirely (eye-to-hand: tiny targets, unreliable PnP).
    vpHomogeneousMatrix cMo;
    const bool pose_valid = computePoseFromCorners(corners, cMo);
    const double z_pnp = (pose_valid && !prefer_area_depth_)
                             ? cMo[2][3]
                             : estimateDepthFromArea(best_area);

    double z_depth = 0.0;
    const bool depth_valid = use_depth_ && sampleTargetDepth(corners, z_depth);

    const cv::Moments m = cv::moments(contours[best]);

    ibvs_msgs::msg::FeatureTarget out;
    out.header = msg->header;
    out.detected = true;
    out.image_width = bgr.cols;
    out.image_height = bgr.rows;
    for (int i = 0; i < 4; i++) {
      out.corner_pixels[2 * i] = corners[i].x;
      out.corner_pixels[2 * i + 1] = corners[i].y;
      double x = 0.0, y = 0.0;
      vpPixelMeterConversion::convertPoint(cam_, corners[i].x, corners[i].y, x, y);
      out.features_normalized[2 * i] = x;
      out.features_normalized[2 * i + 1] = y;
    }
    out.centroid_u = m.m10 / m.m00;
    out.centroid_v = m.m01 / m.m00;
    out.area = best_area;
    out.pose_valid = pose_valid;
    out.z_pnp = z_pnp;
    out.depth_valid = depth_valid;
    out.z_depth = z_depth;
    out.fx = px_;
    out.fy = py_;
    out.cx = u0_;
    out.cy = v0_;
    feature_pub_->publish(out);

    publishViz(msg->header, bgr, corners, depth_valid ? z_depth : z_pnp, true);
  }

  void publishViz(const std_msgs::msg::Header& header, const cv::Mat& bgr,
                  const std::vector<cv::Point2f>& corners, double Z, bool detected = true) {
    if (viz_pub_->get_subscription_count() == 0) return;

    cv::Mat viz = bgr.clone();
    static const cv::Scalar kCornerColors[4] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {0, 255, 255}};
    if (detected && corners.size() == 4) {
      for (int i = 0; i < 4; i++) {
        cv::circle(viz, corners[i], 6, kCornerColors[i], 2);
        cv::line(viz, corners[i], corners[(i + 1) % 4], {255, 255, 255}, 1);
        cv::putText(viz, std::to_string(i), corners[i] + cv::Point2f(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, kCornerColors[i], 1);
      }
      char buf[64];
      std::snprintf(buf, sizeof(buf), "Z=%.3f m", Z);
      cv::putText(viz, buf, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
    } else {
      cv::putText(viz, "TARGET LOST", {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 255},
                  2);
    }
    viz_pub_->publish(*cv_bridge::CvImage(header, "bgr8", viz).toImageMsg());
  }

  // Overload used on the lost path (no Z available).
  void publishViz(const std_msgs::msg::Header& header, const cv::Mat& bgr,
                  const std::vector<cv::Point2f>& corners, bool detected) {
    publishViz(header, bgr, corners, 0.0, detected);
  }

  // Topics / intrinsics
  std::string image_topic_, depth_topic_, camera_info_topic_;
  double px_, py_, u0_, v0_;
  bool have_camera_info_ = false;
  vpCameraParameters cam_;

  // Target geometry
  double cube_width_, cube_height_;

  // Segmentation
  int h_low1_, h_high1_, h_low2_, h_high2_, s_min_, v_min_;
  bool use_hue_range2_;
  double min_area_;

  // Temporal matching
  double corner_reset_cost_, corner_ema_alpha_;
  std::vector<cv::Point2f> prev_corners_;
  bool first_detection_ = true;

  // Depth
  bool use_depth_;
  bool prefer_area_depth_;
  int depth_window_;
  double depth_min_, depth_max_, Z_min_, Z_max_;
  std::mutex depth_mutex_;
  cv::Mat latest_depth_;
  std::string latest_depth_encoding_;

  rclcpp::Publisher<ibvs_msgs::msg::FeatureTarget>::SharedPtr feature_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr viz_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_, depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
};

}  // namespace ibvs_perception

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ibvs_perception::CubeDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
