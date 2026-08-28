// Visual-only cable in Gazebo Harmonic, driven by SOFA centerline frames.
//
// Subscribes /cable/sofa_frames (PoseArray in base_link), transforms to the
// Gazebo world frame via TF, spawns one static cylinder per frame pair
// (no collision, no physics) and updates all poses with a single native
// gz-transport set_pose_vector request per cycle — no CLI subprocess, so
// updates keep up with the robot and the cable moves smoothly.

#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <gz/transport/Node.hh>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/entity_factory.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/entity.pb.h>

namespace
{

struct Vec3 { double x, y, z; };
struct Quat { double x, y, z, w; };

Quat quatZTo(const Vec3 & d)
{
  if (d.z > 0.999999) return {0, 0, 0, 1};
  if (d.z < -0.999999) return {1, 0, 0, 0};
  const double ax = -d.y, ay = d.x;  // z cross d
  const double w = 1.0 + d.z;
  const double n = std::sqrt(ax * ax + ay * ay + w * w);
  return {ax / n, ay / n, 0.0, w / n};
}

Vec3 rotate(const Quat & q, const Vec3 & v)
{
  // v' = v + 2 q_vec x (q_vec x v + w v)
  const double tx = 2.0 * (q.y * v.z - q.z * v.y);
  const double ty = 2.0 * (q.z * v.x - q.x * v.z);
  const double tz = 2.0 * (q.x * v.y - q.y * v.x);
  return {v.x + q.w * tx + (q.y * tz - q.z * ty),
          v.y + q.w * ty + (q.z * tx - q.x * tz),
          v.z + q.w * tz + (q.x * ty - q.y * tx)};
}

}  // namespace

class CableVisualNode : public rclcpp::Node
{
public:
  CableVisualNode()
  : Node("cable_visual_node")
  {
    world_ = declare_parameter<std::string>("world", "ibvs_world");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    radius_ = declare_parameter<double>("radius", 0.006);
    const double rate = declare_parameter<double>("update_rate_hz", 30.0);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "/cable/sofa_frames", rclcpp::QoS(1).best_effort(),
      [this](geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        latest_ = std::move(msg);
      });

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate), [this] { update(); });
  }

private:
  // ---- gz helpers --------------------------------------------------------
  template<typename ReqT>
  bool gzRequestBool(const std::string & service, const ReqT & req,
                     unsigned timeout_ms = 2000)
  {
    gz::msgs::Boolean rep;
    bool result = false;
    bool executed = gz_node_.Request(service, req, timeout_ms, rep, result);
    return executed && result && rep.data();
  }

  void removeStaleSegments()
  {
    // Clear leftovers from a previous run so respawn never collides
    for (int i = 0; i < 200; ++i) {
      gz::msgs::Entity req;
      req.set_name("cable_seg_" + std::to_string(i));
      req.set_type(gz::msgs::Entity::MODEL);
      gz::msgs::Boolean rep;
      bool result = false;
      if (!gz_node_.Request("/world/" + world_ + "/remove", req, 300, rep, result) ||
          !result || !rep.data()) {
        if (i > 60) break;  // beyond any plausible segment count
      }
    }
  }

  struct Segment { Vec3 mid; Quat q; double len; };

  std::vector<Segment> segments(const std::vector<Vec3> & pts) const
  {
    std::vector<Segment> out;
    out.reserve(pts.size());
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
      const Vec3 & a = pts[i];
      const Vec3 & b = pts[i + 1];
      const Vec3 d{b.x - a.x, b.y - a.y, b.z - a.z};
      const double ln = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
      if (ln < 1e-9) continue;
      out.push_back({{(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2},
                     quatZTo({d.x / ln, d.y / ln, d.z / ln}), ln});
    }
    return out;
  }

  bool spawn(const std::vector<Segment> & segs)
  {
    removeStaleSegments();
    bool ok_all = true;
    for (size_t i = 0; i < segs.size(); ++i) {
      std::ostringstream sdf;
      sdf << "<?xml version='1.0'?><sdf version='1.9'>"
          << "<model name='cable_seg_" << i << "'><static>true</static>"
          << "<link name='link'><visual name='v'><geometry><cylinder>"
          << "<radius>" << radius_ << "</radius>"
          << "<length>" << segs[i].len * 1.1 << "</length>"
          << "</cylinder></geometry><material>"
          << "<ambient>0.85 0.35 0.05 1</ambient>"
          << "<diffuse>0.85 0.35 0.05 1</diffuse>"
          << "</material></visual></link></model></sdf>";

      gz::msgs::EntityFactory req;
      req.set_sdf(sdf.str());
      req.set_name("cable_seg_" + std::to_string(i));
      req.set_allow_renaming(false);
      auto * p = req.mutable_pose();
      p->mutable_position()->set_x(segs[i].mid.x);
      p->mutable_position()->set_y(segs[i].mid.y);
      p->mutable_position()->set_z(segs[i].mid.z);
      p->mutable_orientation()->set_x(segs[i].q.x);
      p->mutable_orientation()->set_y(segs[i].q.y);
      p->mutable_orientation()->set_z(segs[i].q.z);
      p->mutable_orientation()->set_w(segs[i].q.w);
      ok_all &= gzRequestBool("/world/" + world_ + "/create", req);
    }
    return ok_all;
  }

  void update()
  {
    geometry_msgs::msg::PoseArray::SharedPtr msg;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      msg = latest_;
    }
    if (!msg || msg->poses.size() < 2) return;

    geometry_msgs::msg::TransformStamped tfm;
    try {
      tfm = tf_buffer_->lookupTransform(world_frame_, msg->header.frame_id,
                                        tf2::TimePointZero);
    } catch (const tf2::TransformException &) {
      return;
    }
    const Quat q{tfm.transform.rotation.x, tfm.transform.rotation.y,
                 tfm.transform.rotation.z, tfm.transform.rotation.w};
    const Vec3 t{tfm.transform.translation.x, tfm.transform.translation.y,
                 tfm.transform.translation.z};

    std::vector<Vec3> pts;
    pts.reserve(msg->poses.size());
    for (const auto & p : msg->poses) {
      const Vec3 v = rotate(q, {p.position.x, p.position.y, p.position.z});
      pts.push_back({v.x + t.x, v.y + t.y, v.z + t.z});
    }
    const auto segs = segments(pts);

    if (spawned_ == 0) {
      const bool ok = spawn(segs);
      spawned_ = static_cast<int>(segs.size());
      RCLCPP_INFO(get_logger(), "spawned %d cable segments in '%s' (ok=%d)",
                  spawned_, world_.c_str(), ok);
      return;
    }

    // Single batched pose update per cycle (native transport, fast)
    gz::msgs::Pose_V req;
    const size_t n = std::min<size_t>(segs.size(), spawned_);
    for (size_t i = 0; i < n; ++i) {
      auto * p = req.add_pose();
      p->set_name("cable_seg_" + std::to_string(i));
      p->mutable_position()->set_x(segs[i].mid.x);
      p->mutable_position()->set_y(segs[i].mid.y);
      p->mutable_position()->set_z(segs[i].mid.z);
      p->mutable_orientation()->set_x(segs[i].q.x);
      p->mutable_orientation()->set_y(segs[i].q.y);
      p->mutable_orientation()->set_z(segs[i].q.z);
      p->mutable_orientation()->set_w(segs[i].q.w);
    }
    gzRequestBool("/world/" + world_ + "/set_pose_vector", req);
  }

  std::string world_, world_frame_;
  double radius_{0.006};
  int spawned_{0};

  std::mutex mtx_;
  geometry_msgs::msg::PoseArray::SharedPtr latest_;

  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  gz::transport::Node gz_node_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CableVisualNode>());
  rclcpp::shutdown();
  return 0;
}
