"""Outer cable TS-PDC controller.

Step 9 of the plan. Regulates the cable SHAPE and emits a desired gripper
motion; it never talks to Servo directly — the inner eye-to-hand TS-IBVS loop
remains the single publisher of robot twists.

Runs at a fixed rate on the latest reduced state. Parameter estimates from the
identification pipeline arrive asynchronously and much slower; they are only
used to keep the operating point honest, and a stale or diverging estimate is
rejected rather than waited for.
"""

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64MultiArray

from cable_msgs.msg import CableParameterEstimate, CableShapeTarget, CableState

from cable_ts_control.modal_contract import (
    COORDINATES, LEGACY_COORDINATES, check_model_contract, target_state)
from cable_ts_control.ts_model import CableTsModel


class CableTsControllerNode(Node):
    def __init__(self):
        super().__init__("cable_ts_controller_node")

        self.declare_parameter("ts_model_file", "")
        self.declare_parameter("ts_gains_file", "")
        self.declare_parameter("state_topic", "/cable/reduced_state")
        self.declare_parameter("target_topic", "/cable/target_shape")
        self.declare_parameter("parameter_topic", "/cable/parameter_estimate")
        # 0 = run at exactly 1/Ts of the identified model. The gains are
        # certified for that discrete period only; a different rate applies
        # them to a plant they were never synthesised for.
        self.declare_parameter("control_rate_hz", 0.0)
        self.declare_parameter("max_linear_vel", 0.05)
        self.declare_parameter("state_timeout_s", 0.5)
        self.declare_parameter("estimate_timeout_s", 5.0)
        self.declare_parameter("max_parameter_std_dev", 0.01)
        self.declare_parameter("max_innovation", 0.05)
        self.declare_parameter("reference_frame", "cable_fixture_frame")
        self.declare_parameter("gripper_frame", "cable_grasp_frame")
        # How far ahead the commanded velocity is projected to build the pose
        # reference handed to the inner visual loop.
        self.declare_parameter("lookahead_s", 0.3)
        # Manifold anchor (audit 9.1): perception bias in the weak modes makes
        # the no-g-feedback law walk the gripper out of the premise box. The
        # anchor low-passes the residual between the measured z-error and the
        # certified quasi-static prediction U'J g and shifts the EFFECTIVE
        # target by it, so at rest u -> 0 exactly and the gripper stays put.
        self.declare_parameter("bias_anchor_alpha", 0.05)
        self.declare_parameter("bias_anchor_max_m", 0.05)
        self.declare_parameter("enable_bias_anchor", True)
        # Coordinates the reduced-state publisher uses; a model in other
        # coordinates (or on another strain basis) is refused at startup.
        self.declare_parameter("state_coordinates", LEGACY_COORDINATES)
        self.declare_parameter("strain_modes_metadata", "")

        model_file = self.get_parameter("ts_model_file").value
        if not model_file:
            raise ValueError("ts_model_file is required")
        gains_file = self.get_parameter("ts_gains_file").value or None
        self.coordinates = self._check_contract(model_file, gains_file)
        self.model = CableTsModel.load(model_file, gains_file)
        if self.model.gains is None:
            raise ValueError("the TS model carries no PDC gains; run the LMI solver")

        self.max_linear = float(self.get_parameter("max_linear_vel").value)
        self.state_timeout = float(self.get_parameter("state_timeout_s").value)
        self.estimate_timeout = float(self.get_parameter("estimate_timeout_s").value)
        self.max_std_dev = float(self.get_parameter("max_parameter_std_dev").value)
        self.max_innovation = float(self.get_parameter("max_innovation").value)
        self.reference_frame = self.get_parameter("reference_frame").value
        self.gripper_frame = self.get_parameter("gripper_frame").value
        self.lookahead = float(self.get_parameter("lookahead_s").value)
        self._init_bias_anchor(self.get_parameter("ts_gains_file").value or None)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._state = None
        self._state_stamp = None
        self._target = np.zeros(self.model.state_dim)
        # NOTE: must not be named _parameters -- that clobbers rclpy's
        # internal Node._parameters dict and undeclares every ROS parameter.
        self._estimates = {}
        self._parameter_stamp = None

        self.twist_pub = self.create_publisher(
            TwistStamped, "/cable/desired_gripper_twist", 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, "/cable/desired_gripper_pose", 10)
        self.error_pub = self.create_publisher(
            Float64MultiArray, "/cable/shape_error", 10)
        self.weights_pub = self.create_publisher(
            Float64MultiArray, "/cable/ts_weights", 10)
        self.lyapunov_pub = self.create_publisher(
            Float64MultiArray, "/cable/lyapunov", 10)

        self.create_subscription(
            CableState, self.get_parameter("state_topic").value, self._on_state, 10)
        self.create_subscription(
            CableShapeTarget, self.get_parameter("target_topic").value,
            self._on_target, 10)
        self.create_subscription(
            CableParameterEstimate, self.get_parameter("parameter_topic").value,
            self._on_parameter, 10)

        rate = float(self.get_parameter("control_rate_hz").value)
        model_rate = 1.0 / float(self.model.sample_time)
        if rate <= 0.0:
            rate = model_rate
        elif abs(rate - model_rate) > 0.01 * model_rate:
            self.get_logger().warn(
                f"control_rate_hz={rate:.2f} but the gains were certified at "
                f"Ts={self.model.sample_time:.4f} s ({model_rate:.2f} Hz); "
                f"the discrete certificate does not transfer across rates")
        self.timer = self.create_timer(1.0 / rate, self._control_cycle)

        self._previous_lyapunov = None
        self.get_logger().info(
            f"cable TS-PDC: {self.model.n_rules} rules, state dim "
            f"{self.model.state_dim}, Ts={self.model.sample_time:.3f} s, "
            f"rho bounds={self.model.premise_bounds}")

    # ------------------------------------------------------------------
    def _check_contract(self, model_file, gains_file):
        with open(model_file, "r") as f:
            section = yaml.safe_load(f)["cable_ts_model"]
        gains = None
        if gains_file:
            with open(gains_file, "r") as f:
                gains = yaml.safe_load(f)["cable_ts_model"]
        metadata = None
        metadata_file = self.get_parameter("strain_modes_metadata").value
        if metadata_file:
            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f)
        expected = self.get_parameter("state_coordinates").value
        if expected == COORDINATES and metadata is None:
            raise ValueError("cosserat_modal control needs strain_modes_metadata")
        return check_model_contract(section, metadata, gains, expected_coordinates=expected)

    def _init_bias_anchor(self, gains_file):
        """Load U and the mean certified Jacobian from the quasi-static gains."""
        self._anchor_u = None
        self._bias = None
        self._anchor_alpha = float(self.get_parameter("bias_anchor_alpha").value)
        self._anchor_max = float(self.get_parameter("bias_anchor_max_m").value)
        if not self.get_parameter("enable_bias_anchor").value or not gains_file:
            return
        try:
            with open(gains_file, "r") as f:
                qs = yaml.safe_load(f)["cable_ts_model"].get("quasi_static")
        except Exception:
            qs = None
        if not qs:
            return
        n_modes = self.model.n_modes
        self._anchor_u = np.asarray(
            qs["output_map_u"], dtype=float).reshape(n_modes, 2)
        jac = np.asarray(qs["jacobians"], dtype=float)
        jac = jac.reshape(-1, n_modes, 2).mean(axis=0)
        self._anchor_m = self._anchor_u.T @ jac
        self._bias = np.zeros(2)
        self.get_logger().info(
            "manifold bias anchor active: alpha="
            f"{self._anchor_alpha}, |b|max={self._anchor_max}")

    def _effective_target(self):
        """Target shifted onto the perceived equilibrium manifold."""
        if self._anchor_u is None or self._state is None:
            return self._target
        n = self.model.n_modes
        z_err = self._anchor_u.T @ (self._state[:n] - self._target[:n])
        residual = z_err - self._anchor_m @ self._state[-2:]
        self._bias += self._anchor_alpha * (residual - self._bias)
        norm = float(np.linalg.norm(self._bias))
        if norm > self._anchor_max:
            self._bias *= self._anchor_max / norm
        target = self._target.copy()
        target[:n] += self._anchor_u @ self._bias
        return target

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg):
        if not msg.valid:
            return
        state = np.asarray(msg.state, dtype=float)
        if state.size != self.model.state_dim:
            # The reducer and the identified model must agree on the state, or
            # the gains are applied to something they were never designed for.
            self.get_logger().error(
                f"reduced state has {state.size} entries but the model expects "
                f"{self.model.state_dim}; refusing to control",
                throttle_duration_sec=10.0)
            return
        self._state = state
        self._state_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

    def _on_target(self, msg):
        q_star = np.asarray(msg.modal_coordinates, dtype=float)
        n_modes = self.model.n_modes
        if self.coordinates == COORDINATES:
            # a*, [a*, g*] or the full state; a non-rest a* needs its g* (equilibrium).
            try:
                self._target = target_state(q_star, n_modes, self.model.state_dim)
            except ValueError as error:
                self.get_logger().warn(str(error), throttle_duration_sec=5.0)
            return
        if q_star.size != n_modes:
            self.get_logger().warn(
                f"target has {q_star.size} modal coordinates, expected {n_modes}")
            return
        # A shape target is a rest condition: reach it and stop there. Any
        # remaining state entries (the gripper) are regulated to their
        # reference, which is zero by construction of the identified model.
        target = np.zeros(self.model.state_dim)
        target[:n_modes] = q_star
        self._target = target

    def _on_parameter(self, msg):
        if msg.std_dev > self.max_std_dev:
            self.get_logger().warn(
                f"rejected {msg.parameter_name}: std dev {msg.std_dev:.4g} > "
                f"{self.max_std_dev:.4g}", throttle_duration_sec=10.0)
            return
        if abs(msg.innovation) > self.max_innovation:
            self.get_logger().warn(
                f"rejected {msg.parameter_name}: innovation {msg.innovation:.4g} "
                f"diverging", throttle_duration_sec=10.0)
            return
        self._estimates[msg.parameter_name] = msg.value
        self._parameter_stamp = self._now_s()

    def _estimate_is_fresh(self):
        return (self._parameter_stamp is not None
                and self._now_s() - self._parameter_stamp <= self.estimate_timeout)

    def _publish_zero_twist(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.reference_frame
        self.twist_pub.publish(msg)

    def _gripper_position(self):
        """Current gripper position in the reference frame, or None."""
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.reference_frame, self.gripper_frame, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tfm.transform.translation
        return np.array([t.x, t.y, t.z])

    def _publish_desired_pose(self, stamp, command):
        """Where the gripper should be, one lookahead ahead of now."""
        current = self._gripper_position()
        if current is None:
            self.get_logger().warn(
                f"no transform {self.reference_frame} -> {self.gripper_frame}; "
                f"no pose reference published", throttle_duration_sec=5.0)
            return
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.reference_frame
        pose.pose.position.x = float(current[0] + self.lookahead * command[0])
        pose.pose.position.y = float(current[1] + self.lookahead * command[1])
        pose.pose.position.z = float(current[2])
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

    def _control_cycle(self):
        if self._state is None:
            return
        if self._now_s() - self._state_stamp > self.state_timeout:
            self._publish_zero_twist()
            self.get_logger().warn("reduced state is stale; holding still",
                                   throttle_duration_sec=2.0)
            return

        target = self._effective_target()
        command, weights = self.model.control(self._state, target)
        norm = float(np.linalg.norm(command))
        if norm > self.max_linear:
            command = command * (self.max_linear / norm)

        stamp = self.get_clock().now().to_msg()

        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = self.reference_frame
        twist.twist.linear.x = float(command[0])
        twist.twist.linear.y = float(command[1])
        self.twist_pub.publish(twist)

        self._publish_desired_pose(stamp, command)

        error = self._state - self._target
        self.error_pub.publish(Float64MultiArray(
            data=[*error.tolist(), float(np.linalg.norm(error))]))
        self.weights_pub.publish(Float64MultiArray(
            data=[*weights.tolist(), *command.tolist(), norm,
                  1.0 if self._estimate_is_fresh() else 0.0]))

        # V is evaluated on the error the law actually regulates.
        lyapunov = self.model.lyapunov_value(self._state, target)
        decrease = (0.0 if self._previous_lyapunov is None
                    else lyapunov - self._previous_lyapunov)
        self._previous_lyapunov = lyapunov
        self.lyapunov_pub.publish(Float64MultiArray(
            data=[lyapunov, decrease, float(np.linalg.norm(error)),
                  1.0 if decrease <= 0.0 else 0.0]))


def main():
    rclpy.init()
    node = CableTsControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
