"""Records the validation metrics the plan asks for into a CSV (step 12).

One row per control period, so a run can be replayed in a table or plotted
without a rosbag. Everything is optional: whatever is not being published stays
empty, which lets the same recorder cover validation runs 1 through 14 as the
stack grows from "two SOFA models, no controller" to the full cascade.

Columns follow the plan's record list: shape RMSE, maximum marker error,
parameter error and standard deviation, innovation, TS weights, Lyapunov value,
command saturation, SOFA computation time, dropped SOFA steps, and end-to-end
latency.
"""

import csv
import os

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray, String

from cable_msgs.msg import CableMarkerArray, CableParameterEstimate

from cable_ts_control.metrics import (
    latency_s,
    marker_errors,
    relative_error,
    saturation_ratio,
)

COLUMNS = [
    "t", "supervisor_state",
    "shape_error_norm", "shape_error_max",
    "marker_rmse_observed", "marker_max_observed",
    "marker_rmse_model", "marker_max_model",
    "parameter", "parameter_value", "parameter_std_dev", "parameter_rel_error",
    "innovation", "estimator_marker_rmse",
    "ts_weights", "lyapunov",
    "command_saturation", "latency_s",
    "sofa_compute_ms", "sofa_dropped_steps",
]


def _points(msg):
    """Valid markers of a CableMarkerArray as ``{id: (x, y, z)}``."""
    return {m.id: (m.position.x, m.position.y, m.position.z)
            for m in msg.markers if m.valid}


class CableMetricsRecorderNode(Node):
    def __init__(self):
        super().__init__("cable_metrics_recorder")

        self.declare_parameter("output_csv", "/root/ibvs_logs/cable_metrics.csv")
        self.declare_parameter("record_rate_hz", 30.0)
        self.declare_parameter("max_linear_vel", 0.05)
        self.declare_parameter("truth_config", "")
        self.declare_parameter("identified_parameters", ["EI"])

        self.limit = float(self.get_parameter("max_linear_vel").value)
        self.truth_values = self._load_truth()

        path = self.get_parameter("output_csv").value
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()

        self._truth_markers = None
        self._observed = None
        self._observed_stamp = None
        self._model = None
        self._estimate = None
        self._shape_error = None
        self._weights = None
        self._lyapunov = None
        self._twist = None
        self._twist_stamp = None
        self._supervisor = ""
        self._solver = None

        self.create_subscription(CableMarkerArray, "/cable/truth/markers",
                                 self._on_truth, 10)
        self.create_subscription(CableMarkerArray, "/cable/observed_markers",
                                 self._on_observed, 10)
        self.create_subscription(CableMarkerArray,
                                 "/cable/model/predicted_markers",
                                 self._on_model, 10)
        self.create_subscription(CableParameterEstimate,
                                 "/cable/parameter_estimate",
                                 self._on_estimate, 10)
        self.create_subscription(Float64MultiArray, "/cable/shape_error",
                                 self._on_shape_error, 10)
        self.create_subscription(Float64MultiArray, "/cable/ts_weights",
                                 self._on_weights, 10)
        self.create_subscription(Float64MultiArray, "/cable/lyapunov",
                                 self._on_lyapunov, 10)
        self.create_subscription(TwistStamped, "/cable/desired_gripper_twist",
                                 self._on_twist, 10)
        self.create_subscription(String, "/cable/supervisor_state",
                                 self._on_supervisor, 10)
        self.create_subscription(Float64MultiArray, "/cable/truth/solver_stats",
                                 self._on_solver, 10)

        rate = float(self.get_parameter("record_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self._record)
        self.get_logger().info(f"recording cable metrics to {path}")

    # ------------------------------------------------------------------
    def _load_truth(self):
        """Hidden truth values, used only to score the estimate."""
        path = self.get_parameter("truth_config").value
        if not path:
            return {}
        from cable_identification.cosserat_model import load_config
        cfg = load_config(path)
        keys = {"EI": "EI_Nm2", "GJ": "GJ_Nm2",
                "rayleigh_stiffness": "rayleigh_stiffness_s",
                "rayleigh_mass": "rayleigh_mass_per_s"}
        return {name: float(cfg[key]) for name, key in keys.items() if key in cfg}

    def _on_truth(self, msg):
        self._truth_markers = _points(msg)

    def _on_observed(self, msg):
        self._observed = _points(msg)
        self._observed_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

    def _on_model(self, msg):
        self._model = _points(msg)

    def _on_estimate(self, msg):
        self._estimate = msg

    def _on_shape_error(self, msg):
        self._shape_error = list(msg.data)

    def _on_weights(self, msg):
        self._weights = list(msg.data)

    def _on_lyapunov(self, msg):
        self._lyapunov = list(msg.data)

    def _on_twist(self, msg):
        self._twist = msg.twist
        self._twist_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

    def _on_supervisor(self, msg):
        self._supervisor = msg.data

    def _on_solver(self, msg):
        self._solver = list(msg.data)

    # ------------------------------------------------------------------
    def _record(self):
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            return
        row = {name: "" for name in COLUMNS}
        row["t"] = f"{now.nanoseconds * 1e-9:.4f}"
        row["supervisor_state"] = self._supervisor

        if self._shape_error:
            norm = sum(v * v for v in self._shape_error) ** 0.5
            row["shape_error_norm"] = f"{norm:.6f}"
            row["shape_error_max"] = f"{max(abs(v) for v in self._shape_error):.6f}"

        if self._truth_markers and self._observed:
            rmse, worst, _ = marker_errors(self._truth_markers, self._observed)
            row["marker_rmse_observed"] = f"{rmse:.6f}"
            row["marker_max_observed"] = f"{worst:.6f}"

        if self._truth_markers and self._model:
            rmse, worst, _ = marker_errors(self._truth_markers, self._model)
            row["marker_rmse_model"] = f"{rmse:.6f}"
            row["marker_max_model"] = f"{worst:.6f}"

        if self._estimate is not None:
            name = self._estimate.parameter_name
            row["parameter"] = name
            row["parameter_value"] = f"{self._estimate.value:.6g}"
            row["parameter_std_dev"] = f"{self._estimate.std_dev:.6g}"
            row["innovation"] = f"{self._estimate.innovation:.6f}"
            row["estimator_marker_rmse"] = f"{self._estimate.marker_rmse:.6f}"
            if name in self.truth_values:
                row["parameter_rel_error"] = (
                    f"{relative_error(self._estimate.value, self.truth_values[name]):.6f}")

        if self._weights:
            row["ts_weights"] = " ".join(f"{w:.4f}" for w in self._weights)
        if self._lyapunov:
            row["lyapunov"] = f"{self._lyapunov[0]:.6g}"

        if self._twist is not None:
            linear = (self._twist.linear.x, self._twist.linear.y,
                      self._twist.linear.z)
            row["command_saturation"] = f"{saturation_ratio(linear, self.limit):.4f}"
            if self._observed_stamp is not None:
                row["latency_s"] = (
                    f"{latency_s(self._observed_stamp, self._twist_stamp):.4f}")

        if self._solver and len(self._solver) >= 2:
            row["sofa_compute_ms"] = f"{self._solver[0]:.3f}"
            row["sofa_dropped_steps"] = f"{self._solver[1]:.0f}"

        self._writer.writerow(row)
        self._file.flush()

    def destroy_node(self):
        self._file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = CableMetricsRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
