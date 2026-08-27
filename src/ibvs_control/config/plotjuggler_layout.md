# PlotJuggler layout for TS-IBVS LMI monitoring
#
# To use:
#   1. ros2 run plotjuggler plotjuggler
#   2. Click "Streaming" -> "ROS 2 Topic Subscriber"
#   3. Select topics:
#        /ts_ibvs_lmi/error
#        /ts_ibvs_lmi/reduced_state
#        /ts_ibvs_lmi/ts_weights
#        /ts_ibvs_lmi/lyapunov
#      (or the _discrete variants)
#   4. Load this layout file
#
# Topic format (Float64MultiArray):
#
# ~/error:          [e0, e1, e2, e3, e4, e5, e6, e7]           (8D raw error)
# ~/reduced_state:  [x0, x1, x2, x3, x4, x5, ||x||]           (6D + norm)
# ~/ts_weights:     [Z, h_close, h_far, vx, vy, vz,            (15 values)
#                    wx, wy, wz, ||u||, ||e||, ||x||,
#                    alpha, Z_min, Z_max]
# ~/lyapunov:       [V, dV/dt_or_deltaV, ||x||, decreasing]    (4 values)
#
# Suggested PlotJuggler panels:
#   Panel 1: Error convergence    -> ~/reduced_state/data[6] (||x||)
#   Panel 2: Lyapunov function    -> ~/lyapunov/data[0] (V), ~/lyapunov/data[1] (dV)
#   Panel 3: TS membership        -> ~/ts_weights/data[1] (h_close), ~/ts_weights/data[2] (h_far)
#   Panel 4: Velocity commands    -> ~/ts_weights/data[3..8] (vx..wz)
#   Panel 5: Depth                -> ~/ts_weights/data[0] (Z)
