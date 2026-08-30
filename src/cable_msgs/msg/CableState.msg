# Reduced (modal) cable shape state produced by the state reducer.
# The TS-PDC controller consumes this instead of raw marker positions.

std_msgs/Header header

float64[] modal_coordinates   # q   [m] modal amplitudes, length = n_modes
float64[] modal_velocities    # qd  [m/s] finite-difference derivative of q
float64[] state               # x = [q; qd], length = 2 * n_modes

float64 reconstruction_rmse   # ||y - (y_mean + Phi q)|| / sqrt(M) [m]
uint32 markers_used           # number of valid markers used for the projection
bool valid                    # false when the projection could not be trusted
