# Desired cable shape for the outer TS-PDC loop.
# Expressed in the same modal basis as CableState (same basis file).

std_msgs/Header header

string target_name              # label of the target shape, for logging
float64[] modal_coordinates     # q*, length = n_modes
float64[] marker_positions      # optional flattened xyz of the target markers [m]
