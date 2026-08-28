# One cable marker observation/prediction, expressed in the frame given by
# the enclosing CableMarkerArray header (canonical frame: base_link).
uint32 id
float64 s_over_l              # normalized curvilinear coordinate in [0, 1]
geometry_msgs/Point position
float64[9] covariance         # row-major 3x3 position covariance [m^2]
float32 confidence
bool valid
