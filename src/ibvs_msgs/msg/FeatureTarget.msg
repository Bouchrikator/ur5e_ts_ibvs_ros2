# Output of the cube detector (perception node).
# Corners are temporally matched so index i refers to the same physical corner
# across frames; features are the corresponding normalized image coordinates.

std_msgs/Header header

bool detected                    # false while the target is lost

uint32 image_width
uint32 image_height

float64[8] corner_pixels         # u0,v0,...,u3,v3 (px), temporally matched
float64[8] features_normalized   # x0,y0,...,x3,y3 (normalized coordinates)

float64 centroid_u               # blob centroid (px), used for edge damping
float64 centroid_v
float64 area                     # blob area (px^2)

bool pose_valid                  # PnP (DEMENTHON + VIRTUAL_VS) succeeded
float64 z_pnp                    # depth from PnP, or area-based estimate (m)

bool depth_valid                 # aligned depth image sample succeeded
float64 z_depth                  # median corner/centroid depth sample (m)

# Camera intrinsics, forwarded so controllers can convert normalized errors to
# pixels and build pixel-space Jacobians without a CameraInfo subscription.
float64 fx
float64 fy
float64 cx
float64 cy
