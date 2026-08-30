# Deterministic grasp state of the SOFA cable tip.
# Replaces the implicit proximity latch so experiments are repeatable.

std_msgs/Header header

uint8 DETACHED=0
uint8 REQUESTED=1     # attach asked for, waiting for the gripper to be in range
uint8 ATTACHED=2

uint8 state
float64 distance_to_tip_m     # xy distance gripper -> cable tip [m]
float64 height_above_plane_m  # gripper z minus fixture z [m]
bool in_range                 # geometric attach condition currently satisfied
