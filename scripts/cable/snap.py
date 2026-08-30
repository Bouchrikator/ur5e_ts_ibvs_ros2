import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

topic = sys.argv[1] if len(sys.argv) > 1 else "/overview/image"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/overview.png"

rclpy.init()
node = Node("snap")
got = {}


def cb(msg):
    got["msg"] = msg


sub = node.create_subscription(Image, topic, cb, 1)
import time

t0 = time.time()
while "msg" not in got and time.time() - t0 < 15:
    rclpy.spin_once(node, timeout_sec=0.5)

if "msg" not in got:
    print("NO IMAGE")
    sys.exit(1)

m = got["msg"]
import numpy as np

arr = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, -1)
if m.encoding in ("rgb8",):
    bgr = arr[:, :, ::-1]
else:
    bgr = arr
import cv2

cv2.imwrite(out, bgr)
print(f"saved {out} {m.width}x{m.height} enc={m.encoding}")
