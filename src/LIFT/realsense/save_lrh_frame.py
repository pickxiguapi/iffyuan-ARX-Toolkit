import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from message_filters import Subscriber, ApproximateTimeSynchronizer


class SaveThreeCameras(Node):
    def __init__(self):
        super().__init__('save_three_cameras')

        self.bridge = CvBridge()
        self.saved = False

        self.sub_l = Subscriber(self, Image, '/camera/camera_l/color/image_rect_raw')
        self.sub_r = Subscriber(self, Image, '/camera/camera_r/color/image_rect_raw')
        self.sub_h = Subscriber(self, Image, '/camera/camera_h/color/image_rect_raw')

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_l, self.sub_r, self.sub_h],
            queue_size=10,
            slop=0.05  # 50 ms
        )
        self.sync.registerCallback(self.cb)
        self.get_logger().info(' init successful ')

    def cb(self, msg_l, msg_r, msg_h):
        if self.saved:
            return

        img_l = self.bridge.imgmsg_to_cv2(msg_l, 'bgr8')
        img_r = self.bridge.imgmsg_to_cv2(msg_r, 'bgr8')
        img_h = self.bridge.imgmsg_to_cv2(msg_h, 'bgr8')

        cv2.imwrite('camera_l.png', img_l)
        cv2.imwrite('camera_r.png', img_r)
        cv2.imwrite('camera_h.png', img_h)

        self.get_logger().info('Saved camera_l / camera_r / camera_h frames')
        self.saved = True


def main():
    rclpy.init()
    node = SaveThreeCameras()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()