import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, JointState
import message_filters

class SyncSubscriber(Node):
    def __init__(self):
        super().__init__('sync_subscriber')

        # Create message filter subscribers instead of regular ones
        image_sub = message_filters.Subscriber(self, Image, "/rgb_publisher/cam1/image")
        info_sub = message_filters.Subscriber(self, JointState, "/PSM1/measured_js")

        # Use ApproximateTimeSynchronizer or TimeSynchronizer
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, info_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)

    def callback(self, image, depth_image):
        self.get_logger().info(
            f"Received synchronized Image (stamp={image.header.stamp.sec}) "
            f"and Depth Image (stamp={depth_image.header.stamp.sec})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = SyncSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()