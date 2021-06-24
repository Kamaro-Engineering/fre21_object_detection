#!/usr/bin/env python3

import onnxruntime
import numpy as np
from cv_bridge import CvBridge
import rospy
import rospkg
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import String
import cv2
import os


class Detector:
    def __init__(self):
        pkg_path = rospkg.RosPack().get_path("fre_object_detection")
        self.ort_session = onnxruntime.InferenceSession(
            os.path.join(pkg_path, "resources", "net.onnx")
        )
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/front/image_raw", Image, callback=self.image_cb, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            "/odometry/filtered", Odometry, callback=self.odom_cb, queue_size=1
        )
        self.detection_pub = rospy.Publisher("/fre_detections", String, queue_size=10)
        self.debug_image_pub = rospy.Publisher("/detector_debug", Image, queue_size=1)
        self.current_image = None
        self.min_distance = 0.5 ** 2
        self.last_weed = Point()
        self.last_trash = Point()
        self.current_pos = Point()

    def image_cb(self, msg):
        self.current_image = msg

    def odom_cb(self, msg):
        self.current_pos = msg.pose.pose.position

    def process_image(self):
        if self.current_image is None:
            return

        msg = self.current_image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        frame = frame.astype(np.float32) / 255.0
        mask = self.ort_session.run(
            None, {"input": frame.swapaxes(0, 2)[np.newaxis, :, :, :]}
        )[0]
        mask = mask.swapaxes(2, 3)
        mask = mask.clip(0, 1)
        weeds = mask[0, 2, :, :] > 0.9
        trash = mask[0, 3, :, :] > 0.9
        weeds_pixels = weeds.sum()
        trash_pixels = trash.sum()
        if weeds_pixels > 0:
            rospy.loginfo("weeds pixels: {}".format(weeds_pixels))
        if trash_pixels > 0:
            rospy.loginfo("trash pixels: {}".format(trash_pixels))

        debug_image = np.stack(
            [mask[0, 2, :, :], mask[0, 1, :, :], mask[0, 3, :, :]], axis=-1
        )

        debug_image = np.concatenate(
            [np.ones((58, 640, 3)), frame, debug_image], axis=0
        )

        if weeds_pixels > 100 and self.min_distance < self.dist2(
            self.current_pos, self.last_weed
        ):
            cv2.putText(
                debug_image, "weed!", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (1, 0, 0), 2
            )
            self.detection_pub.publish("weed")
            self.last_weed = self.current_pos

        if trash_pixels > 1000 and self.min_distance < self.dist2(
            self.current_pos, self.last_trash
        ):
            cv2.putText(
                debug_image,
                "litter!",
                (550, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 1),
                2,
            )
            self.detection_pub.publish("litter")
            self.last_trash = self.current_pos

        debug_image = (debug_image * 255).astype(np.uint8)
        self.debug_image_pub.publish(
            self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
        )

    def dist2(self, pose1, pose2):
        return (pose1.x - pose2.x) ** 2 + (pose1.y - pose2.y) ** 2


if __name__ == "__main__":
    rospy.init_node("fre21_object_detection_node")
    detector = Detector()
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        detector.process_image()
        r.sleep()
