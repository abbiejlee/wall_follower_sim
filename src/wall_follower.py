#!/usr/bin/env python2
"""
    16.405 Spring 2019 | Wall-following racecar
    Author: Abbie Lee (abbielee@mit.edu)
"""

import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Float32
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollower:
    # Import ROS parameters from the "params.yaml" file.
    # Access these variables in class functions with self:
    # i.e. self.CONSTANT
    SCAN_TOPIC = rospy.get_param("wall_follower/scan_topic")
    DRIVE_TOPIC = rospy.get_param("wall_follower/drive_topic")
    SIDE = rospy.get_param("wall_follower/side")
    VELOCITY = rospy.get_param("wall_follower/velocity")
    DESIRED_DISTANCE = rospy.get_param("wall_follower/desired_distance")
    LOOP_RATE = 50.0
    WHEEL_BASE = rospy.get_param("racecar_simulator/wheelbase")
    BASE_FRAME = rospy.get_param("racecar_simulator/base_frame")

    def __init__(self):
        self.pub = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size = 10)
        self.sub = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.laser_cb)
        self.cmd_timer = rospy.Timer(rospy.Duration(1.0/self.LOOP_RATE), self.control_loop_cb)
        self.scan = LaserScan() # current LaserScan
        self.drive = AckermannDriveStamped() # current Drive command

        # Ackermann message setup
        self.seq = 0

        self.state = {"distance": 0.0, "deriv": 0.0}
        self.gains = np.array([20.0, 5.0, 0.5]) # PDI gains
        self.error = np.array([0.0, 0.0, 0.0]) # error, derror/dt, int of error

        self.pub2 = rospy.Publisher("distances", Float32, queue_size=10)
        self.fake_scan_pub = rospy.Publisher("/fake_scan", LaserScan, queue_size=1)

        self.debug = True


    def laser_cb(self, laser_data):
        self.scan = laser_data
        self.estimate_state()

    def control_loop_cb(self, event):
        if self.debug:
            rospy.loginfo(self.state)
            self.pub2.publish(self.state)

        self.get_drive_cmd()
        # self.calc_steering_angle()

        self.pub.publish(self.drive)

    def estimate_state(self):
        """
        Given scan data and wall follower side, use laser data to estimate
        distance from the wall.

        Takes average of hcos(a) where h is the laser measurement and a is its
        corresponding angle for a cone of measurements spanning XXX degrees.
        """
        beta = np.radians(45) # range of cone of measurements to use for dist
        ref_ang = np.radians(30) # angle from straight ahead on which cone is centered
        meas = np.array(self.scan.ranges)
        angle_min = self.scan.angle_min
        angle_max = self.scan.angle_max
        angle_increment = self.scan.angle_increment
        angle_ranges = np.linspace(angle_min, angle_max, len(meas))

        if self.SIDE == -1:
            # right side
            start_i = int(len(meas)/2 - ((ref_ang + beta/2)/angle_increment))
            stop_i = int(start_i + beta/angle_increment)
            side_i = int(len(meas)/2-(ref_ang)/angle_increment)
        elif self.SIDE == 1:
            # left side
            start_i = int(len(meas)/2 + (ref_ang - beta/2)/angle_increment)
            stop_i = int(start_i + beta/angle_increment)
            side_i = int(len(meas)/2+(ref_ang)/angle_increment)

        dists = np.absolute(meas[start_i:stop_i] * np.sin(angle_ranges[start_i:stop_i]))

        # filter out points that are more than 1 standard deviation away from
        # the mean

        mean = np.mean(dists)
        sd = np.std(dists)

        filt = lambda x: x if mean-sd < x < mean + sd else mean

        filtered = list(map(filt, dists))

        fit_line = np.polyfit(angle_ranges[start_i:stop_i], filtered, 1)

        self.state["distance"] = np.polyval(fit_line, angle_ranges[side_i])

        # _ = plt.plot(angle_ranges[start_i:stop_i], filtered, np.poly1d(fit_line)(angle_ranges[start_i:stop_i]))
        # plt.show()

        # deriv > 1 for convex corner, < 1 for concave
        self.state["deriv"] = fit_line[0] # slope of line of best fit


        if self.debug:
            # publishes fake LaserScan to visualize range used to calculate
            # distance to wall
            scan_narrow_msg = LaserScan()
            scan_narrow_msg.header.seq += 1
            scan_narrow_msg.header.stamp = rospy.Time.now()
            scan_narrow_msg.header.frame_id = self.scan.header.frame_id
            scan_narrow_msg.angle_min = self.scan.angle_min
            scan_narrow_msg.angle_max = self.scan.angle_max
            scan_narrow_msg.angle_increment = self.scan.angle_increment
            scan_narrow_msg.time_increment = self.scan.time_increment
            scan_narrow_msg.scan_time = rospy.Time.now().to_sec()
            scan_narrow_msg.range_min = self.scan.range_min
            scan_narrow_msg.range_max = self.scan.range_max
            scan_narrow_msg.intensities = self.scan.intensities
            scan_narrow_ranges = [0] * len(self.scan.ranges)
            for i in range(start_i, stop_i):
                scan_narrow_ranges[i] = meas[i]
            scan_narrow_msg.ranges = scan_narrow_ranges
            self.fake_scan_pub.publish(scan_narrow_msg)

    def gen_header(self):
        self.drive.header.seq += 1
        self.drive.header.stamp = rospy.Time.now()
        self.drive.header.frame_id = self.BASE_FRAME

    # def calc_steering_angle(self):
    #     ''' Calculate the desired Ackermann steering angle
    #     The reference is a distance twice the wheelbase in front of the car
    #     and 1m away from the wall.
    #     '''
    #     current_error = self.state["distance"] - self.DESIRED_DISTANCE
    #     ref_dist = 2.0 * self.WHEEL_BASE
    #     eta = np.arctan(current_error/ref_dist)
    #     L1 = current_error/np.sin(eta)
    #
    #     steering_angle = 2.0 * self.SIDE * np.arctan(ref_dist*np.sin(eta)/L1)
    #     self.drive.drive.speed = self.VELOCITY
    #     self.drive.drive.steering_angle = steering_angle # radians


    def get_drive_cmd(self):
        """
        Generates a steering angle with PD controller
        lecture 5 (20 Feb).
        """
        current_error = self.state["distance"] - self.DESIRED_DISTANCE
        self.error[1] = current_error - self.error[0]
        self.error[0] = current_error
        self.error[2] += current_error

        ref_dist = 2.0 * self.WHEEL_BASE

        steering_angle = self.SIDE * self.gains[0] * np.arctan(self.error[0]/ref_dist)

        # use derivative term to predict corners
        steering_angle += self.SIDE * self.gains[1] * self.state["deriv"]

        # add error derivative term to mitigate wobble
        steering_angle += self.SIDE * self.gains[1] * self.error[1]

        # add error integral term to mitigate steady state error
        # steering_angle += self.SIDE * self.gains[2] * self.error[2]

        # vel_factor = 1.0 + np.absolute(self.error[0])*self.gains[0]/3.0

        self.drive.drive.speed = self.VELOCITY
        self.drive.drive.steering_angle = steering_angle # radians

if __name__ == "__main__":
    rospy.init_node('wall_follower')
    wall_follower = WallFollower()
    rospy.spin()
