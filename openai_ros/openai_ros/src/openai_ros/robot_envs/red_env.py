import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
# IMU
from sensor_msgs.msg import Imu
# robot pose
# robot-container distance
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from openai_ros.openai_ros_common import ROSLauncher


class RedEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new RedEnv environment.
        Turtlebot2 doesnt use controller_manager, therefore we wont reset the
        controllers in the standard fashion. For the moment we wont reset them.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/depth/image_raw: 2d Depth image of the depth sensor.
        * /camera/depth/points: Pointcloud sensor readings
        * /camera/rgb/image_raw: RGB camera
        * /kobuki/laser/scan: Laser Readings

        Actuators Topic List: /cmd_vel,

        Args:
        """
        rospy.logdebug("Start RedEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        # red spawn
        ROSLauncher(rospackage_name="turtlebot_gazebo",
                    launch_file_name="put_robot_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(RedEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")




        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        # red subscriber
        rospy.Subscriber("/imu/data_raw", Imu, self._imu_callback)
        rospy.Subscriber("/pose_gazebo", PoseStamped, self._pose_callback)
        rospy.Subscriber("/three_dist_3d", PoseStamped, self._dist_callback)
        # publisher
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()

        self.gazebo.pauseSim()

        rospy.logdebug("Finished RedEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------


    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    # all
    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_imu_ready()
        self._check_pose_ready()
        self._check_dist_ready()
        rospy.logdebug("ALL SENSORS READY")

    # # several sensors
    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /imu/data_raw to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/imu/data_raw", Imu, timeout=5.0)
                rospy.logdebug("Current /imu/data_raw READY=>")

            except:
                rospy.logerr("Current /imu/data_raw not ready yet, retrying for getting imu")
        return self.imu

    def _check_pose_ready(self):
        self.pose = None
        rospy.logdebug("Waiting for /pose_gazebo to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/pose_gazebo", PoseStamped, timeout=5.0)
                rospy.logdebug("Current /pose_gazebo READY=>")

            except:
                rospy.logerr("Current /pose_gazebo not ready yet, retrying for getting imu")
        return self.imu
    
    def _check_dist_ready(self):
        self.dist = None
        rospy.logdebug("Waiting for /three_dist_3d to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/three_dist_3d", PoseStamped, timeout=5.0)
                rospy.logdebug("Current /three_dist_3d READY=>")

            except:
                rospy.logerr("Current /three_dist_3d not ready yet, retrying for getting imu")
        return self.imu


    # callback function
    def _imu_callback(self, data):
        self.imu = data

    def _pose_callback(self, data):
        self.pose = data

    def _dist_callback(self, data):
        self.dist = data


    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("Red Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(0.2)
        #time.sleep(0.02)
        """
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        min_laser_distance)
        """


    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate, min_laser_distance=-1):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rospy.logwarn("START wait_until_twist_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05

        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))

        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon

        while not rospy.is_shutdown():

            crashed_into_something = self.has_crashed(min_laser_distance)

            current_odometry = self._check_odom_ready()
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z

            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) + ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")

            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)

            if linear_vel_are_close and angular_vel_are_close:
                rospy.logwarn("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break

            if crashed_into_something:
                rospy.logerr("TurtleBot has crashed, stopping movement!")
                break

            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")

        rospy.logwarn("END wait_until_twist_achieved...")

        return delta_time

    def has_crashed(self, min_laser_distance):
        """
        It states based on the laser scan if the robot has crashed or not.
        Crashed means that the minimum laser reading is lower than the
        min_laser_distance value given.
        If min_laser_distance == -1, it returns always false, because its the way
        to deactivate this check.
        """
        robot_has_crashed = False

        if min_laser_distance != -1:
            laser_data = self.get_laser_scan()
            for i, item in enumerate(laser_data.ranges):
                if item == float ('Inf') or numpy.isinf(item):
                    pass
                elif numpy.isnan(item):
                   pass
                else:
                    # Has a Non Infinite or Nan Value
                    if (item < min_laser_distance):
                        rospy.logerr("TurtleBot HAS CRASHED >>> item=" + str(item)+"< "+str(min_laser_distance))
                        robot_has_crashed = True
                        break
        return robot_has_crashed


    def get_odom(self):
        return self.odom

    def get_camera_depth_image_raw(self):
        return self.camera_depth_image_raw

    def get_camera_depth_points(self):
        return self.camera_depth_points

    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw

    def get_laser_scan(self):
        return self.laser_scan

    def get_imu(self):
        return self.imu
    
    def get_pose(self):
        return self.imu
    
    def get_dist(self):
        return self.imu

    def reinit_sensors(self):
        """
        This method is for the tasks so that when reseting the episode
        the sensors values are forced to be updated with the real data and

        """

