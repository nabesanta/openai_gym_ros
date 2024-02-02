import rospy
import numpy
import time
import math
from gym import spaces
from openai_ros.robot_envs import red_env
from gym.envs.registration import register
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os


class RedMadoanaEnv(red_env.RedEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        # This parameter HAS to be set up in the MAIN launch of the AI RL script
        ros_ws_abspath = rospy.get_param("/red/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path "+ros_ws_abspath + \
            " DOESNT exist, execute: mkdir -p "+ros_ws_abspath + \
            "/src;cd "+ros_ws_abspath+";catkin_make"

        # gazebo world start
        ROSLauncher(rospackage_name="turtlebot_gazebo",
                    launch_file_name="start_world_madoana.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot2/config",
                               yaml_file_name="red_madoana.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(RedMadoanaEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/red/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        # OpenAI default value
        self.reward_range = (-numpy.inf, numpy.inf)

        #number_observations = rospy.get_param('/turtlebot2/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.dec_obs = rospy.get_param(
            "/red/number_decimals_precision_obs", 1)
        self.linear_forward_speed_high = rospy.get_param(
            '/red/linear_forward_speed_high')
        self.linear_forward_speed_middle = rospy.get_param(
            '/red/linear_forward_speed_middle')
        self.linear_forward_speed_low = rospy.get_param(
            '/red/linear_forward_speed_low')
        self.linear_turn_speed = rospy.get_param(
            '/red/linear_turn_speed')
        self.angular_speed_high = rospy.get_param('/red/angular_speed_high')
        self.angular_speed_middle = rospy.get_param('/red/angular_speed_middle')
        self.angular_speed_low = rospy.get_param('/red/angular_speed_low')
        # initialization
        self.init_linear_forward_speed_high = rospy.get_param(
            '/red/init_linear_forward_speed_high')
        self.init_linear_forward_speed_middle = rospy.get_param(
            '/red/init_linear_forward_speed_middle')
        self.init_linear_forward_speed_low = rospy.get_param(
            '/red/init_linear_forward_speed_low')
        self.init_linear_turn_speed = rospy.get_param(
            '/red/init_linear_turn_speed')
        self.init_angular_speed_high = rospy.get_param(
            '/red/init_angular_speed_high')
        self.init_angular_speed_middle = rospy.get_param(
            '/red/init_angular_speed_middle')
        self.init_angular_speed_low = rospy.get_param(
            '/red/init_angular_speed_low')

        self.n_observations = rospy.get_param('/red/n_observations')
        self.min_range = rospy.get_param('/red/min_range')
        self.max_laser_value = rospy.get_param('/red/max_laser_value')
        self.min_laser_value = rospy.get_param('/red/min_laser_value')

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        # ==== laer ====
        #laser_scan = self._check_laser_scan_ready()
        # laser_scan = self.get_imu()
        # rospy.logdebug("laser_scan len===>"+str(len(laser_scan.ranges)))
        # ==== imu, pose, dist ====
        imu = self.get_imu()
        rospy.logdebug("imu len===>"+str(len(imu.ranges)))
        pose = self.get_pose()
        rospy.logdebug("pose len===>"+str(len(pose.ranges)))
        dist = self.get_dist()
        rospy.logdebug("dist len===>"+str(len(dist.ranges)))


        # # Laser data
        # self.laser_scan_frame = laser_scan.header.frame_id

        # imu, pose, dist data
        self.imu_frame = imu.header.frame_id
        self.pose_frame = pose.header.frame_id
        self.dist_frame = dist.header.frame_id

        # Number of laser reading jumped
        # Math.ceil() 関数は、引数として与えた数以上の最小の整数を返します。
        # self.new_ranges = int(
        #     math.ceil(float(len(laser_scan.ranges)) / float(self.n_observations)))

        # action patter
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/red/forwards_reward")
        self.turn_reward = rospy.get_param("/red/turn_reward")
        self.end_episode_points = rospy.get_param(
            "/red/end_episode_points")

        self.cumulated_steps = 0.0

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base(self.init_linear_forward_speed_high,                       
                       self.init_linear_turn_speed,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=-1)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0:
            linear_speed = self.linear_forward_speed_high
            angular_speed = 0.0
            self.last_action = "FORWARDS_HIGH"
        elif action == 1:
            linear_speed = self.linear_forward_speed_middle
            angular_speed = 0.0
            self.last_action = "FORWARDS_MIDDLE"
        elif action == 2:
            linear_speed = self.linear_forward_speed_low
            angular_speed = 0.0
            self.last_action = "FORWARDS_LOW"
        elif action == 3: 
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed_high
            self.last_action = "TURN_LEFT_HIGH"
        elif action == 4:
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed_middle
            self.last_action = "TURN_LEFT_MIDDLE"
        elif action == 5:
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed_low
            self.last_action = "TURN_LEFT_LOW"
        elif action == 6:
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed_high
            self.last_action = "TURN_RIGHT_HIGH"
        elif action == 7:
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed_middle
            self.last_action = "TURN_RIGHT_MIDDLE"
        elif action == 8:
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed_low
            self.last_action = "TURN_RIGHT_LOW"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed,
                       angular_speed,
                       epsilon=0.05,
                       update_rate=10,
                       min_laser_distance=self.min_range)

        rospy.logdebug("END Set Action ==>"+str(action) +
                       ", NAME="+str(self.last_action))

    def _is_done(self, observations):

        if self._episode_done:
            rospy.logdebug("TurtleBot2 is Too Close to wall==>" +
                           str(self._episode_done))
        else:
            rospy.logerr("TurtleBot2 is Ok ==>")

        return self._episode_done

    def _compute_reward(self, observations, done):
        # コンテナとの距離が近づいたら報酬を与える
        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
        else:
            reward = -1*self.end_episode_points

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods
