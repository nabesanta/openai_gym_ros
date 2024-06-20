#!/usr/bin/env python3

import os
import time
import rospy
import numpy as np
# ROS関係
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
# 強化学習関係
from gym import spaces
from openai_ros.robot_envs import red_env
from gym.envs.registration import register
from openai_ros.openai_ros_common import ROSLauncher
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest

class RedMadoanaEnv(red_env.RedEnv):
    def __init__(self):
        ros_ws_abspath = rospy.get_param("/myrobot_1/ros_ws_abspath", None)
        # assert文を用いて例外処理
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path "+ros_ws_abspath + \
            " DOESNT exist, execute: mkdir -p "+ros_ws_abspath + \
            "/src;cd "+ros_ws_abspath+";catkin build"
            
        ROSLauncher(rospackage_name="robot_simulation",
                    launch_file_name="start_red_madoana.launch",
                    ros_ws_abspath=ros_ws_abspath)
        
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                                rel_path_from_package_to_file="src/openai_ros/task_envs/red/config",
                                yaml_file_name="red_madoana_pend.yaml")
        
        super(RedMadoanaEnv, self).__init__(ros_ws_abspath)
        
        #~~~ Only variable needed to be set here ~~~
        # 行動空間の設定
        # redの行動は2つ
        number_actions = rospy.get_param('/myrobot_1/n_actions')
        # gym.spaces.Discrete(n): 範囲[0、n-1]の離散値、Int型の数値
        self.action_space = spaces.Discrete(number_actions)
        
        self.reward_range = (-np.inf, np.inf)

        # action
        self.dec_obs = rospy.get_param(
            "/myrobot_1/number_decimals_precision_obs", 1)
        self.linear_forward_speed_high = rospy.get_param(
            '/myrobot_1/linear_forward_speed_high')
        self.angular_speed_high = rospy.get_param(
            '/myrobot_1/angular_speed_high')
        # initialization
        self.init_linear_forward_speed_high = rospy.get_param(
            '/myrobot_1/init_linear_forward_speed_high')
        self.init_angular_speed_high = rospy.get_param(
            '/myrobot_1/init_angular_speed_high')

        # 観測値はロボットの位置、コンテナとの距離
        self.n_observations = rospy.get_param('/myrobot_1/n_observations')
        self.new_ranges = rospy.get_param('/myrobot_1/new_ranges')
        self.min_range = rospy.get_param('/myrobot_1/min_range')
        self.max_distance_value = rospy.get_param('/myrobot_1/max_distance_value')
        self.min_distance_value = rospy.get_param('/myrobot_1/min_distance_value')

        #~~~ We create two arrays based on the binary values that will be assigned ~~~
        # In the discretization method.
        # ==== dist, odom ====
        position = self.get_odom()

        #~~~ imu, pose, dist, odom data ~~~
        self.position_frame = position.header.frame_id

        # robot positoion
        self.robot_x = 0
        self.robot_y = 0
        self.robot_z = 0

        #~~~ 観測値のhighとlowを設定する ~~~
        low = np.array([-np.inf])
        high = np.array([np.inf])
        self.observation_space = spaces.Box(low, high)

        #~~~ action patter ~~~
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        #~~~ observation pattern ~~~
        rospy.logdebug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        #~~~ Rewards ~~~
        # 報酬設定をどうするか
        self.stuck_escape = rospy.get_param("/myrobot_1/stuck_escape")
        self.stuck_escape_container = rospy.get_param("/myrobot_1/stuck_escape_container")
        self.end_episode_points = rospy.get_param("/myrobot_1/end_episode_points")

        # step数の積算
        self.cumulated_steps = 0.0

    #~~~ 最初の速度を与える ~~~
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        # move_baseをどうするかはもう決まっている
        # 差動二輪の原理から算出
        self.move_base(self.init_linear_forward_speed_high,                       
                        self.init_angular_speed_high,
                        epsilon=0.05,
                        update_rate=10,
                        min_laser_distance=-1)

        return True

    #~~~ 環境のセットアップ ~~~
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

        position = self.get_odom()

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

    #~~~ 行動の決定 ~~~
    # move_baseはcmd_velをpublishする
    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        # 前進
        if action == 0:
            linear_speed = self.linear_forward_speed_high
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1:
            linear_speed = -self.linear_forward_speed_high
            angular_speed = 0.0
            self.last_action = "BACKWORD"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed,
                        angular_speed,
                        epsilon=0.05,
                        update_rate=10,
                        min_laser_distance=self.min_range)

        rospy.logdebug("END Set Action ==>"+str(action) + ", NAME="+str(self.last_action))

    #~~~ 状態空間の設定 ~~~
    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        position = self.get_odom()
        odom = position.pose.position.x
        self.robot_x = position.pose.orientation.x
        self.robot_y = position.pose.orientation.y
        self.robot_z = position.pose.orientation.z

        # We round to only two decimals to avoid very big Observation space
        # laser_array = [int(laser_left), int(laser_right)]
        # imu_array = [int(a), int(g)]
        # odom_array = [int(odom), int(odom_x), int(odom_y)]
        
        odom_array = [odom]

        # We only want the X and Y position and the Yaw
        observations = odom_array

        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
    
        return observations, [self.robot_x, self.robot_y, self.robot_z]

    #~~~ 完了判定 ~~~ 
    def _is_done(self, observations):

        if self._episode_done:
            rospy.logdebug("red can't escape stuck" + str(self._episode_done))
        else:

            # 現在のロボットの距離を見る
            current_position = PoseStamped()
            current_position.pose.position.x = observations[-0]
            current_position.pose.position.y = 0.0
            current_position.pose.position.z = 0.0
            # We see if it got to the desired point
            if (90 < abs(current_position.pose.position.x)):
                print(abs(current_position.pose.position.x))
                self._episode_done = True
            else:
                self._episode_done = False

        return self._episode_done

    #~~~ 報酬積算値の計算 ~~~
    def _compute_reward(self, observations, done):

        # 現在のロボットの距離を見る
        current_position = PoseStamped()
        current_position.pose.position.x = observations[-0]
        current_position.pose.position.y = 0.0
        current_position.pose.position.z = 0.0

        # コンテナとの距離が近づいたら報酬を与える
        if not done:
            if (abs(current_position.pose.position.x) < 90):
                reward = -90+abs(current_position.pose.position.x)
            else:
                reward = abs(current_position.pose.position.x)-90

        else:
            if (90 < current_position.pose.position.x):
                reward = 100
        
        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods

    #~~~ データの離散化(ここは使わないかも) ~~~
    def discretize_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        # num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logwarn("new_ranges=" + str(new_ranges))
        rospy.logwarn("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or np.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif np.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges


    #~~~ コンテナの位置に辿り着けば完了とする ~~~
    def is_in_desired_position(self,current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False
        # ちょっと幅を持たせる
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        # 現在のロボットの位置
        x_current = current_position.pose.position.x
        y_current = current_position.pose.position.y
        # 位置判定
        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos

    #~~~~ 現在の位置と所望の位置の距離の差を計算する ~~~
    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    #~~~ 距離計算の関数 ~~~ 
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.pose.position.x, pstart.pose.position.y, pstart.pose.position.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance

