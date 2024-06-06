# python関係
import rospy
import numpy as np
import time
import math
import os
# ROS関係
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
# 強化学習関係
from gym import spaces
from openai_ros.robot_envs import red_env
from gym.envs.registration import register
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher

# roslaunchの起動
# roslaunch robot_simulation start_red_maodoana.launch
class RedMadoanaEnv(red_env.RedEnv):
    def __init__(self):
        #~~~ ワークスペースの取得 ~~~
        # ros_ws_abspath: home/nabesanta/red_RL
        ros_ws_abspath = rospy.get_param("/myrobot_1/ros_ws_abspath", None)
        # assert文を用いて例外処理
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path "+ros_ws_abspath + \
            " DOESNT exist, execute: mkdir -p "+ros_ws_abspath + \
            "/src;cd "+ros_ws_abspath+";catkin build"

        #~~~ gazebo world start ~~~
        ROSLauncher(rospackage_name="robot_simulation",
                    launch_file_name="start_red_madoana.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                                rel_path_from_package_to_file="src/openai_ros/task_envs/red/config",
                                yaml_file_name="red_madoana.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(RedMadoanaEnv, self).__init__(ros_ws_abspath)

        #~~~ Only variable needed to be set here ~~~
        # 行動空間の設定
        # redの行動は9つ: z
        number_actions = rospy.get_param('/myrobot_1/n_actions')
        # gym.spaces.Discrete(n): 範囲[0、n-1]の離散値、Int型の数値
        self.action_space = spaces.Discrete(number_actions)

        #~~~ We set the reward range, which is not compulsory but here we do it. ~~~
        # OpenAIのデフォルト値
        self.reward_range = (-np.inf, np.inf)

        """
        We set the Observation space for the 10 observations
        cube_observations = [
            round(lx_acceleration, 1),
            round(ly_acceleration, 1),
            round(lz_acceleration, 1),
            round(gx_acceleration, 1),
            round(gy_acceleration, 1),
            round(gz_acceleration, 1),
            round(roll, 1),
            round(pitch, 1),
            round(yaw, 1),
            round(rc_distance, 1),
        ]
        """

        #~~~ Actions and Observations ~~~
        # action
        self.dec_obs = rospy.get_param(
            "/myrobot_1/number_decimals_precision_obs", 1)
        self.linear_forward_speed_high = rospy.get_param(
            '/myrobot_1/linear_forward_speed_high')
        self.linear_forward_speed_middle = rospy.get_param(
            '/myrobot_1/linear_forward_speed_middle')
        self.linear_forward_speed_low = rospy.get_param(
            '/myrobot_1/linear_forward_speed_low')
        self.linear_turn_speed = rospy.get_param(
            '/myrobot_1/linear_turn_speed')
        self.angular_speed_high = rospy.get_param('/myrobot_1/angular_speed_high')
        self.angular_speed_middle = rospy.get_param('/myrobot_1/angular_speed_middle')
        self.angular_speed_low = rospy.get_param('/myrobot_1/angular_speed_low')
        # initialization
        self.init_linear_forward_speed_high = rospy.get_param(
            '/myrobot_1/init_linear_forward_speed_high')
        self.init_linear_forward_speed_middle = rospy.get_param(
            '/myrobot_1/init_linear_forward_speed_middle')
        self.init_linear_forward_speed_low = rospy.get_param(
            '/myrobot_1/init_linear_forward_speed_low')
        self.init_linear_turn_speed = rospy.get_param(
            '/myrobot_1/init_linear_turn_speed')
        self.init_angular_speed_high = rospy.get_param(
            '/myrobot_1/init_angular_speed_high')
        self.init_angular_speed_middle = rospy.get_param(
            '/myrobot_1/init_angular_speed_middle')
        self.init_angular_speed_low = rospy.get_param(
            '/myrobot_1/init_angular_speed_low')

        #~~~ 観測空間の設定 ~~~
        # 観測値はロボットの位置、コンテナとの距離
        self.n_observations = rospy.get_param('/myrobot_1/n_observations')
        self.new_ranges = rospy.get_param('/myrobot_1/new_ranges')
        self.min_range = rospy.get_param('/myrobot_1/min_range')
        self.max_distance_value = rospy.get_param('/myrobot_1/max_distance_value')
        self.min_distance_value = rospy.get_param('/myrobot_1/min_distance_value')

        # Get Desired Point to Get
        # ここはコンテナの位置にしたいな        
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/myrobot_1/desired_pose/x")
        self.desired_point.y = rospy.get_param("/myrobot_1/desired_pose/y")
        self.desired_point.z = rospy.get_param("/myrobot_1/desired_pose/z")

        #~~~ We create two arrays based on the binary values that will be assigned ~~~
        # In the discretization method.
        # ==== dist, odom ====
        dist = self.get_dist()
        odom = self.get_odom()

        #~~~ imu, pose, dist, odom data ~~~
        self.dist_frame = dist.header.frame_id
        self.odom_frame = odom.header.frame_id

        #~~~ 観測値のhighとlowを設定する ~~~
        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low, high)

        #~~~ action patter ~~~
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logwarn("ACTION SPACES TYPE===>"+str(self.action_space))
        #~~~ observation pattern ~~~
        rospy.logdebug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))
        rospy.logwarn("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        #~~~ Rewards ~~~
        # 報酬設定をどうするか
        self.distance_close = rospy.get_param("/myrobot_1/distance_close")
        self.distance_little = rospy.get_param("/myrobot_1/distance_little")
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
                        self.init_linear_turn_speed,
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

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose)
        dist = self.get_dist()
        self.pose_before_point = dist.pose

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

        rospy.logdebug("END Set Action ==>"+str(action) + ", NAME="+str(self.last_action))
        rospy.logwarn("END Set Action ==>"+str(action) + ", NAME="+str(self.last_action))

    #~~~ 状態空間の設定 ~~~
    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the odometry so that SumitXL knows where it is.
        # dist
        obs_dist = self.get_dist()
        # r
        r = obs_dist.pose.position.x
        # theta
        theta = obs_dist.pose.position.y
        # phi
        phi = obs_dist.pose.position.z
        # odom
        obs_odom = self.get_odom()
        # x
        x_position = obs_odom.pose.pose.position.x
        # y
        y_position = obs_odom.pose.pose.position.y
        # z
        z_position = obs_odom.pose.pose.position.z

        # We round to only two decimals to avoid very big Observation space
        dist_array = [round(r, 2), round(theta, 2), round(phi, 2)]
        odom_array = [round(x_position, 2), round(y_position, 2), round(z_position, 2)]

        # We only want the X and Y position and the Yaw
        observations = dist_array + odom_array

        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    #~~~ 完了判定 ~~~ 
    def _is_done(self, observations):

        if self._episode_done:
            rospy.logdebug("red can't escape stuck" + str(self._episode_done))
        else:
            rospy.logerr("escape stuck!!!!!!!!!")

            # 現在のロボットの距離を見る
            current_position = PoseStamped()
            current_position.pose.position.x = observations[-2]
            current_position.pose.position.y = observations[-1]
            current_position.pose.position.z = 0.0

            MAX_X = 100.0
            MIN_X = -100.0
            MAX_Y = 100.0
            MIN_Y = -100.0

            # We see if we are outside the Learning Space

            if current_position.pose.position.x <= MAX_X and current_position.pose.position.x > MIN_X:
                if current_position.pose.position.y <= MAX_Y and current_position.pose.position.y > MIN_Y:
                    rospy.logdebug("Red Position is OK ==>["+str(current_position.pose.position.x)+","+str(current_position.pose.position.y)+"]")

                    # We see if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        self._episode_done = True

                else:
                    rospy.logerr("TurtleBot to Far in Y Pos ==>"+str(current_position.pose.position.x))
                    self._episode_done = True
            else:
                rospy.logerr("TurtleBot to Far in X Pos ==>"+str(current_position.pose.position.x))
                self._episode_done = True

        return self._episode_done

    #~~~ 報酬積算値の計算 ~~~
    def _compute_reward(self, observations, done):

        # 現在のロボットの距離を見る
        current_position = PoseStamped()
        current_position.pose.position.x = observations[-2]
        current_position.pose.position.y = observations[-1]
        current_position.pose.position.z = observations[-0]

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point
        
        # 現在のロボットの距離を見る
        current_pose_theta = observations[1]
        current_pose_phi = observations[2]
        theta_difference =  current_pose_theta - self.pose_before_point.position.y
        phi_difference =  current_pose_phi - self.pose_before_point.position.z

        print('{:.3f}'.format(theta_difference))
        print('{:.3f}'.format(current_pose_theta))
        print('{:.3f}'.format(self.pose_before_point.position.y))

        # コンテナとの距離が近づいたら報酬を与える
        if not done:
            if (self.last_action == "FORWARDS_HIGH" or self.last_action == "FORWARDS_MIDDLE" or self.last_action == "FORWARDS_LOW"):
                reward = 0
            else:
                reward = 0

            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward += self.distance_close
            else:
                rospy.logerr("ENCREASE IN DISTANCE BAD")
                reward += -1
            
            if (current_pose_theta < 0.0 and self.pose_before_point.position.y < 0.0):
                if  0.0 < theta_difference:
                    rospy.logwarn("DECREASE IN theta GOOD")
                    reward += self.distance_little
                else:
                    rospy.logerr("ENCREASE IN theta BAD")
                    reward += 0

            elif (0.0 < current_pose_theta and 0.0 < self.pose_before_point.position.y):
                if theta_difference < 0.0:
                    rospy.logwarn("DECREASE IN theta GOOD")
                    reward += self.distance_little
                else:
                    rospy.logerr("ENCREASE IN theta BAD")
                    reward += 0

            elif ((current_pose_theta < 0.0  and 0.0 < self.pose_before_point.position.y) or 
                (0.0 < current_pose_theta and self.pose_before_point.position.y < 0.0)):
                if abs(current_pose_theta) < abs(self.pose_before_point.position.y):
                    rospy.logwarn("DECREASE IN theta GOOD")
                    reward += self.distance_little
                else:
                    rospy.logerr("ENCREASE IN theta BAD")
                    reward += 0

            else:
                reward += 0

            # if 0.5 < theta_difference:
            #     rospy.logwarn("DECREASE IN theta GOOD")
            #     reward += self.distance_little
            # else:
            #     rospy.logerr("ENCREASE IN theta BAD")
            #     reward += 0

            # if 0.5 < phi_difference:
            #     rospy.logwarn("DECREASE IN phi GOOD")
            #     reward += self.distance_little
            # else:
            #     rospy.logerr("ENCREASE IN phi BAD")
            #     reward += 0

        else:
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points
            else:
                reward = -1*self.end_episode_points

        self.previous_distance_from_des_point = distance_from_des_point
        self.pose_before_point.position.y = current_pose_theta
        self.pose_before_point.position.z = current_pose_phi
        
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

