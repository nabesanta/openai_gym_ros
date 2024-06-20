#!/usr/bin/env python3

import os
import gym
import csv
import time
import rospy
import signal
import subprocess
from gym.utils import seeding
from openai_ros.msg import RLExperimentInfo
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection
from std_msgs.msg import Float64
from rospy.rostime import Duration
from rosgraph_msgs.msg import Clock

# 参考URL
# https://github.com/openai/gym/blob/master/gym/core.py
# gym.Envの継承
# seed, reset, render, close, seedを作成する必要がある
# https://qiita.com/ohtaman/items/edcb3b0a2ff9d48a7def
class RobotGazeboEnv(gym.Env):

    def __init__(self, robot_name_space, controllers_list, reset_controls, start_init_physics_parameters, reset_world_or_sim="SIMULATION"):
        """
        初期化関数。主にROSとGazeboの接続を設定する。

        :param robot_name_space: ロボットの名前空間
        :param controllers_list: 使用するコントローラのリスト
        :param reset_controls: コントローラをリセットするかどうか
        :param start_init_physics_parameters: 物理パラメータの初期化を行うかどうか
        :param reset_world_or_sim: シミュレーションをリセットするか、ワールドをリセットするか
        """

        # gazeboの初期化設定
        rospy.logdebug("START init RobotGazeboEnv")
        self.gazebo = GazeboConnection(start_init_physics_parameters, reset_world_or_sim)
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()

        # ROS関連の変数を設定
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        # 報酬値のパブリッシュ
        self.reward_pub = rospy.Publisher('/myrobot_1/openai/reward', RLExperimentInfo, queue_size=1)
        # clockからreal time factorを取得
        self.sim_start_time = None
        self.real_start_time = rospy.Time.now()
        # rospy.Subscriber('/clock', Clock, self.clock_callback)

        # シミュレーションを再開し、コントローラをリセットする
        self.gazebo.unpauseSim()
        if self.reset_controls:
            self.controllers_object.reset_controllers()

        rospy.logdebug("END init RobotGazeboEnv")

    # def clock_callback(self, msg):
    #     if self.sim_start_time is None:
    #         self.sim_start_time = msg.clock
    #     self.sim_time = msg.clock

    # def get_real_time_factor(self):
    #     if self.sim_time is None or self.sim_start_time is None:
    #         return 1.0  # デフォルトのリアルタイムファクター
    #     current_real_time = rospy.Time.now()
    #     elapsed_real_time = (current_real_time - self.real_start_time).to_sec()
    #     elapsed_sim_time = (self.sim_time - self.sim_start_time).to_sec()
    #     if elapsed_real_time > 0:
    #         return elapsed_sim_time / elapsed_real_time
    #     else:
    #         return 1.0

    # gym.Envの必須メソッド
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        シミュレーション上で行動を実行し、報酬を計算する。

        :param action: 実行する行動
        :return: 観測値、報酬、エピソード終了フラグ、デバッグ情報
        """
        rospy.logdebug("START STEP OpenAIROS")

        self.gazebo.unpauseSim()  # シミュレーションを再開
        self._set_action(action)  # 行動を実行
        
        # シミュレーション内の時間で0.5秒スリープさせる
        # sim_time_sleep_simulation = 0.5
        # real_time_factor = self.get_real_time_factor()
        # real_time_sleep = sim_time_sleep_simulation / real_time_factor
        rospy.sleep(rospy.Duration(0.2))
        
        # 状態変化に応じて、ステップ数を管理
        # initial_obs = self._get_obs()  # 初期観測値を取得
        # obs = initial_obs
        # start_time = time.time()
        # max_duration = 10  # 最大待機時間（秒）
        # while not self._is_done(obs) and (time.time() - start_time < max_duration):
        #     time.sleep(0.1)  # 少し待機してから観測値を再取得
        #     obs = self._get_obs()  # 観測値を再取得
        #     # 観測値が変化したか確認
        #     if obs != initial_obs:
        #         rospy.logwarn("Change obs!!!!!!!!!!!!!!!")
        #         break
        
        self.gazebo.pauseSim()    # シミュレーションを停止
        obs, position = self._get_obs()     # 観測値を取得
        # CSVファイルに報酬を書き込む
        directory = '/mnt/usb/pend/' + str(self.episode_num-1)
        # ディレクトリが存在しない場合は作成
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'diff.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([position, action])
        done = self._is_done(obs) # エピソード終了条件をチェック
        info = {}
        reward = self._compute_reward(obs, done)  # 報酬を計算
        self.cumulated_episode_reward += reward  # 積算報酬を更新

        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, done, True, info

    def reset(self):
        """
        環境をリセットする。

        :return: 初期観測値
        """
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        rospy.logwarn("Reseting RobotGazeboEnvironment")
        self._reset_sim()           # シミュレーションをリセット
        self._init_env_variables()  # 環境変数を初期化
        self._update_episode()      # エピソードを更新
        obs, position = self._get_obs()       # 初期観測値を取得
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        エピソードの更新と報酬のパブリッシュを行う。
        """
        rospy.logwarn("PUBLISHING REWARD...")
        self._publish_reward_topic(
            self.cumulated_episode_reward,
            self.episode_num
        )
        rospy.logwarn("PUBLISHING REWARD...DONE=" + str(self.cumulated_episode_reward) + ",EP=" + str(self.episode_num))

        self.episode_num += 1  # エピソード番号を更新
        self.cumulated_episode_reward = 0  # 積算報酬をリセット

    def _publish_reward_topic(self, reward, episode_number=1):
        """
        報酬をROSのトピックにパブリッシュする。

        :param reward: 積算報酬
        :param episode_number: エピソード番号
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    def _render(self, mode="human", close=False):
        """
        シミュレーションのレンダリングを行う。

        :param mode: レンダリングモード
        :param close: レンダリングを終了するかどうか
        """
        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof", "-s", "gzclient"]))
        else:
            self.gzclient_pid = 0

    def _close(self):
        """
        シミュレーションの終了処理を行う。
        """
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if gzclient_count or gzserver_count or roscore_count or rosmaster_count > 0:
            os.wait()

    # 拡張メソッド
    def _reset_sim(self):
        """
        シミュレーションをリセットする。
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls:
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()
        else:
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        rospy.logdebug("RESET SIM END")
        return True

    def _set_init_pose(self):
        """
        初期姿勢を設定する（具体的な実装はサブクラスで行う）。
        """
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        """
        すべてのセンサー、パブリッシャー、および他のシステムが動作しているか確認する（具体的な実装はサブクラスで行う）。
        """
        raise NotImplementedError()

    def _get_obs(self):
        """
        観測値を取得する（具体的な実装はサブクラスで行う）。
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """
        環境変数を初期化する（具体的な実装はサブクラスで行う）。
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """
        指定された行動をシミュレーションに適用する（具体的な実装はサブクラスで行う）。
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """
        エピソードが終了したかどうかを判定する（具体的な実装はサブクラスで行う）。
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """
        観測値に基づいて報酬を計算する（具体的な実装はサブクラスで行う）。
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """
        環境の初期設定を行う（具体的な実装はサブクラスで行う）。
        """
        raise NotImplementedError()
