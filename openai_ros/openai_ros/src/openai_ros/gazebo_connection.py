#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest

class GazeboConnection():
    """
    Gazeboシミュレーションとの接続と制御を行うクラス
    """

    def __init__(self, start_init_physics_parameters, reset_world_or_sim, max_retry=20):
        """
        初期化メソッド
        :param start_init_physics_parameters: 物理パラメータを初期化するかどうか
        :param reset_world_or_sim: ワールドまたはシミュレーションをリセットするか
        :param max_retry: サービスコールの最大リトライ回数
        """
        self._max_retry = max_retry

        # Gazeboのサービスプロキシを作成
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # 物理パラメータの設定サービスを待機
        service_name = '/gazebo/set_physics_properties'
        rospy.logdebug("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        rospy.logdebug("Service Found " + str(service_name))

        self.set_physics = rospy.ServiceProxy(service_name, SetPhysicsProperties)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        self.init_values()

        # シミュレーションの一時停止
        self.pauseSim()

    def pauseSim(self):
        """
        シミュレーションを一時停止する
        """
        rospy.logdebug("PAUSING service found...")
        paused_done = False
        counter = 0
        while not paused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    rospy.logdebug("PAUSING service calling...")
                    self.pause()
                    paused_done = True
                    rospy.logdebug("PAUSING service calling...DONE")
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/pause_physics service call failed")
            else:
                error_message = "Maximum retries done " + str(self._max_retry) + ", please check Gazebo pause service"
                rospy.logerr(error_message)
                assert False, error_message
        rospy.logdebug("PAUSING FINISH")

    def unpauseSim(self):
        """
        シミュレーションを再開する
        """
        rospy.logdebug("UNPAUSING service found...")
        unpaused_done = False
        counter = 0
        while not unpaused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    rospy.logdebug("UNPAUSING service calling...")
                    self.unpause()
                    unpaused_done = True
                    rospy.logdebug("UNPAUSING service calling...DONE")
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/unpause_physics service call failed...Retrying " + str(counter))
            else:
                error_message = "Maximum retries done " + str(self._max_retry) + ", please check Gazebo unpause service"
                rospy.logerr(error_message)
                assert False, error_message
        rospy.logdebug("UNPAUSING FINISH")

    def resetSim(self):
        """
        シミュレーションをリセットする
        システムのリセットオプションに応じてシミュレーションまたはワールドをリセットする
        """
        if self.reset_world_or_sim == "SIMULATION":
            rospy.logerr("SIMULATION RESET")
            self.resetSimulation()
        elif self.reset_world_or_sim == "WORLD":
            rospy.logerr("WORLD RESET")
            self.resetWorld()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logerr("NO RESET SIMULATION SELECTED")
        else:
            rospy.logerr("WRONG Reset Option: " + str(self.reset_world_or_sim))

    def resetSimulation(self):
        """
        シミュレーション全体をリセットする
        """
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        """
        ワールドのみをリセットする
        """
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_world service call failed")

    def init_values(self):
        """
        初期値を設定する
        """
        self.resetSim()

        if self.start_init_physics_parameters:
            rospy.logdebug("Initialising Simulation Physics Parameters")
            self.init_physics_parameters()
        else:
            rospy.logerr("NOT Initialising Simulation Physics Parameters")

    def init_physics_parameters(self):
        """
        シミュレーションの物理パラメータを初期化する
        """
        # シミュレーション内の1ステップの時間: 0.01秒
        # この値を大きくしすぎるとレンダリングが間に合わない
        self._time_step = Float64(0.02)
        # 1秒間の最大更新数
        # 1秒間: 1000回更新
        self._max_update_rate = Float64(1000.0)
        self._real_time_update_rate = Float64(1000.0)

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config = ODEPhysics()
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self.update_gravity_call()

    def update_gravity_call(self):
        """
        重力設定の更新を行う
        """
        self.pauseSim()

        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data
        set_physics_request.max_update_rate = self._max_update_rate.data
        set_physics_request.gravity = self._gravity
        set_physics_request.ode_config = self._ode_config

        rospy.logdebug(str(set_physics_request.gravity))

        result = self.set_physics(set_physics_request)
        rospy.logdebug("Gravity Update Result == " + str(result.success) + ", message == " + str(result.status_message))

        self.unpauseSim()

    def change_gravity(self, x, y, z):
        """
        重力を変更する
        :param x: 重力のx成分
        :param y: 重力のy成分
        :param z: 重力のz成分
        """
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z

        self.update_gravity_call()
