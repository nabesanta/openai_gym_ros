#!/usr/bin/env python3

import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

class ControllersConnection():
    """
    コントローラとの接続と制御を行うクラス
    """
    
    def __init__(self, namespace, controllers_list):
        """
        初期化メソッド
        :param namespace: 名前空間
        :param controllers_list: 制御するコントローラのリスト
        """
        rospy.logwarn("Start Init ControllersConnection")
        self.controllers_list = controllers_list
        self.switch_service_name = '/'+namespace+'/controller_manager/switch_controller'
        self.switch_service = rospy.ServiceProxy(self.switch_service_name, SwitchController)
        rospy.logwarn("END Init ControllersConnection")

    def switch_controllers(self, controllers_on, controllers_off, strictness=1):
        """
        コントローラをオンまたはオフに切り替える
        :param controllers_on: オンにするコントローラのリスト
        :param controllers_off: オフにするコントローラのリスト
        :param strictness: 制御の厳密さ
        :return: 切り替え結果
        """
        rospy.wait_for_service(self.switch_service_name)

        try:
            switch_request_object = SwitchControllerRequest()
            switch_request_object.start_controllers = controllers_on
            switch_request_object.stop_controllers = controllers_off
            switch_request_object.strictness = strictness

            switch_result = self.switch_service(switch_request_object)

            rospy.logdebug("Switch Result ==> " + str(switch_result.ok))
            rospy.logwarn("Switch Result ==> " + str(switch_result.ok))

            return switch_result.ok

        except rospy.ServiceException as e:
            rospy.logerr(self.switch_service_name + " service call failed")
            return None

    def reset_controllers(self):
        """
        コントローラをリセットする
        """
        reset_result = False

        result_off_ok = self.switch_controllers(controllers_on=[],
                                                controllers_off=self.controllers_list)

        rospy.logdebug("Deactivated Controllers")

        if result_off_ok:
            rospy.logdebug("Activating Controllers")
            result_on_ok = self.switch_controllers(controllers_on=self.controllers_list,
                                                    controllers_off=[])
            if result_on_ok:
                rospy.logdebug("Controllers Reset ==> " + str(self.controllers_list))
                rospy.logwarn("Controllers Reset ==> " + str(self.controllers_list))
                reset_result = True
            else:
                rospy.logdebug("result_on_ok ==> " + str(result_on_ok))
        else:
            rospy.logdebug("result_off_ok ==> " + str(result_off_ok))

        return reset_result

    def update_controllers_list(self, new_controllers_list):
        """
        コントローラリストを更新する
        :param new_controllers_list: 新しいコントローラリスト
        """
        self.controllers_list = new_controllers_list
