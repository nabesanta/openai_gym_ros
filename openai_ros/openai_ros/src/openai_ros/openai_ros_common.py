#!/usr/bin/env python3

import os
import gym
import git
import sys
import rospy
import rospkg
import subprocess
from .task_envs.task_envs_list import RegisterOpenAI_Ros_Env

def StartOpenAI_ROS_Environment(task_and_robot_environment_name):
    """
    指定されたROS環境を登録し、環境を呼び出す関数。

    :param task_and_robot_environment_name: 登録するROS環境の名前
    :return: gym環境オブジェクト
    """
    # ROS環境名の表示
    rospy.logwarn("Env: {} will be imported".format(task_and_robot_environment_name))
    
    # ROS環境の登録
    result = RegisterOpenAI_Ros_Env(task_env=task_and_robot_environment_name, max_episode_steps=10000)

    # 登録の可否を確認
    rospy.logwarn("Registration Env: {}".format(result))

    # 環境が登録されたらgym.makeで環境の呼び出し
    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
        # 登録した環境の呼び出し
        env = gym.make(task_and_robot_environment_name)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env

class ROSLauncher(object):
    """
    roslaunchを使用してROS環境を起動するクラス。

    :param rospackage_name: 起動するROSパッケージの名前
    :param launch_file_name: 起動するlaunchファイルの名前
    :param ros_ws_abspath: ROSワークスペースの絶対パス
    """

    def __init__(self, rospackage_name, launch_file_name, ros_ws_abspath="/home/maedalab/red_RL"):
        # パッケージ名とlaunchファイル名を格納
        self._rospackage_name = rospackage_name
        self._launch_file_name = launch_file_name

        # rosのパッケージ取得
        self.rospack = rospkg.RosPack()

        # パッケージが存在するか確認
        try:
            pkg_path = self.rospack.get_path(rospackage_name)
            rospy.logdebug("Package FOUND...")
            rospy.logwarn("rospackage path: {}".format(pkg_path))
        except rospkg.common.ResourceNotFound:
            rospy.logwarn("Package NOT FOUND, lets Download it...")
            pkg_path = self.DownloadRepo(package_name=rospackage_name, ros_ws_abspath=ros_ws_abspath)

        # パッケージが指定されたワークスペースに存在するか確認
        if ros_ws_abspath in pkg_path:
            rospy.logdebug("Package FOUND in the correct WS!")
        else:
            rospy.logwarn("Package FOUND in "+pkg_path + ", BUT not in the ws="+ros_ws_abspath+", lets Download it...")
            pkg_path = self.DownloadRepo(package_name=rospackage_name, ros_ws_abspath=ros_ws_abspath)
            rospy.logdebug("Package FOUND from repository!")

        # パッケージが見つかった場合、launchファイルを起動
        if pkg_path:
            rospy.loginfo(">>>>>>>>>>Package found in workspace-->"+str(pkg_path))
            # launchディレクトリとlaunchファイルのパスを作成
            launch_dir = os.path.join(pkg_path, "launch")
            path_launch_file_name = os.path.join(launch_dir, launch_file_name)
            rospy.logwarn("path_launch_file_name=="+str(path_launch_file_name))

            # roslaunchコマンドの作成
            source_env_command = "source "+ros_ws_abspath+"/devel/setup.bash;"
            roslaunch_command = "roslaunch  {0} {1}".format(rospackage_name, launch_file_name)
            command = source_env_command + roslaunch_command
            rospy.logwarn("Launching command="+str(command))

            # sourceコマンドとlaunchファイルを起動
            p = subprocess.Popen(command, shell=True, executable='/bin/bash')

            # プロセスの状態を確認
            state = p.poll()
            if state is None:
                rospy.loginfo("process is running fine")
            elif state < 0:
                rospy.loginfo("Process terminated with error")
            elif state > 0:
                rospy.loginfo("Process terminated without error")

            rospy.loginfo(">>>>>>>>>STARTED Roslaunch-->" + str(self._launch_file_name))
        else:
            assert False, "No Package Path was found for ROS package ==>" + str(rospackage_name)

    def DownloadRepo(self, package_name, ros_ws_abspath):
        """
        指定されたROSパッケージをGitリポジトリからダウンロードする関数。

        :param package_name: ダウンロードするパッケージ名
        :param ros_ws_abspath: ROSワークスペースの絶対パス
        :return: ダウンロードされたパッケージのパス
        """
        commands_to_take_effect = "\nIn a new Shell:::>\ncd "+ros_ws_abspath+"\ncatkin build\nsource devel/setup.bash\nrospack profile\n"
        commands_to_take_effect2 = "\nIn your deeplearning program execute shell catkin_ws:::>\ncd /home/user/catkin_ws\nsource devel/setup.bash\nrospack profile\n"

        ros_ws_src_abspath_src = os.path.join(ros_ws_abspath, "src")
        pkg_path = None
        package_git = None
        package_to_branch_dict = {}

        rospy.logdebug("package_name===>"+str(package_name)+"<===")

        if package_name == "robot_simulation":
            # パッケージ情報の設定（例：SSH URLまたはHTTPS URL）
            url_git_1 = "https://github.com/maedalab/red_ws.git"
            package_git = [url_git_1]
            package_to_branch_dict[url_git_1] = "develop"

        if package_name == "turtlebot_gazebo":
            url_git_1 = "https://bitbucket.org/theconstructcore/turtlebot.git"
            package_git = [url_git_1]
            package_to_branch_dict[url_git_1] = "kinetic-gazebo9"

        elif package_name == "turtlebot3_gazebo":
            url_git_1 = "https://bitbucket.org/theconstructcore/turtlebot3.git"
            package_git = [url_git_1]
            package_to_branch_dict[url_git_1] = "master"

        # サポートされていないパッケージのエラーメッセージ
        else:
            rospy.logerr("Package [ >"+package_name+"< ] is not supported for autodownload, do it manually into >"+str(ros_ws_abspath))
            assert False, "The package " + package_name + " is not supported, please check the package name and the git support in openai_ros_common.py"

        # パッケージのGitリポジトリがサポートされている場合、ダウンロードを実行
        if package_git:
            for git_url in package_git:
                try:
                    rospy.logdebug("Lets download git="+git_url+", in ws="+ros_ws_src_abspath_src)
                    if git_url in package_to_branch_dict:
                        branch_repo_name = package_to_branch_dict[git_url]
                        git.Git(ros_ws_src_abspath_src).clone(git_url, branch=branch_repo_name)
                    else:
                        git.Git(ros_ws_src_abspath_src).clone(git_url)
                    rospy.logdebug("Download git="+git_url+", in ws="+ros_ws_src_abspath_src+"...DONE")
                except git.exc.GitCommandError as e:
                    rospy.logwarn(str(e))
                    rospy.logwarn("The Git "+git_url+" already exists in "+ros_ws_src_abspath_src+", not downloading")

            # パッケージがダウンロードされたか確認
            try:
                pkg_path = self.rospack.get_path(package_name)
                rospy.logwarn("The package "+package_name+" was FOUND by ROS.")

                if ros_ws_abspath in pkg_path:
                    rospy.logdebug("Package FOUND in the correct WS!")
                else:
                    rospy.logwarn("Package FOUND in="+pkg_path+", BUT not in the ws="+ros_ws_abspath)
                    rospy.logerr("IMPORTANT!: You need to execute the following commands and rerun to downloads to take effect.")
                    rospy.logerr(commands_to_take_effect)
                    rospy.logerr(commands_to_take_effect2)
                    sys.exit()

            except rospkg.common.ResourceNotFound:
                rospy.logerr("Package "+package_name+" NOT FOUND by ROS.")
                # ユーザーに再ビルドとソースを要求
                rospy.logerr("IMPORTANT!: You need to execute the following commands and rerun to downloads to take effect.")
                rospy.logerr(commands_to_take_effect)
                rospy.logerr(commands_to_take_effect2)
                sys.exit()

        return pkg_path
