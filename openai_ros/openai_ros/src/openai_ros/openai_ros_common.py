#!/usr/bin/env python
import gym
from .task_envs.task_envs_list import RegisterOpenAI_Ros_Env
import roslaunch
import rospy
import rospkg
import os
import git
import sys
import subprocess


def StartOpenAI_ROS_Environment(task_and_robot_environment_name):
    """
    It Does all the stuff that the user would have to do to make it simpler
    for the user.
    This means:
    0) Registers the TaskEnvironment wanted, if it exists in the Task_Envs.
    2) Checks that the workspace of the user has all that is needed for launching this.
    Which means that it will check that the robot spawn launch is there and the worls spawn is there.
    4) Launches the world launch and the robot spawn.
    5) It will import the Gym Env and Make it.
    """
    rospy.logwarn("Env: {} will be imported".format(
        task_and_robot_environment_name))
    result = RegisterOpenAI_Ros_Env(task_env=task_and_robot_environment_name,
                                    max_episode_steps=10000)

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
        env = gym.make(task_and_robot_environment_name)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env


class ROSLauncher(object):
    # ros_ws_abspathは何でも良い
    def __init__(self, rospackage_name, launch_file_name, ros_ws_abspath="/home/maedalab/open_ai_gazebo_custom_model"):

        self._rospackage_name = rospackage_name
        self._launch_file_name = launch_file_name

        self.rospack = rospkg.RosPack()

        # Check Package Exists
        try:
            pkg_path = self.rospack.get_path(rospackage_name)
            rospy.logdebug("Package FOUND...")
        except rospkg.common.ResourceNotFound:
            rospy.logwarn("Package NOT FOUND, lets Download it...")
            pkg_path = self.DownloadRepo(package_name=rospackage_name,
                                         ros_ws_abspath=ros_ws_abspath)

        # Now we check that the Package path is inside the ros_ws_abspath
        # This is to force the system to have the packages in that ws, and not in another.
        if ros_ws_abspath in pkg_path:
            rospy.logdebug("Package FOUND in the correct WS!")
        else:
            rospy.logwarn("Package FOUND in "+pkg_path +
                          ", BUT not in the ws="+ros_ws_abspath+", lets Download it...")
            pkg_path = self.DownloadRepo(package_name=rospackage_name,
                                         ros_ws_abspath=ros_ws_abspath)

        # If the package was found then we launch
        if pkg_path:
            rospy.loginfo(
                ">>>>>>>>>>Package found in workspace-->"+str(pkg_path))
            # pkg_path と "launch"を結合し、pkg_path/launch というパスを作るサンプルプログラム
            # join_path = os.path.join(pkg_path, "launch")
            launch_dir = os.path.join(pkg_path, "launch")
            path_launch_file_name = os.path.join(launch_dir, launch_file_name)

            rospy.logwarn("path_launch_file_name=="+str(path_launch_file_name))

            # ros_ws_abspath = home/maedalab/red_ws
            source_env_command = "source "+ros_ws_abspath+"/devel/setup.bash;"
            # 
            roslaunch_command = "roslaunch  {0} {1}".format(rospackage_name, launch_file_name)
            command = source_env_command+roslaunch_command
            rospy.logwarn("Launching command="+str(command))

            # 別のファイルを起動するためのもの
            p = subprocess.Popen(command, shell=True)

            state = p.poll()
            if state is None:
                rospy.loginfo("process is running fine")
            elif state < 0:
                rospy.loginfo("Process terminated with error")
            elif state > 0:
                rospy.loginfo("Process terminated without error")
            """
            self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(self.uuid)
            self.launch = roslaunch.parent.ROSLaunchParent(
                self.uuid, [path_launch_file_name])
            self.launch.start()
            """


            rospy.loginfo(">>>>>>>>>STARTED Roslaunch-->" +
                          str(self._launch_file_name))
        else:
            assert False, "No Package Path was found for ROS apckage ==>" + \
                str(rospackage_name)

    def DownloadRepo(self, package_name, ros_ws_abspath):
        """
        This has to be installed
        sudo pip install gitpython
        """
        commands_to_take_effect = "\nIn a new Shell:::>\ncd "+ros_ws_abspath + \
            "\ncatkin build\nsource devel/setup.bash\nrospack profile\n"
        commands_to_take_effect2 = "\nIn your deeplearning program execute shell catkin_ws:::>\ncd /home/user/catkin_ws\nsource devel/setup.bash\nrospack profile\n"

        ros_ws_src_abspath_src = os.path.join(ros_ws_abspath, "src")
        pkg_path = None
        # We retrieve the got for the package asked
        package_git = None
        package_to_branch_dict = {}

        rospy.logdebug("package_name===>"+str(package_name)+"<===")

        if  package_name == "turtlebot_gazebo":

                url_git_1 = "https://bitbucket.org/theconstructcore/turtlebot.git"
                package_git = [url_git_1]
                package_to_branch_dict[url_git_1] = "kinetic-gazebo9"

        if  package_name == "turtlebot_gazebo":

            url_git_1 = "https://bitbucket.org/theconstructcore/turtlebot.git"
            package_git = [url_git_1]
            package_to_branch_dict[url_git_1] = "kinetic-gazebo9"


        elif package_name == "turtlebot3_gazebo":

            url_git_1 = "https://bitbucket.org/theconstructcore/turtlebot3.git"
            package_git = [url_git_1]
            package_to_branch_dict[url_git_1] = "master"


        # ADD HERE THE GITs List To Your Simuation

        else:
            rospy.logerr("Package [ >"+package_name +
                         "< ] is not supported for autodownload, do it manually into >"+str(ros_ws_abspath))
            assert False, "The package "++ \
                " is not supported, please check the package name and the git support in openai_ros_common.py"

        # If a Git for the package is supported
        if package_git:
            for git_url in package_git:
                try:
                    rospy.logdebug("Lets download git="+git_url +
                                   ", in ws="+ros_ws_src_abspath_src)
                    if git_url in package_to_branch_dict:
                        branch_repo_name = package_to_branch_dict[git_url]
                        git.Git(ros_ws_src_abspath_src).clone(git_url,branch=branch_repo_name)
                    else:
                        git.Git(ros_ws_src_abspath_src).clone(git_url)

                    rospy.logdebug("Download git="+git_url +
                                   ", in ws="+ros_ws_src_abspath_src+"...DONE")
                except git.exc.GitCommandError as e:
                    rospy.logwarn(str(e))
                    rospy.logwarn("The Git "+git_url+" already exists in " +
                                  ros_ws_src_abspath_src+", not downloading")

            # We check that the package is there
            try:
                pkg_path = self.rospack.get_path(package_name)
                rospy.logwarn("The package "+package_name+" was FOUND by ROS.")

                if ros_ws_abspath in pkg_path:
                    rospy.logdebug("Package FOUND in the correct WS!")
                else:
                    rospy.logwarn("Package FOUND in="+pkg_path +
                                  ", BUT not in the ws="+ros_ws_abspath)
                    rospy.logerr(
                        "IMPORTANT!: You need to execute the following commands and rerun to dowloads to take effect.")
                    rospy.logerr(commands_to_take_effect)
                    rospy.logerr(commands_to_take_effect2)
                    sys.exit()

            except rospkg.common.ResourceNotFound:
                rospy.logerr("Package "+package_name+" NOT FOUND by ROS.")
                # We have to make the user compile and source to make ROS be able to find the new packages
                # TODO: Make this automatic
                rospy.logerr(
                    "IMPORTANT!: You need to execute the following commands and rerun to dowloads to take effect.")
                rospy.logerr(commands_to_take_effect)
                rospy.logerr(commands_to_take_effect2)
                sys.exit()

        return pkg_path
