#!/usr/bin/env python3

import gym
import numpy
import time
import qlearn
from functools import reduce
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym import wrappers
import liveplot
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

#~~~ renderは描画 ~~~ 
def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':

    #~~~ node ~~~
    rospy.init_node('red_madoana_learn',
                    anonymous=True, log_level=rospy.WARN)

    #~~~ Init OpenAI_ROS ENV ~~~
    # task_and_robot_environment_name = 'RedMadoana-v0'
    task_and_robot_environment_name = rospy.get_param(
        '/red/task_and_robot_environment_name')

    #~~~ 環境の登録と呼び出し ~~~
    # gym.makeによって登録された環境
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)

    #~~~ Create the Gym environment ~~~
    # rospy.loginfo() = print + time
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    #~~~ Set the logging system ~~~s
    # rospkg.RosPack(): rosのパッケージパスの取得
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('red_training')
    outdir = pkg_path + '/training_results/before_training.mp4'
    video = VideoRecorder(env, outdir)
    # env = RecordVideo(env, outdir)
    # env = wrappers.Monitor(env, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)
    rospy.loginfo("Monitor Wrapper started")

    #~~~ store step numbers ~~~
    # list: 異なる型を格納可能
    # 配列array: 同じ型のみ格納, 1次元配列のみ
    # 多次元配列 numpy.ndarray(np.array): 同じ型のみ格納, 多次元配列
    last_time_steps = numpy.ndarray(0)

    #~~~ Loads parameters from the ROS param server ~~~
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    # α: 0.1
    Alpha = rospy.get_param("/red/alpha")
    # ε: 0.9
    Epsilon = rospy.get_param("/red/epsilon")
    # γ: 0.7
    Gamma = rospy.get_param("/red/gamma")
    # ε_dis : 0.999
    epsilon_discount = rospy.get_param("/red/epsilon_discount")
    # エピソード数: 10000
    n_episodes = rospy.get_param("/red/n_episodes")
    # ステップ数: 500
    n_steps = rospy.get_param("/red/n_steps")
    # 1ステップ当たりの時間: 0.06
    running_step = rospy.get_param("/red/running_step")

    #~~~ Initialises the algorithm that we are going to use for learning ~~~
    # Q学習のinitialize
    # python range(): 
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           epsilon=Epsilon, alpha=Alpha, gamma=Gamma)
    # Q学習のepsilonにアクセス
    initial_epsilon = qlearn.epsilon

    #~~~ 学習の開始時間取得 ~~~
    start_time = time.time()
    #~~~ 最も高い報酬格納 ~~~
    highest_reward = 0

    # start the main training loop: the one about the episodes to do
    for x in range(n_episodes):
        #~~~ debug ~~~
        rospy.logdebug("############### WALL START EPISODE=>" + str(x))

        #~~~ cumulated_reward: 報酬の積み重ね ~~~
        cumulated_reward = 0

        #~~~ done: ステップの終了 ~~~
        done = False
        if qlearn.epsilon > 0.05:
            #~~~ 割引率を掛けいき, 将来の報酬ほど小さくする ~~~
            qlearn.epsilon *= epsilon_discount

        #~~~ Initialize the gazebo environment and get first state of the robot ~~~
        # ロボットを初期位置に戻す, 報酬の積算値をトピックとしてpublish
        observation = env.reset()
        # video.capture_frame()
        # join: リス内の要素の結合
        # >>> v = ["Hello", "Python"]
        # >>> "".join(v)
        # 'HelloPython'
        # observationはリスト, map関数によってstr型のリストに変換
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for n_steps
        for i in range(n_steps):
            rospy.logwarn("############### Start Step=>" + str(i))

            # env.render()

            #~~~ Pick an action based on the current state ~~~
            # 現在の状態から次に行う適切な行動を抽出する
            action = qlearn.chooseAction(state)
            rospy.logwarn("Next action is:%d", action)

            #~~~ Execute the action in the environment and get feedback ~~~
            # robot_gazebo_env.py
            observation, reward, done, nabe, info = env.step(action)

            #~~~ 観測値と報酬の確認 ~~~
            rospy.logwarn(str(observation) + " " + str(reward))

            #~~~ 積算値を算出, 報酬値の更新 ~~~
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            #~~~ str型のobservation値を格納する ~~~
            nextState = ''.join(map(str, observation))

            #~~~ Make the algorithm learn based on the results ~~~
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logwarn("# State in which we will start next step=>" + str(nextState))

            #~~~ Q学習による価値関数の計算 ~~~ 
            qlearn.learn(state, action, reward, nextState)

            # env._flush(force=True)

            #~~~ 学習の終了を確認する
            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            # raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)

        #~~~ 1エピソードの時間を計算 ~~~
        # 割り算の商と余りを計算
        # int(time.time() - start_time) ÷ 60
        m, s = divmod(int(time.time() - start_time), 60)
        # m ÷ 60
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(n_episodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    # np.araay → list に変換
    l = last_time_steps.tolist()
    # ソートする
    l.sort()

    # print("Parameters: a="+str)
    #~~~ 全体の収益を計算 ~~~
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    #~~~ スコアの計算 ~~~
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # video.close()
    env.close()
