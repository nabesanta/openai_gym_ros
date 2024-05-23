#!/usr/bin/env python3

import csv
import pprint
import gym
import numpy
import time
import qlearn
import liveplot
import rospy
import rospkg
from functools import reduce
from gym import wrappers
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

def render():
    render_skip = 0  # 初めのXエピソードをスキップ
    render_interval = 50  # Yエピソードごとにレンダリングを表示
    render_episodes = 10  # レンダリングの際にZエピソードを表示

    if (x % render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x - render_episodes) % render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':
    # ノードの初期化
    rospy.init_node('red_training', anonymous=True, log_level=rospy.WARN)

    # 強化学習環境の名前取得
    task_and_robot_environment_name = rospy.get_param('/myrobot_1/task_and_robot_environment_name')

    # 環境の登録と呼び出し
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

    rospy.logwarn("Gym environment initialized")
    rospy.logwarn("Starting Learning")

    # 画面録画
    outdir = '~/red_RL/src/openai_gym_ros/results'
    plotter = liveplot.LivePlot(outdir)

    # ステップ数の格納変数
    last_time_steps = numpy.ndarray(0)

    # ROSパラメータの読み込み
    Alpha = rospy.get_param("/myrobot_1/alpha")
    Epsilon = rospy.get_param("/myrobot_1/epsilon")
    Gamma = rospy.get_param("/myrobot_1/gamma")
    epsilon_discount = rospy.get_param("/myrobot_1/epsilon_discount")
    n_episodes = rospy.get_param("/myrobot_1/n_episodes")
    n_steps = rospy.get_param("/myrobot_1/n_steps")
    running_step = rospy.get_param("/myrobot_1/running_step")

    # Q学習の初期化
    qlearn = qlearn.QLearn(actions=range(env.action_space.n), epsilon=Epsilon, alpha=Alpha, gamma=Gamma)
    initial_epsilon = qlearn.epsilon

    # 学習の開始時間取得
    start_time = time.time()
    highest_reward = 0

    # メイントレーニングループの開始
    for x in range(n_episodes):
        rospy.logdebug("############### START EPISODE => " + str(x))
        cumulated_reward = 0
        done = False

        # 探査率の減少
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # 環境のリセットと初期状態の取得
        observation = env.reset()
        state = ''.join(map(str, observation))

        # 各エピソードでロボットをn_stepsテスト
        for i in range(n_steps):
            rospy.logwarn("############### Start Step => " + str(i))

            # 現在の状態から次に行う行動を選択
            action = qlearn.chooseAction(state)
            rospy.logwarn("Next action is: %d", action)

            # 行動を実行し、フィードバックを取得
            observation, reward, done, bool_rl, info = env.step(action)
            rospy.logwarn(str(observation) + " " + str(reward))

            # 報酬の累積と更新
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Q学習による価値関数の計算
            qlearn.learn(state, action, reward, nextState)

            if not done:
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

            rospy.logwarn("############### END Step => " + str(i))

        # エピソードの時間を計算
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(
            "EP: {} - [alpha: {:.2f} - gamma: {:.2f} - epsilon: {:.2f}] - Reward: {} - Time: {:02d}:{:02d}:{:02d}".format(
                x + 1, qlearn.alpha, qlearn.gamma, qlearn.epsilon, cumulated_reward, h, m, s))

        # CSVファイルに報酬を書き込む
        with open('/home/nabesanta/red_RL/src/openai_gym_ros/robots/red_ws/src/csv/odom_to_dist/data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([cumulated_reward])

    rospy.loginfo("|{}|{}|{}|{}|{}|{}| PICTURE |".format(n_episodes, qlearn.alpha, qlearn.gamma, initial_epsilon,
                                                            epsilon_discount, highest_reward))

    # スコアの計算
    l = last_time_steps.tolist()
    l.sort()

    rospy.loginfo("Overall score: {:.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
