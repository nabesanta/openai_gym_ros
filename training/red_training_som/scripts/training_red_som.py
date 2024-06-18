#!/usr/bin/env python3

import os
import csv
import time
import numpy
import rospy
import som
import qlearn
import liveplot
from functools import reduce
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
    # OK=ノードの初期化
    rospy.init_node('red_training_som', anonymous=True, log_level=rospy.WARN)
    
    # OK=強化学習環境の名前取得
    task_and_robot_environment_name = rospy.get_param('/myrobot_1/task_and_robot_environment_name')
    
    # OK=環境の登録と呼び出し
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    
    # OK=強化学習環境の呼び出し
    rospy.logwarn("Gym environment initialized")
    rospy.logwarn("Starting Learning")
    
    # OK-画面録画(使わない)
    outdir = '~/red_RL/src/openai_gym_ros/results'
    plotter = liveplot.LivePlot(outdir)
    
    # OK=ステップ数の格納変数
    last_time_steps = numpy.ndarray(0)
    
    # OK=Q学習のパラメータ設定
    Alpha = rospy.get_param("/myrobot_1/alpha")
    Gamma = rospy.get_param("/myrobot_1/gamma")
    Epsilon = rospy.get_param("/myrobot_1/epsilon")
    epsilon_discount = rospy.get_param("/myrobot_1/epsilon_discount")
    n_episodes = rospy.get_param("/myrobot_1/n_episodes")
    n_steps = rospy.get_param("/myrobot_1/n_steps")
    
    # OK=Q学習の初期化
    qlearn = qlearn.QLearn(actions=range(env.action_space.n), epsilon=Epsilon, alpha=Alpha, gamma=Gamma)
    initial_epsilon = qlearn.epsilon
    
    # OK=学習の開始時間取得
    start_time = time.time()
    # OK=最高報酬の初期化
    highest_reward = 0
    
    # OK=メイントレーニングループの開始
    for x in range(n_episodes):
        rospy.logdebug("############### START EPISODE => " + str(x))
        # OK=積算報酬の初期化
        cumulated_reward = 0
        done = False

        # OK=探査率の減少, 下限を0.05にする
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        
        # OK=環境のリセットと初期状態の取得
        observation = env.reset()

        # OK=SOMの初期化
        n_side = 20  # 20×20の格子に変更
        n_vector = 11
        som_rl = som.SOM(n_side,n_vector)
        som_rl.initialize_weights()
        observation_vec = som_rl.transform(numpy.array(observation))
        state = ''.join(map(str, observation_vec))

        # OK=初期状態量
        previous_obs = observation

        # CSVファイルに報酬を書き込む
        directory = '~/rl/som/' + str(x+1)
        # ディレクトリが存在しない場合は作成
        os.makedirs(directory, exist_ok=True)

        # 観測値の差分を格納するlist
        diff_array = []
        # SOMにかけた後の状態量を格納するlist
        obs_array = []

        directory = '/media/usb1/som/' + str(x)
        # ディレクトリが存在しない場合は作成
        os.makedirs(directory, exist_ok=True)
        # 各エピソードでロボットをn_stepsテスト
        for i in range(n_steps):
            rospy.logwarn("############### Start Step => " + str(i))
            
            # 現在の状態から次に行う行動を選択
            # 観測・方策
            action = qlearn.chooseAction(state)
            rospy.logwarn("Next action is: %d", action)
            # 行動を実行
            # 観測値、報酬、エピソード終了などを取得
            observation, reward, done, bool_rl, info = env.step(action)
            rospy.logwarn("observation: " + str(observation) + ", reward: " + str(reward))
            
            # 観測値の差分を計算
            diff_obs = numpy.array(observation) - numpy.array(previous_obs)
            diff_array.append(diff_obs)
            with open(os.path.join(directory, 'array.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(diff_obs)

            # 観測値に対してSOMをかける
            winner_index = som_rl.update_weights(numpy.array(observation), i, n_steps)
            observation_vec = som_rl.transform(numpy.array(observation))
            obs_array.append(observation_vec)
            with open(os.path.join(directory, 'obs.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(observation_vec)
            rospy.logwarn("observation_vec: " + str(observation_vec) + ", reward: " + str(reward))

            # 報酬の累積と更新
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
            
            # 観測値から得られる次の行動
            nextState = ''.join(map(str, observation_vec))
            
            # Q学習による価値関数の計算
            # 現在の状態、行動、報酬、次の状態からQ値の更新
            qlearn.learn(state, action, reward, nextState)

            # エピソードが完了しなかったら次のステップへ
            if not done:
                state = nextState
                previous_obs = observation
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

            rospy.logwarn("############### END Step => " + str(i))
            
        # 1エピソードの時間を計算
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(
            "EP: {} - [alpha: {:.2f} - gamma: {:.2f} - epsilon: {:.2f}] - Reward: {} - Time: {:02d}:{:02d}:{:02d}".format(
                x + 1, qlearn.alpha, qlearn.gamma, qlearn.epsilon, cumulated_reward, h, m, s))
        
        with open('/media/usb1/som/reward.csv', 'a') as f:
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
