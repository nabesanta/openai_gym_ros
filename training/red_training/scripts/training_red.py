#!/usr/bin/env python3

import os
import csv
import time
import numpy
import rospy
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
    # ノードの初期化
    rospy.init_node('red_training', anonymous=True, log_level=rospy.WARN)
    
    # 強化学習環境の名前取得
    task_and_robot_environment_name = rospy.get_param('/myrobot_1/task_and_robot_environment_name')
    
    # 環境の登録と呼び出し
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    
    # 強化学習環境の呼び出し
    rospy.logwarn("Gym environment initialized")
    rospy.logwarn("Starting Learning")
    
    # 画面録画
    outdir = '~/red_RL/src/openai_gym_ros/results'
    plotter = liveplot.LivePlot(outdir)
    
    # ステップ数の格納変数
    last_time_steps = numpy.ndarray(0)
    
    # Q学習のパラメータ設定
    Alpha = rospy.get_param("/myrobot_1/alpha")
    Epsilon = rospy.get_param("/myrobot_1/epsilon")
    Gamma = rospy.get_param("/myrobot_1/gamma")
    epsilon_discount = rospy.get_param("/myrobot_1/epsilon_discount")
    n_episodes = rospy.get_param("/myrobot_1/n_episodes")
    n_steps = rospy.get_param("/myrobot_1/n_steps")
    
    # Q学習の初期化
    qlearn = qlearn.QLearn(actions=range(env.action_space.n), epsilon=Epsilon, alpha=Alpha, gamma=Gamma)
    initial_epsilon = qlearn.epsilon
    
    # 学習の開始時間取得
    start_time = time.time()
    # 最高報酬の初期化
    highest_reward = 0
    
    # メイントレーニングループの開始
    for x in range(n_episodes):
        rospy.logdebug("############### START EPISODE => " + str(x))
        # 積算報酬の初期化
        cumulated_reward = 0
        done = False
        
        # 探査率の減少
        # 各エピソードごとに減少させていく
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        
        # 環境のリセットと初期状態の取得
        observation = env.reset()
        state = ''.join(map(str, observation))

        # 初期状態量
        previous_obs = observation
        
        # CSVファイルに報酬を書き込む
        directory = '/media/usb1/' + str(x+1)
        # ディレクトリが存在しない場合は作成
        os.makedirs(directory, exist_ok=True)
        
        # 観測値の差分を格納するlist
        diff_array = []
        # SOMにかけた後の状態量を格納するlist
        obs_array = []
        
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

            # 報酬の累積と更新
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
            
            # 観測値から得られる次の行動
            nextState = ''.join(map(str, observation))
            
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
        
        with open('/media/usb1/reward.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([cumulated_reward])
        with open(os.path.join(directory, 'diff_obs.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([diff_array])
        with open(os.path.join(directory, 'obs.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([obs_array])


            
    rospy.loginfo("|{}|{}|{}|{}|{}|{}| PICTURE |".format(n_episodes, qlearn.alpha, qlearn.gamma, initial_epsilon,
                                                            epsilon_discount, highest_reward))
    
    # スコアの計算
    l = last_time_steps.tolist()
    l.sort()
    
    rospy.loginfo("Overall score: {:.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
    
    env.close()
