# ここを参照
# https://github.com/keras-rl/keras-rl/tree/master

import gym
import rospy

import pickle
import os
import numpy as np
import random

import tensorflow as tf

from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Activation, Flatten, Dense, Permute
from keras.models import Model
from keras.layers import *
from keras import backend as K

import rl.core

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

# rl.core.Processorの関数を引き継いで変更
# すべて値の変更など
class PendulumProcessorForDQN(rl.core.Processor):
    def __init__(self, enable_image=False, reshape_size=(84, 84)):
        self.shape = reshape_size
        self.enable_image = enable_image
    
    # 観測値の変換
    # 該当なし
    def process_observation(self, observation):
        if not self.enable_image:
            return observation
        img = self._get_rgb_state(observation)
        img = Image.fromarray(img)
        img = img.resize(self.shape).convert('L')  # resize and convert to grayscale
        return np.array(img) / 255

    # actionコード
    # red_madoana内の_set_actionに該当
    def process_action(self, action):
        ACT_ID_TO_VALUE = {
            0: [-2.0], 
            1: [-1.0], 
            2: [0.0], 
            3: [+1.0],
            4: [+2.0],
        }
        return ACT_ID_TO_VALUE[action]

    # 値を範囲内に収める
    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

    # 状態（x,y座標）から対応画像を描画する関数
    def _get_rgb_state(self, state):
        img_size = 84

        h_size = img_size/2.0

        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        dr = ImageDraw.Draw(img)

        # 棒の長さ
        l = img_size/4.0 * 3.0/ 2.0

        # 棒のラインの描写
        dr.line(((h_size - l * state[1], h_size - l * state[0]), (h_size, h_size)), (0, 0, 0), 1)

        # 棒の中心の円を描写（それっぽくしてみた）
        buff = img_size/32.0
        dr.ellipse(((h_size - buff, h_size - buff), (h_size + buff, h_size + buff)), 
                    outline=(0, 0, 0), fill=(255, 0, 0))

        # 画像の一次元化（GrayScale化）とarrayへの変換
        pilImg = img.convert("L")
        img_arr = np.asarray(pilImg)

        # 画像の規格化
        img_arr = img_arr/255.0

        return img_arr

# rl.core.Agentを引き継いで変更を加える
class DQNAgent(rl.core.Agent):
    def __init__(self, 
        input_shape, 
        enable_image_layer,
        nb_actions,
        window_length=4,          # 入力フレーム数
        memory_capacity=1000000,  # 確保するメモリーサイズ
        nb_steps_warmup=50000,    # 初期のメモリー確保用step数(学習しない)
        target_model_update=500,  # target networkのupdate間隔
        action_interval=4,  # アクションを実行する間隔
        train_interval=4,   # 学習間隔
        batch_size=32,      # batch_size
        gamma=0.99,        # Q学習の割引率
        initial_epsilon=1.0,  # ϵ-greedy法の初期値
        final_epsilon=0.1,    # ϵ-greedy法の最終値
        exploration_steps=1000000,  # ϵ-greedy法の減少step数
        **kwargs):
        
        # 
        super(DQNAgent, self).__init__(**kwargs)
        self.compiled = False

        self.input_shape = input_shape
        self.enable_image_layer = enable_image_layer
        self.nb_actions = nb_actions
        self.window_length = window_length
        self.nb_steps_warmup = nb_steps_warmup
        self.target_model_update = target_model_update
        self.action_interval = action_interval
        self.train_interval = train_interval
        self.gamma = gamma
        self.batch_size = batch_size

        self.initial_epsilon = initial_epsilon  
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps
        self.final_epsilon = final_epsilon

        self.memory = ReplayMemory(capacity=memory_capacity)

        self.model = self.build_network()         # Q network
        self.target_model = self.build_network()  # target network
        
        assert memory_capacity > self.batch_size, "Memory capacity is small.(Larger than batch size)"
        assert self.nb_steps_warmup > self.batch_size, "Warmup steps is few.(Larger than batch size)"

    # 変更必要なさそう
    def reset_states(self):
        # 直近のステップのいくつ分、保持するか
        self.recent_observations = [np.zeros(self.input_shape) for _ in range(self.window_length+1)]
        self.recent_action = 0
        self.recent_reward = 0
        self.repeated_action = 0

    # 変更必要なさそう
    def compile(self, optimizer=None, metrics=[]):
        # target networkは更新がないので optimizerとlossは何でもいい
        self.target_model.compile(optimizer='sgd', loss='mse')

        def clipped_error_loss(y_true, y_pred):
            err = y_true - y_pred  # エラー
            L2 = 0.5 * K.square(err)
            L1 = K.abs(err) - 0.5

            # エラーが[-1,1]区間ならL2、それ以外ならL1を選択する。
            loss = tf.where((K.abs(err) < 1.0), L2, L1)   # Keras does not cover where function in tensorflow :-(
            return K.mean(loss)
        self.model.compile(loss=clipped_error_loss, optimizer=optimizer, metrics=metrics)
        
        self.compiled = True

    # 変更必要なさそう
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    # 変更必要なさそう
    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    # 観測値から行動を選択
    # qlearn.chooseActionに該当、変更必要なさそう
    def forward(self, observation):
        # windowサイズ分observationを保存する
        self.recent_observations.append(observation)  # 最後に追加
        self.recent_observations.pop(0)  # 先頭を削除

        # 学習(次の状態が欲しいのでforwardで学習)
        self.forward_train()

        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.step % self.action_interval == 0:
            if self.training:
                
                # ϵ をstepで減少。
                epsilon = self.initial_epsilon - self.step*self.epsilon_step
                if epsilon < self.final_epsilon:
                    epsilon = self.final_epsilon
                
                # ϵ-greedy法
                if epsilon > np.random.uniform(0, 1):
                    # ランダム
                    action = np.random.randint(0, self.nb_actions)
                else:
                    # 現在の状態を取得し、最大Q値から行動を取得。
                    state0 = self.recent_observations[1:]
                    q_values = self.model.predict(np.asarray([state0]), batch_size=1)[0]
                    action = np.argmax(q_values)
            else:
                # 現在の状態を取得し、最大Q値から行動を取得。
                state0 = self.recent_observations[1:]
                q_values = self.model.predict(np.asarray([state0]), batch_size=1)[0]
                action = np.argmax(q_values)

            self.repeated_action = action
        
        self.recent_action = action
        return action

    # qlearn.learnに該当、変更必要なさそう
    def forward_train(self):
        if not self.training:
            return

        self.memory.add((self.recent_observations[:self.window_length], self.recent_action, self.recent_reward, self.recent_observations[1:]))

        # ReplayMemory確保のため一定期間学習しない。
        if self.step <= self.nb_steps_warmup:
            return

        # 学習の更新間隔
        if self.step % self.train_interval != 0:
            return

        batchs = self.memory.sample(self.batch_size)
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        for batch in batchs:
            state0_batch.append(batch[0])
            action_batch.append(batch[1])
            reward_batch.append(batch[2])
            state1_batch.append(batch[3])

        # 更新用に現在のQネットワークを出力(Q network)
        outputs = self.model.predict(np.asarray(state0_batch), self.batch_size)

        # 次の状態のQ値を取得(target_network)
        target_qvals = self.target_model.predict(np.asarray(state1_batch), self.batch_size)

        # Q学習、Q(St,At)=Q(St,At)+α(r+γmax(St+1,At+1)-Q(St,At))
        for i in range(self.batch_size):
            maxq = np.max(target_qvals[i])
            td_error = reward_batch[i] + self.gamma * maxq
            outputs[i][action_batch[i]] = td_error

        # 学習
        self.model.train_on_batch(np.asarray(state0_batch), np.asarray(outputs))


    def backward(self, reward, terminal):
        # 一定間隔でtarget modelに重さをコピー
        if self.step % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        self.recent_reward = reward
        return []

    # ここから下は自作関数
    @property
    def layers(self):
        return self.model.layers[:]

    # NNモデルの作成
    def build_network(self):

        # 入力層(window_length, width, height)
        c = input_ = Input(shape=(self.window_length,) + self.input_shape)

        if self.enable_image_layer:
            c = Permute((2, 3, 1))(c)  # (window,w,h) -> (w,h,window)

            c = Conv2D(32, (8, 8), strides=(4, 4), padding="same")(c)
            c = Activation("relu")(c)
            c = Conv2D(64, (4, 4), strides=(2, 2), padding="same")(c)
            c = Activation("relu")(c)
            c = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(c)
            c = Activation("relu")(c)
        c = Flatten()(c)

        c = Dense(512, activation="relu")(c)

        c = Dense(self.nb_actions, activation="linear")(c)  # 出力層

        return Model(input_, c)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity= capacity
        self.index = 0
        self.memory = []

    def add(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

if __name__ == '__main__':
    # ノードの初期化
    rospy.init_node('red_training_DQN', anonymous=True, log_level=rospy.WARN)
    
    # 強化学習環境の名前取得
    task_and_robot_environment_name = rospy.get_param('/myrobot_1/task_and_robot_environment_name')
    
    # 環境の登録と呼び出し
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

    # 行動数
    nb_actions = 14

    # 取得した値、行動などの変更
    processor = PendulumProcessorForDQN(enable_image=False)

    # 辞書型引数
    args={
        "input_shape": env.observation_space.shape, 
        "enable_image_layer": False, 
        "nb_actions": nb_actions, 
        "window_length": 1,         # 入力フレーム数
        "memory_capacity": 100000,  # 確保するメモリーサイズ
        "nb_steps_warmup": 200,     # 初期のメモリー確保用step数(学習しない)
        "target_model_update": 100, # target networkのupdate間隔
        "action_interval": 1,  # アクションを実行する間隔
        "train_interval": 1,   # 学習する間隔
        "batch_size": 64,   # batch_size
        "gamma": 0.99,     # Q学習の割引率
        "initial_epsilon": 1.0,  # ϵ-greedy法の初期値
        "final_epsilon": 0.1,    # ϵ-greedy法の最終値
        "exploration_steps": 5000,  # ϵ-greedy法の減少step数
    }

    agent = DQNAgent(**args)
    agent.compile(optimizer=Adam())

    # 訓練
    print("--- start ---")
    print("'Ctrl + C' is stop.")
    history = agent.fit(env, nb_steps=50_000, visualize=False, verbose=1)

    # 結果を表示
    plt.subplot(2,1,1)
    plt.plot(history.history["nb_episode_steps"])
    plt.ylabel("step")

    plt.subplot(2,1,2)
    plt.plot(history.history["episode_reward"])
    plt.xlabel("episode")
    plt.ylabel("reward")

    plt.show()

    # 訓練結果を見る
    agent.test(env, nb_episodes=5, visualize=True)