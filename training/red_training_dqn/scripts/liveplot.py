#!/usr/bin/env python3

# 必要なライブラリのインポート
import gym
import matplotlib
import matplotlib.pyplot as plt

# 定数の定義
rewards_key = 'episode_rewards'

# ライブプロットのクラス定義
class LivePlot(object):
    def __init__(self, outdir, data_key=rewards_key, line_color='blue'):
        """
        ライブプロットはエピソードごとの報酬またはエピソード長のグラフを描画します。
        Args:
            outdir (str): モニター出力ファイルの場所。グラフのデータを保存するディレクトリ。
            data_key (str, optional): グラフに描画するデータのキー（episode_rewards または episode_lengths）。
            line_color (str, optional): プロットの線の色。
        """
        self.outdir = outdir
        self.data_key = data_key
        self.line_color = line_color

        # スタイリングオプション
        matplotlib.rcParams['toolbar'] = 'None'  # ツールバーを非表示にする
        plt.style.use('ggplot')  # グラフのスタイルを 'ggplot' に設定
        plt.xlabel("Episodes")  # X軸のラベルを設定
        plt.ylabel(data_key)  # Y軸のラベルを設定
        self.fig = plt.gcf()  # 現在の図を取得
        self.fig.canvas.manager.set_window_title('simulation_graph')  # 図のマネージャーを使用してウィンドウタイトルを設定

    def plot(self, env):
        """
        環境からデータを取得し、グラフを更新して表示する。
        Args:
            env (gym.Env): OpenAI Gymの環境オブジェクト。
        """
        # データキーに基づいて適切なデータを取得
        if self.data_key is rewards_key:
            data = gym.wrappers.Monitor.get_episode_rewards(env)  # エピソードごとの報酬を取得
        else:
            data = gym.wrappers.Monitor.get_episode_lengths(env)  # エピソードごとの長さを取得

        plt.plot(data, color=self.line_color)  # データをプロット

        # プロットを表示するために一時停止
        # 将来的にはmatplotlibのアニメーション機能を使用するか、別のライブラリを検討すること
        plt.pause(0.000001)
