#!/usr/bin/env python3

from gym.envs.registration import register
from gym import envs

def RegisterOpenAI_Ros_Env(task_env, max_episode_steps=10000):
    """
    OpenAI Gymの環境を登録する関数。
    :param task_env: タスク環境の名前
    :param max_episode_steps: 最大エピソードステップ数
    :return: 登録結果
    """
    # 登録結果の初期化
    result = True
    
    # タスク環境に応じて登録を行う
    if task_env == 'RedMadoana-v0':
        # RedMadoana環境を登録
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.red.red_madoana:RedMadoanaEnv',
            max_episode_steps=max_episode_steps,
        )
        # RedMadoana環境をインポート
        from openai_ros.task_envs.red import red_madoana

    elif task_env == 'TurtleBot2Maze-v0':
        # TurtleBot2Maze環境を登録
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_maze:TurtleBot2MazeEnv',
            max_episode_steps=max_episode_steps,
        )
        # TurtleBot2Maze環境をインポート
        from openai_ros.task_envs.turtlebot2 import turtlebot2_maze

    # 他のタスク環境も同様に登録

    # 未知のタスク環境の場合
    else:
        result = False

    # 登録の確認
    if result:
        # 登録されたすべての環境を取得
        supported_gym_envs = GetAllRegisteredGymEnvs()
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + \
            str(task_env)

    return result

def GetAllRegisteredGymEnvs():
    """
    登録されたすべてのOpenAI Gym環境のリストを取得する関数。
    """
    all_envs = envs.registry.keys()
    env_ids = list(all_envs)

    return env_ids
