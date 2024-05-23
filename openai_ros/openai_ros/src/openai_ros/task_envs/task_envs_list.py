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

    elif task_env == 'MyTurtleBot2Wall-v0':
        # MyTurtleBot2Wall環境を登録
            register(
                id=task_env,
                entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_wall:TurtleBot2WallEnv',
                max_episode_steps=max_episode_steps,
            )
            # MyTurtleBot2Wall環境をインポート
            from openai_ros.task_envs.turtlebot2 import turtlebot2_wall

    elif task_env == 'MyTurtleBot2WillowGarage-v0':
        # MyTurtleBot2WillowGarage環境を登録
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_willow_garage:TurtleBot2WillowGarageEnv',
            max_episode_steps=max_episode_steps,
        )
        # MyTurtleBot2WillowGarage環境をインポート
        from openai_ros.task_envs.turtlebot2 import turtlebot2_willow_garage

    elif task_env == 'TurtleBot3World-v0':
        # TurtleBot3World環境を登録
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.turtlebot3.turtlebot3_world:TurtleBot3WorldEnv',
            max_episode_steps=max_episode_steps,
        )
        # TurtleBot3World環境をインポート
        from openai_ros.task_envs.turtlebot3 import turtlebot3_world

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
