#!/usr/bin/env python3

import rosparam
import rospkg
import rospy
import os

def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):
    """
    YAMLファイルからROSパラメータをロードする関数。
    :param rospackage_name: ROSパッケージの名前
    :param rel_path_from_package_to_file: パッケージ内のYAMLファイルへの相対パス
    :param yaml_file_name: YAMLファイルの名前
    """
    # RosPackオブジェクトの作成
    rospack = rospkg.RosPack()
    # パッケージのパスを取得
    pkg_path = rospack.get_path(rospackage_name)
    # YAMLファイルのディレクトリを作成
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file) 
    # YAMLファイルのパスを作成
    path_config_file = os.path.join(config_dir, yaml_file_name)
    
    # YAMLファイルからパラメータをロード
    paramlist = rosparam.load_file(path_config_file)
    
    # ロードが成功したかどうかのフラグ
    fin = False
    
    # パラメータをROSパラメータサーバーにアップロード
    for params, ns in paramlist:
        rosparam.upload_params(ns, params)
        fin = True
    
    rospy.logdebug("LoadYamlFileParamsTest: " + str(fin))
