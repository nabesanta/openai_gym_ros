from ugtm import eGTM
import numpy as np
import altair as alt
import pandas as pd

# 初期データ
X_initial = np.random.randn(10, 7)  # 初期データとして10個の7次元ベクトルを使用

# GTM モデルの初期化
gtm = eGTM(grid_shape=(20, 20))  # GTMにより射影された空間を20×20に設定

# オンライン学習と射影空間の更新
for i in range(len(X_initial)):
    gtm.partial_fit(X_initial[i:i+1])  # オンライン学習（1ステップずつ学習）

# 射影された空間の確認用データ（X_test）
X_test = np.random.randn(50, 7)  # 50個の7次元ベクトルを使用

# X_testをGTMにより射影
transformed = gtm.transform(X_test)