import pandas as pd
import matplotlib.pyplot as plt

# ファイルパスを指定してください
file_path = './rl.csv'

# CSVファイルの読み込み
data = pd.read_csv(file_path)

# データの最初の数行を表示
print(data.head())

# 'reward'カラムの統計情報を表示
print(data['reward'].describe())

# プロットの作成
plt.figure(figsize=(12, 6))

# 'reward'カラムをプロット
plt.plot(data['reward'])

# タイトルと各軸のラベルを設定（フォントサイズを大きく）
plt.title('Reward in reoinforcement learning', fontsize=20)
plt.xlabel('Epsode', fontsize=16)
plt.ylabel('Reward', fontsize=16)

# 軸のメモリの文字サイズを設定
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# グリッドを追加
plt.grid(True)

# プロットを表示
# plt.show()
plt.savefig("rl_rewad.png")
plt.close()
