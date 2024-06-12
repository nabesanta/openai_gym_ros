import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, n_side, learning_rate=0.5):
        self.n_side = n_side
        self.learning_rate = learning_rate
        self.n_weight = self.n_side * self.n_side
        self.weight = None
        self.points = np.array([[i // self.n_side, i % self.n_side] for i in range(self.n_weight)])
        self.points = self.points / (self.n_side - 1)  # 0〜1の範囲に正規化
        self.num_bins = 20  # 20分割

    def initialize_weights(self, n_vector):
        self.weight = np.random.rand(self.n_weight, n_vector)

    def update_weights(self, vec, t, n_learn):
        alpha = 1.0 - float(t) / n_learn
        diff = vec - self.weight
        diff[np.isinf(diff)] = 10  # infを10に置換
        winner_index = np.argmin(np.linalg.norm(diff, axis=1))
        winner_point = self.points[winner_index]
        delta_point = self.points - winner_point
        dist = np.linalg.norm(delta_point, axis=1)
        h = self.learning_rate * alpha * np.exp(-(dist / alpha) ** 2)
        self.weight += np.atleast_2d(h).T * diff
        return winner_index

    def transform(self, input_vector):
        diff = input_vector - self.weight
        diff[np.isinf(diff)] = 10  # infを10に置換
        winner_index = np.argmin(np.linalg.norm(diff, axis=1))
        winner_point = self.points[winner_index]
        transformed_vec = ((winner_point - 0.5) * 2).tolist()  # [-1, 1]の範囲に変換
        
        # -1〜1の範囲を20分割して値を調整
        bin_edges = np.linspace(-1, 1, self.num_bins + 1)
        binned_vec = [bin_edges[np.digitize(val, bin_edges) - 1] for val in transformed_vec]
        binned_vec = [round(val, 1) for val in binned_vec]  # 小数第1位に丸める
        return binned_vec
