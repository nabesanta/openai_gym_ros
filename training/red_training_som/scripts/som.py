import numpy as np
import matplotlib.pyplot as plt

# 自己組織化マップ（SOM: Self-Organizing Map）
# 入力7つ: [測距センサ2つ、合成加速度・合成角速度、スタック内の位置、コンテナとの位置関係x,y]
# 出力2つ: [x, y]、[-1, 1]の範囲で20分割された値
class SOM:
    def __init__(self, n_side, n_vector, learning_rate=0.9):
        self.n_side = n_side
        self.n_vector = n_vector
        self.learning_rate = learning_rate
        self.n_weight = self.n_side * self.n_side
        self.weight = None
        self.points = np.array([[i // self.n_side, i % self.n_side] for i in range(self.n_weight)])
        self.points = self.points / (self.n_side - 1)
        self.num_bins = 20
        self.sigma = 0.1

    def initialize_weights(self):
        self.weight = np.random.rand(self.n_side, self.n_side, self.n_vector)

    def handle_invalid_values(self, arr):
        """ NaNを0に、infを10に変換する """
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
        return arr

    def update_weights(self, input_vector, t, max_iter):
        alpha = self.learning_rate * (1 - t / max_iter)
        bmu_idx = np.unravel_index(np.argmin(np.linalg.norm(input_vector - self.weight, axis=-1), axis=None),
                                   (self.n_side, self.n_side))
        sigma = self.sigma * np.exp(-t / max_iter)

        for x in range(self.n_side):
            for y in range(self.n_side):
                distance_squared = np.sum((np.array([x, y]) - np.array(bmu_idx)) ** 2)
                influence = np.exp(-distance_squared / (2 * sigma ** 2))
                diff = input_vector - self.weight[x, y]
                diff = self.handle_invalid_values(diff)
                self.weight[x, y] += alpha * influence * diff

    def transform(self, input_vector):
        bmu_idx = np.unravel_index(np.argmin(np.linalg.norm(input_vector - self.weight, axis=-1), axis=None),
                                   (self.n_side, self.n_side))
        bmu_pos = np.array(bmu_idx) / (self.n_side - 1)
        transformed_vec = ((bmu_pos - 0.5) * 2).tolist()
        bin_edges = np.linspace(-1, 1, self.num_bins + 1)
        binned_vec = [bin_edges[np.digitize(val, bin_edges) - 1] for val in transformed_vec]
        binned_vec = [round(val, 1) for val in binned_vec]
        return binned_vec

    def fit(self, X, num_iterations=1000):
        self.initialize_weights()
        for t in range(num_iterations):
            for input_vector in X:
                input_vector = self.handle_invalid_values(input_vector)
                self.update_weights(input_vector, t, num_iterations)

