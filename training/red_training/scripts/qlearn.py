import random

# Q学習の概要
class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}  # 状態と行動のペアに対するQ値を格納する辞書
        self.epsilon = epsilon  # 探索率（ε）
        self.alpha = alpha  # 学習率（α）
        self.gamma = gamma  # 割引率（γ）
        self.actions = actions  # ロボットの行動パターンのリスト

    def getQ(self, state, action):
        """
        状態と行動のペアに対するQ値を取得する。
        辞書に存在しない場合は0.0を返す。
        """
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        """
        Q値を学習・更新する。
        新しいQ値は以下の式で計算される:
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s', a')) - Q(s, a))
        """
        oldv = self.q.get((state, action), 0.0)
        # 常にQ値の更新ルールに従うように修正
        self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        """
        ε-greedy法に基づいて行動を選択する。
        探索率εの確率でランダムに行動を選択し、それ以外の場合は最もQ値の高い行動を選択する。
        """
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        # ランダムな探索を行う場合
        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            q = [q[i] + random.random() * mag - 0.5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        if count > 1:
            # 同じ最大Q値を持つ行動が複数ある場合、ランダムに選択する
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            # 最大Q値を持つ行動を選択する
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        """
        Q学習のメイン部分。
        現在の状態と行動に基づいて得られる最大のQ値を計算し、
        Q(s, a)を更新する。
        """
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        # Q値の更新ルールに基づく修正
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)
