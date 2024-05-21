import random

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {} # pair action and state
        self.epsilon = epsilon  # ε: exploration constant
        self.alpha = alpha      # α: discount constant
        self.gamma = gamma      # γ: discount factor
        self.actions = actions # robot action pattern

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0) # true: state, action, false: 0.0

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        # 報酬の初期値設定
        if oldv is None:
            self.q[(state, action)] = reward
        # 状態と行動における価値の更新
        else:
            self.q[(state, action)] = oldv + self.alpha * (self.gamma * value - oldv)

    # ε-greedy法
    # Q値を最大にする行動を返す
    def chooseAction(self, state, return_q=False):
        # 状態における最善の行動を探す
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        # state2における各行動に対する最大のQ値を取得する
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        # Q値の更新
        # Q(s,a)←Q(s,a)+α⋅[R(s,a)+γ⋅maxa′Q(s′,a′)−Q(s,a)]
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
