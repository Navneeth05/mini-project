import numpy as np, random

class DAHGA_PSO:
    def __init__(self, pop_size, dim, low, high, max_iter):
        self.pop, self.dim = pop_size, dim
        self.low, self.high = np.array(low), np.array(high)
        self.max_iter = max_iter
        self.pos = np.random.uniform(self.low, self.high, (pop_size, dim))
        self.vel = np.zeros((pop_size, dim))
        self.pbest = self.pos.copy(); self.pbest_scores = np.full(pop_size, -np.inf)
        self.gbest = self.pos[0].copy(); self.gbest_score = -np.inf

    def clip(self, x): return np.clip(x, self.low, self.high)

    def step(self, fitness_fn):
        w_max, w_min = 0.9, 0.35
        c1, c2 = 2.0, 2.0
        mutation_base = 0.15; stagnation = 0; history = []
        for t in range(self.max_iter):
            w = w_max - (w_max - w_min) * (t / self.max_iter)
            for i in range(self.pop):
                score = fitness_fn(self.pos[i])
                if score > self.pbest_scores[i]:
                    self.pbest_scores[i] = score; self.pbest[i] = self.pos[i].copy()
                if score > self.gbest_score:
                    self.gbest_score = score; self.gbest = self.pos[i].copy()
            history.append(self.gbest_score)

            # stagnation check
            if len(history) > 5 and max(history[-5:]) - min(history[-5:]) < 1e-3:
                stagnation += 1
            else: stagnation = 0

            mutation_prob = min(0.6, mutation_base * (1 + stagnation * 0.2))

            for i in range(self.pop):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.vel[i] = w*self.vel[i] + c1*r1*(self.pbest[i]-self.pos[i]) + c2*r2*(self.gbest-self.pos[i])
                self.pos[i] = self.clip(self.pos[i] + self.vel[i])

            # GA-like mutation
            for i in range(self.pop):
                if random.random() < mutation_prob:
                    j = random.randint(0, self.dim - 1)
                    self.pos[i][j] = np.random.uniform(self.low[j], self.high[j])
        return history
